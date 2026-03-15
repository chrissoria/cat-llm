"""
Async batch inference for CatLLM.

Supports OpenAI, Anthropic, Google, Mistral, and xAI — all offer 50% cost
savings and higher rate limits compared to synchronous API calls.

All five providers follow the same conceptual pattern:
  1. Package all requests as JSONL
  2. Submit a batch job (file upload → job creation, or inline for Anthropic)
  3. Poll until the job reaches a terminal state
  4. Download and parse results
  5. Return a DataFrame identical in format to the synchronous single-model path

Not supported: HuggingFace, Perplexity, Ollama (no batch API).
Ensemble mode: supported. Each model submits its own batch job concurrently.
  Providers without batch API (HuggingFace, Perplexity, Ollama) fall back to
  synchronous calls and are merged in with the batch results.
Not compatible: PDF/image input (text only).
"""

import io
import json
import os
import time

import requests

from ._providers import UnifiedLLMClient
from ._utils import extract_json

# =============================================================================
# Constants
# =============================================================================

BATCH_ENDPOINTS = {
    "openai": {
        "upload":  "https://api.openai.com/v1/files",
        "create":  "https://api.openai.com/v1/batches",
        "status":  "https://api.openai.com/v1/batches/{job_id}",
        "results": "https://api.openai.com/v1/files/{file_id}/content",
    },
    "anthropic": {
        # No file upload — requests are sent inline at job creation
        "create":  "https://api.anthropic.com/v1/messages/batches",
        "status":  "https://api.anthropic.com/v1/messages/batches/{job_id}",
        "results": "https://api.anthropic.com/v1/messages/batches/{job_id}/results",
    },
    "google": {
        "upload":   "https://generativelanguage.googleapis.com/upload/v1beta/files",
        "create":   "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent",
        "status":   "https://generativelanguage.googleapis.com/v1beta/{job_name}",
        "download": "https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download",
    },
    "mistral": {
        "upload":  "https://api.mistral.ai/v1/files",
        "create":  "https://api.mistral.ai/v1/batch/jobs",
        "status":  "https://api.mistral.ai/v1/batch/jobs/{job_id}",
        "results": "https://api.mistral.ai/v1/files/{file_id}/content",
    },
    "xai": {
        "create":  "https://api.x.ai/v1/batches",
        "add":     "https://api.x.ai/v1/batches/{job_id}/requests",
        "status":  "https://api.x.ai/v1/batches/{job_id}",
        "results": "https://api.x.ai/v1/batches/{job_id}/results",
    },
}

UNSUPPORTED_BATCH_PROVIDERS = {"huggingface", "huggingface-together", "perplexity", "ollama"}

# Terminal states per provider
_TERMINAL_STATES = {
    "openai":    {"completed", "failed", "expired", "cancelled"},
    "anthropic": {"ended"},
    "google":    {
        "BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED",
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    },
    "mistral":   {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLATION_REQUESTED"},
    "xai":       {"completed", "failed", "expired", "cancelled"},
}

_SUCCESS_STATES = {
    "openai":    {"completed"},
    "anthropic": {"ended"},   # must check request_counts.failed separately
    "google":    {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"},
    "mistral":   {"SUCCESS"},
    "xai":       {"completed"},
}


# =============================================================================
# Exceptions
# =============================================================================

class BatchJobExpiredError(RuntimeError):
    """Raised when a batch job expires before completing."""
    pass


class BatchJobFailedError(RuntimeError):
    """Raised when a batch job terminates in a failed state."""
    pass


# =============================================================================
# Auth headers
# =============================================================================

def _get_batch_headers(provider: str, api_key: str) -> dict:
    """Return HTTP headers for the given provider's batch API."""
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        headers["anthropic-beta"] = "message-batches-2024-09-24"
    elif provider == "google":
        headers["x-goog-api-key"] = api_key
    elif provider == "mistral":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider == "xai":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# =============================================================================
# JSONL line builders
# =============================================================================

def _build_jsonl_line(provider: str, custom_id: str, payload: dict, model: str) -> dict:
    """
    Wrap a provider payload in the provider's batch JSONL envelope.

    Returns a dict that will be serialized as one JSONL line.
    """
    if provider == "openai":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload,
        }
    elif provider == "anthropic":
        # Anthropic uses "params" key; no file upload needed
        return {
            "custom_id": custom_id,
            "params": payload,
        }
    elif provider == "google":
        # Google batch JSONL format: request payload + metadata with key
        return {
            "request": payload,
            "metadata": {"key": custom_id},
        }
    elif provider == "mistral":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload,
        }
    elif provider == "xai":
        # xAI requests are added one-by-one after batch creation; same OpenAI-compat format
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload,
        }
    raise ValueError(f"Unsupported batch provider: {provider}")


# =============================================================================
# File upload (OpenAI, Google, Mistral)
# =============================================================================

def _upload_jsonl(provider: str, api_key: str, jsonl_bytes: bytes, filename: str = "batch_requests.jsonl") -> str:
    """
    Upload a JSONL file to the provider's files API.

    Returns:
        file_id string used when creating the batch job.
    """
    headers = _get_batch_headers(provider, api_key)
    # Content-Type for multipart upload — remove JSON header
    headers.pop("Content-Type", None)

    if provider == "openai":
        url = BATCH_ENDPOINTS["openai"]["upload"]
        files = {"file": (filename, io.BytesIO(jsonl_bytes), "application/jsonl")}
        data = {"purpose": "batch"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        resp.raise_for_status()
        return resp.json()["id"]

    elif provider == "mistral":
        url = BATCH_ENDPOINTS["mistral"]["upload"]
        files = {"file": (filename, io.BytesIO(jsonl_bytes), "application/octet-stream")}
        data = {"purpose": "batch"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        resp.raise_for_status()
        return resp.json()["id"]

    elif provider == "google":
        # Google Files API: resumable upload
        upload_url = BATCH_ENDPOINTS["google"]["upload"]
        # Step 1: Initiate upload
        init_headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "X-Goog-Upload-Header-Content-Length": str(len(jsonl_bytes)),
            "Content-Type": "application/json",
        }
        init_body = json.dumps({"file": {"display_name": filename}})
        init_resp = requests.post(upload_url, headers=init_headers, data=init_body, timeout=60)
        init_resp.raise_for_status()
        session_url = init_resp.headers.get("X-Goog-Upload-URL")
        if not session_url:
            raise RuntimeError("Google file upload: no session URL returned")
        # Step 2: Upload bytes
        upload_headers = {
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Type": "application/jsonl",
        }
        upload_resp = requests.post(session_url, headers=upload_headers, data=jsonl_bytes, timeout=120)
        upload_resp.raise_for_status()
        file_info = upload_resp.json()
        # Response wraps file metadata under a "file" key: {"file": {"name": "files/abc", ...}}
        file_obj = file_info.get("file", file_info)
        file_name = file_obj.get("name") or file_obj.get("uri")
        if not file_name:
            raise RuntimeError(f"Google file upload: could not extract file name. Response: {file_info}")
        return file_name

    raise ValueError(f"Provider '{provider}' does not use file upload for batch")


# =============================================================================
# Batch job creation
# =============================================================================

def _create_batch_job(
    provider: str,
    api_key: str,
    model: str,
    file_id: str = None,
    requests_list: list = None,
) -> str:
    """
    Create a batch job and return the job ID.

    Args:
        provider: Provider name
        api_key: API key
        model: Model name (used for Mistral and xAI)
        file_id: Uploaded file ID (OpenAI, Google, Mistral)
        requests_list: Inline request list (Anthropic only)

    Returns:
        job_id string for polling
    """
    headers = _get_batch_headers(provider, api_key)

    if provider == "openai":
        url = BATCH_ENDPOINTS["openai"]["create"]
        body = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["id"]

    elif provider == "anthropic":
        url = BATCH_ENDPOINTS["anthropic"]["create"]
        body = {"requests": requests_list}
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["id"]

    elif provider == "google":
        url = BATCH_ENDPOINTS["google"]["create"].format(model=model)
        # Google inline batch: requests are sent in the body (no file upload needed)
        body = {
            "batch": {
                "display_name": f"catllm_batch_{int(time.time())}",
                "input_config": {
                    "requests": {
                        "requests": [
                            {"request": line["request"], "metadata": line["metadata"]}
                            for line in (requests_list or [])
                        ]
                    }
                },
            }
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        # Google returns the job name (e.g. "batches/abc123") — use as job_id
        return result.get("name", result.get("id"))

    elif provider == "mistral":
        url = BATCH_ENDPOINTS["mistral"]["create"]
        body = {
            "input_files": [file_id],
            "model": model,
            "endpoint": "/v1/chat/completions",
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["id"]

    elif provider == "xai":
        # Step 1: Create empty batch
        url = BATCH_ENDPOINTS["xai"]["create"]
        body = {"completion_window": "24h"}
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        job_id = resp.json()["id"]

        # Step 2: Add all requests to the batch
        add_url = BATCH_ENDPOINTS["xai"]["add"].format(job_id=job_id)
        add_resp = requests.post(add_url, headers=headers, json=requests_list, timeout=120)
        add_resp.raise_for_status()
        return job_id

    raise ValueError(f"Unsupported batch provider: {provider}")


# =============================================================================
# Polling
# =============================================================================

def _poll_batch_job(
    provider: str,
    api_key: str,
    job_id: str,
    interval: float = 30.0,
    timeout: float = 86400.0,
) -> dict:
    """
    Poll the batch job until it reaches a terminal state.

    Prints one-line status updates each poll cycle.

    Returns:
        The final status response dict (contains output file ID or job name for result retrieval).

    Raises:
        BatchJobExpiredError: If the job expired or was cancelled.
        BatchJobFailedError: If the job terminated in a failed state.
        TimeoutError: If timeout is reached before the job completes.
    """
    headers = _get_batch_headers(provider, api_key)
    terminal = _TERMINAL_STATES[provider]
    success = _SUCCESS_STATES[provider]

    start = time.time()
    attempt = 0

    if provider == "google":
        status_url = BATCH_ENDPOINTS["google"]["status"].format(job_name=job_id)
    else:
        status_url = BATCH_ENDPOINTS[provider]["status"].format(job_id=job_id)

    while True:
        elapsed = time.time() - start
        if elapsed >= timeout:
            raise TimeoutError(
                f"Batch job '{job_id}' did not complete within {timeout/3600:.1f}h. "
                f"Increase batch_timeout or switch to synchronous mode."
            )

        try:
            resp = requests.get(status_url, headers=headers, timeout=30)
            if resp.status_code >= 500:
                # Server error — back off and retry
                wait = min(60 * (2 ** min(attempt, 4)), 300)
                print(f"  [batch] Server error {resp.status_code}; retrying in {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = min(60 * (2 ** min(attempt, 4)), 300)
            print(f"  [batch] Network error ({e}); retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1
            continue

        attempt = 0
        status_data = resp.json()

        # Extract state string per provider
        if provider == "openai":
            state = status_data.get("status", "")
            counts = status_data.get("request_counts", {})
            progress_str = (
                f"completed={counts.get('completed', '?')} "
                f"failed={counts.get('failed', '?')} "
                f"total={counts.get('total', '?')}"
            )
        elif provider == "anthropic":
            state = status_data.get("processing_status", "")
            counts = status_data.get("request_counts", {})
            progress_str = (
                f"processing={counts.get('processing', '?')} "
                f"succeeded={counts.get('succeeded', '?')} "
                f"errored={counts.get('errored', '?')}"
            )
        elif provider == "google":
            # State lives at metadata.state in the batchGenerateContent response
            state = (status_data.get("metadata", {}).get("state", "")
                     or status_data.get("state", ""))
            progress_str = f"state={state}"
        elif provider == "mistral":
            state = status_data.get("status", "")
            progress_str = (
                f"succeeded={status_data.get('succeeded_requests', '?')} "
                f"failed={status_data.get('failed_requests', '?')} "
                f"total={status_data.get('total_requests', '?')}"
            )
        elif provider == "xai":
            state = status_data.get("status", "")
            counts = status_data.get("request_counts", {})
            progress_str = (
                f"completed={counts.get('completed', '?')} "
                f"failed={counts.get('failed', '?')}"
            )
        else:
            state = ""
            progress_str = ""

        print(f"  [batch] {provider} | elapsed={elapsed:.0f}s | {progress_str} | state={state}")

        if state in terminal:
            if state not in success:
                expired = {"expired", "TIMEOUT_EXCEEDED", "JOB_STATE_CANCELLED", "CANCELLATION_REQUESTED"}
                if state in expired or "cancel" in state.lower() or "timeout" in state.lower():
                    raise BatchJobExpiredError(
                        f"Batch job '{job_id}' expired/was cancelled (state: {state}). "
                        f"Job ID saved above — check provider dashboard for details."
                    )
                raise BatchJobFailedError(
                    f"Batch job '{job_id}' failed (state: {state}). "
                    f"Check the provider dashboard for details."
                )
            return status_data

        time.sleep(interval)


# =============================================================================
# Result download
# =============================================================================

def _download_batch_results(
    provider: str,
    api_key: str,
    job_id: str,
    status_data: dict,
) -> str:
    """
    Download completed batch results as raw JSONL text.

    Args:
        provider: Provider name
        api_key: API key
        job_id: Batch job ID
        status_data: Final status dict from polling (contains output file references)

    Returns:
        Raw JSONL string (one JSON object per line)
    """
    headers = _get_batch_headers(provider, api_key)

    if provider == "openai":
        output_file_id = status_data.get("output_file_id")
        if not output_file_id:
            raise RuntimeError("OpenAI batch: no output_file_id in completed status")
        url = BATCH_ENDPOINTS["openai"]["results"].format(file_id=output_file_id)
        headers_dl = dict(headers)
        headers_dl.pop("Content-Type", None)
        resp = requests.get(url, headers=headers_dl, timeout=120)
        resp.raise_for_status()
        return resp.text

    elif provider == "anthropic":
        url = BATCH_ENDPOINTS["anthropic"]["results"].format(job_id=job_id)
        headers_dl = dict(headers)
        headers_dl.pop("Content-Type", None)
        resp = requests.get(url, headers=headers_dl, timeout=120, stream=True)
        resp.raise_for_status()
        return resp.text

    elif provider == "google":
        # Inline batch results live in the operation response:
        # status_data["response"]["inlinedResponses"]["inlinedResponses"] → list of items
        resp_outer = status_data.get("response", {})
        inlined_wrapper = resp_outer.get("inlinedResponses", {})
        if isinstance(inlined_wrapper, dict):
            inlined = inlined_wrapper.get("inlinedResponses", [])
        elif isinstance(inlined_wrapper, list):
            inlined = inlined_wrapper
        else:
            inlined = []
        if not inlined:
            raise RuntimeError(
                f"Google batch: no inlinedResponses in completed status. "
                f"Status keys: {list(status_data.keys())}, "
                f"response keys: {list(resp_outer.keys()) if isinstance(resp_outer, dict) else resp_outer}"
            )
        # Responses are NOT necessarily in order — use the metadata.key from each item,
        # which preserves the original request key (e.g. "item-42") for correct mapping.
        lines = [
            json.dumps({"key": item.get("metadata", {}).get("key", f"item-{i}"), **item})
            for i, item in enumerate(inlined)
        ]
        return "\n".join(lines)

    elif provider == "mistral":
        output_file_id = status_data.get("output_file")
        if not output_file_id:
            raise RuntimeError("Mistral batch: no output_file in completed status")
        url = BATCH_ENDPOINTS["mistral"]["results"].format(file_id=output_file_id)
        headers_dl = dict(headers)
        headers_dl.pop("Content-Type", None)
        resp = requests.get(url, headers=headers_dl, timeout=120)
        resp.raise_for_status()
        return resp.text

    elif provider == "xai":
        url = BATCH_ENDPOINTS["xai"]["results"].format(job_id=job_id)
        headers_dl = dict(headers)
        headers_dl.pop("Content-Type", None)
        resp = requests.get(url, headers=headers_dl, timeout=120)
        resp.raise_for_status()
        return resp.text

    raise ValueError(f"Unsupported batch provider: {provider}")


# =============================================================================
# Result parsing
# =============================================================================

def _parse_batch_results(
    provider: str,
    raw_results: str,
    custom_id_map: dict,
    client: "UnifiedLLMClient",
    parse_mode: str = "json",
) -> dict:
    """
    Parse the downloaded JSONL results back into per-item strings.

    Args:
        provider: Provider name
        raw_results: Raw JSONL text from the batch results download
        custom_id_map: Dict mapping custom_id string → original item index
        client: UnifiedLLMClient instance (used to call _parse_response())
        parse_mode: "json" (default) runs extract_json() on responses,
            "text" returns raw text as-is (for summarization)

    Returns:
        Dict mapping item_index → (result_str, error_or_None)
        Missing items (job dropped them) get (None, "Missing from batch results").
    """
    parsed_results = {}

    for line in raw_results.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract custom_id and the embedded response object
        if provider == "openai":
            custom_id = data.get("custom_id")
            response_body = data.get("response", {}).get("body")
            error_val = data.get("response", {}).get("error")
            if error_val or response_body is None:
                error_msg = str(error_val) if error_val else "No response body"
                idx = custom_id_map.get(custom_id)
                if idx is not None:
                    parsed_results[idx] = (None, error_msg)
                continue
            raw_text = client._parse_response(response_body)

        elif provider == "anthropic":
            custom_id = data.get("custom_id")
            result = data.get("result", {})
            if result.get("type") != "succeeded":
                error_msg = str(result.get("error", "Request did not succeed"))
                idx = custom_id_map.get(custom_id)
                if idx is not None:
                    parsed_results[idx] = (None, error_msg)
                continue
            raw_text = client._parse_response(result.get("message", {}))

        elif provider == "google":
            # Output JSONL: {"key": "item-0", "response": {generateContent response}}
            # or           {"key": "item-0", "error": {"code": ..., "message": ...}}
            custom_id = data.get("key")
            error_val = data.get("error")
            response_data = data.get("response")
            if error_val or response_data is None:
                error_msg = str(error_val) if error_val else "No response in batch output"
                idx = custom_id_map.get(custom_id)
                if idx is not None:
                    parsed_results[idx] = (None, error_msg)
                continue
            raw_text = client._parse_response(response_data)

        elif provider == "mistral":
            # Mistral batch output mirrors OpenAI: response.body holds the completion
            custom_id = data.get("custom_id")
            response_obj = data.get("response", {})
            error_val = data.get("error") or response_obj.get("error")
            if error_val:
                idx = custom_id_map.get(custom_id)
                if idx is not None:
                    parsed_results[idx] = (None, str(error_val))
                continue
            response_body = response_obj.get("body", response_obj)
            raw_text = client._parse_response(response_body)

        elif provider == "xai":
            custom_id = data.get("custom_id")
            response_body = data.get("response", {}).get("body")
            error_val = data.get("response", {}).get("error")
            if error_val or response_body is None:
                error_msg = str(error_val) if error_val else "No response body"
                idx = custom_id_map.get(custom_id)
                if idx is not None:
                    parsed_results[idx] = (None, error_msg)
                continue
            raw_text = client._parse_response(response_body)

        else:
            continue

        idx = custom_id_map.get(custom_id)
        if idx is None:
            continue

        if parse_mode == "text":
            parsed_results[idx] = (raw_text, None)
        else:
            json_str = extract_json(raw_text)
            parsed_results[idx] = (json_str, None)

    return parsed_results


# =============================================================================
# Per-model helpers (used by both single-model and ensemble batch paths)
# =============================================================================

def _run_one_batch_job(
    cfg: dict,
    items: list,
    prompt_params: dict,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
) -> dict:
    """
    Submit, poll, download, and parse a batch job for one model.
    Returns {item_index: (json_str_or_None, error_or_None)}.
    """
    from .text_functions_ensemble import build_text_classification_prompt

    provider = cfg["provider"]
    api_key = cfg["api_key"]
    model = cfg["model"]

    categories_str = prompt_params["categories_str"]
    survey_question_context = prompt_params.get("survey_question_context", "")
    examples_text = prompt_params.get("examples_text", "")
    chain_of_thought = prompt_params.get("chain_of_thought", False)
    context_prompt = prompt_params.get("context_prompt", False)
    step_back_prompt = prompt_params.get("step_back_prompt", False)
    stepback_insights = prompt_params.get("stepback_insights", {})
    json_schema = prompt_params.get("json_schema")
    creativity = prompt_params.get("creativity")
    thinking_budget = prompt_params.get("thinking_budget", 0)

    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    print(f"\n[batch] Building {len(items)} request(s) for {model} ({provider})...")

    # Step 1: Build per-item payloads and JSONL
    custom_id_map = {}
    jsonl_lines = []
    requests_list = []

    for idx, item in enumerate(items):
        custom_id = f"item-{idx}"
        custom_id_map[custom_id] = idx

        messages = build_text_classification_prompt(
            response_text=str(item) if item is not None else "",
            categories_str=categories_str,
            survey_question_context=survey_question_context,
            examples_text=examples_text,
            chain_of_thought=chain_of_thought,
            context_prompt=context_prompt,
            step_back_prompt=step_back_prompt,
            stepback_insights=stepback_insights,
            model_name=model,
            multi_label=prompt_params.get("multi_label", True),
        )

        payload = client._build_payload(
            messages=messages,
            json_schema=json_schema,
            creativity=creativity,
            thinking_budget=thinking_budget if thinking_budget and thinking_budget > 0 else None,
        )

        line = _build_jsonl_line(provider, custom_id, payload, model)
        jsonl_lines.append(line)
        if provider in ("anthropic", "xai", "google"):
            requests_list.append(line)

    jsonl_bytes = b"\n".join(json.dumps(line).encode("utf-8") for line in jsonl_lines)

    # Step 2: Upload file (OpenAI, Mistral only — Google uses inline requests)
    file_id = None
    if provider in ("openai", "mistral"):
        print(f"[batch] Uploading JSONL ({len(jsonl_bytes)/1024:.1f} KB) to {provider}...")
        file_id = _upload_jsonl(provider, api_key, jsonl_bytes)
        print(f"[batch] File uploaded: {file_id}")

    # Step 3: Create batch job
    print(f"[batch] Creating batch job for {model}...")
    job_id = _create_batch_job(
        provider=provider,
        api_key=api_key,
        model=model,
        file_id=file_id,
        requests_list=requests_list if provider in ("anthropic", "xai", "google") else None,
    )
    print(f"[batch] Job created: {job_id}")
    print(f"[batch] Polling every {batch_poll_interval}s (timeout={batch_timeout/3600:.1f}h)...")

    # Step 4: Poll until complete
    status_data = _poll_batch_job(
        provider=provider,
        api_key=api_key,
        job_id=job_id,
        interval=batch_poll_interval,
        timeout=batch_timeout,
    )
    print(f"[batch] Job complete for {model}.")

    # Step 5: Download results
    print(f"[batch] Downloading results for {model}...")
    raw_results = _download_batch_results(
        provider=provider,
        api_key=api_key,
        job_id=job_id,
        status_data=status_data,
    )

    # Step 6: Parse results
    return _parse_batch_results(
        provider=provider,
        raw_results=raw_results,
        custom_id_map=custom_id_map,
        client=client,
    )


def _run_one_sync_model(
    cfg: dict,
    items: list,
    prompt_params: dict,
) -> dict:
    """
    Classify all items synchronously for one model (fallback for unsupported batch providers).
    Returns {item_index: (json_str_or_None, error_or_None)}.
    """
    from .text_functions_ensemble import build_text_classification_prompt

    provider = cfg["provider"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    json_schema = prompt_params.get("json_schema")
    creativity = prompt_params.get("creativity")
    thinking_budget = prompt_params.get("thinking_budget", 0)

    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)
    item_results = {}

    print(f"\n[batch] Synchronous fallback for {model} ({provider}): {len(items)} item(s)...")

    for idx, item in enumerate(items):
        messages = build_text_classification_prompt(
            response_text=str(item) if item is not None else "",
            categories_str=prompt_params["categories_str"],
            survey_question_context=prompt_params.get("survey_question_context", ""),
            examples_text=prompt_params.get("examples_text", ""),
            chain_of_thought=prompt_params.get("chain_of_thought", False),
            context_prompt=prompt_params.get("context_prompt", False),
            step_back_prompt=prompt_params.get("step_back_prompt", False),
            stepback_insights=prompt_params.get("stepback_insights", {}),
            model_name=model,
            multi_label=prompt_params.get("multi_label", True),
        )
        try:
            raw = client.complete(
                messages=messages,
                json_schema=json_schema,
                creativity=creativity,
                thinking_budget=thinking_budget if thinking_budget and thinking_budget > 0 else None,
            )
            item_results[idx] = (extract_json(raw), None)
        except Exception as e:
            item_results[idx] = (None, str(e))

    return item_results


# =============================================================================
# Main entry point (single-model)
# =============================================================================

def run_batch_classify(
    items: list,
    cfg: dict,
    categories: list,
    prompt_params: dict,
    filename: str = None,
    save_directory: str = None,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
    fail_strategy: str = "partial",
) -> "pd.DataFrame":
    """
    Run batch classification for a single model against a list of text items.

    This is the main entry point called from classify() when batch_mode=True.
    Returns a DataFrame in the same format as the synchronous single-model path.

    Args:
        items: List of text strings to classify
        cfg: Model config dict from prepare_model_configs() (single entry)
        categories: List of category names
        prompt_params: Dict containing prompt-building parameters:
            - categories_str (str)
            - survey_question_context (str)
            - examples_text (str)
            - chain_of_thought (bool)
            - context_prompt (bool)
            - step_back_prompt (bool)
            - stepback_insights (dict)
            - json_schema (dict or None)
            - creativity (float or None)
            - thinking_budget (int)
        filename: Optional CSV filename to save results
        save_directory: Optional directory to save results
        batch_poll_interval: Seconds between poll checks (default 30)
        batch_timeout: Max seconds to wait for job (default 86400 = 24h)
        fail_strategy: "partial" or "strict"

    Returns:
        pd.DataFrame with category_1, category_2, ... columns (same as sync path)
    """
    from .text_functions_ensemble import aggregate_results, build_output_dataframes

    # =========================================================================
    # Steps 1-6: Submit, poll, download, and parse the batch job
    # =========================================================================
    item_results = _run_one_batch_job(
        cfg=cfg,
        items=items,
        prompt_params=prompt_params,
        batch_poll_interval=batch_poll_interval,
        batch_timeout=batch_timeout,
    )

    # =========================================================================
    # Step 7: Build all_results list in the format aggregate_results expects
    # =========================================================================
    model_name = cfg["sanitized_name"]
    all_results = []

    for idx, item in enumerate(items):
        json_str, error = item_results.get(idx, (None, "Missing from batch results"))

        model_results = {model_name: (json_str, error)}
        aggregated = aggregate_results(
            model_results=model_results,
            categories=categories,
            consensus_threshold="unanimous",
            fail_strategy=fail_strategy,
        )

        all_results.append({
            "response": str(item) if item is not None else "",
            "model_results": model_results,
            "aggregated": aggregated,
            "skipped": (item is None or (isinstance(item, float) and __import__("math").isnan(item))),
        })

    # =========================================================================
    # Step 8: Build output DataFrame (reuses existing pipeline)
    # =========================================================================
    return build_output_dataframes(
        all_results=all_results,
        model_configs=[cfg],
        categories=categories,
        filename=filename,
        save_directory=save_directory,
    )


# =============================================================================
# Ensemble batch entry point
# =============================================================================

def run_batch_ensemble_classify(
    items: list,
    model_configs: list,
    categories: list,
    prompt_params_per_model: dict,
    consensus_threshold,
    fail_strategy: str = "partial",
    filename: str = None,
    save_directory: str = None,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
) -> "pd.DataFrame":
    """
    Run batch classification for multiple models concurrently, then merge results.

    Batch-capable providers (openai, anthropic, google, mistral, xai) submit jobs
    concurrently. Unsupported providers (huggingface, perplexity, ollama) fall back
    to synchronous per-item calls and are merged with the batch results.

    Args:
        items: List of text strings to classify
        model_configs: List of model config dicts from prepare_model_configs()
        categories: List of category names
        prompt_params_per_model: Dict mapping model name → prompt_params dict
        consensus_threshold: Agreement threshold (str or float)
        fail_strategy: "partial" or "strict"
        filename: Optional CSV filename to save results
        save_directory: Optional directory to save results
        batch_poll_interval: Seconds between poll checks (default 30)
        batch_timeout: Max seconds to wait for job (default 86400 = 24h)

    Returns:
        pd.DataFrame with consensus columns and per-model columns (same as sync ensemble)
    """
    import math
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .text_functions_ensemble import aggregate_results, build_output_dataframes

    batch_cfgs = [c for c in model_configs if c["provider"] not in UNSUPPORTED_BATCH_PROVIDERS]
    sync_cfgs  = [c for c in model_configs if c["provider"] in UNSUPPORTED_BATCH_PROVIDERS]

    if batch_cfgs:
        print(
            f"\n[batch ensemble] {len(batch_cfgs)} model(s) will use batch API: "
            f"{', '.join(c['model'] for c in batch_cfgs)}"
        )
    if sync_cfgs:
        print(
            f"[batch ensemble] {len(sync_cfgs)} model(s) will use synchronous fallback: "
            f"{', '.join(c['model'] for c in sync_cfgs)}"
        )

    all_model_results = {}

    def _run_cfg(cfg):
        model_key = cfg["sanitized_name"]
        pp = prompt_params_per_model[cfg["model"]]
        if cfg["provider"] in UNSUPPORTED_BATCH_PROVIDERS:
            return model_key, _run_one_sync_model(cfg, items, pp)
        else:
            return model_key, _run_one_batch_job(cfg, items, pp, batch_poll_interval, batch_timeout)

    with ThreadPoolExecutor(max_workers=len(model_configs)) as executor:
        futures = {executor.submit(_run_cfg, cfg): cfg for cfg in model_configs}
        for future in as_completed(futures):
            model_key, result = future.result()
            all_model_results[model_key] = result

    all_results = []
    for idx, item in enumerate(items):
        model_results = {
            cfg["sanitized_name"]: all_model_results[cfg["sanitized_name"]].get(
                idx, (None, "Missing from batch results")
            )
            for cfg in model_configs
        }
        aggregated = aggregate_results(
            model_results=model_results,
            categories=categories,
            consensus_threshold=consensus_threshold,
            fail_strategy=fail_strategy,
        )
        skipped = item is None or (isinstance(item, float) and math.isnan(item))
        all_results.append({
            "response": str(item) if not skipped else "",
            "model_results": model_results,
            "aggregated": aggregated,
            "skipped": skipped,
        })

    return build_output_dataframes(all_results, model_configs, categories, filename, save_directory)


# =============================================================================
# Batch summarization
# =============================================================================

def _run_one_batch_summarize_job(
    cfg: dict,
    items: list,
    prompt_params: dict,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
) -> dict:
    """
    Submit, poll, download, and parse a batch summarization job for one model.
    Returns {item_index: (summary_text_or_None, error_or_None)}.
    """
    from .text_functions_ensemble import build_text_summarization_prompt, build_summary_json_schema

    provider = cfg["provider"]
    api_key = cfg["api_key"]
    model = cfg["model"]

    input_description = prompt_params.get("input_description", "")
    summary_instructions = prompt_params.get("summary_instructions", "")
    max_length = prompt_params.get("max_length")
    focus = prompt_params.get("focus")
    chain_of_thought = prompt_params.get("chain_of_thought", False)
    context_prompt = prompt_params.get("context_prompt", False)
    step_back_prompt = prompt_params.get("step_back_prompt", False)
    stepback_insights = prompt_params.get("stepback_insights", {})
    creativity = prompt_params.get("creativity")

    include_additional = provider != "google"
    json_schema = build_summary_json_schema(include_additional)

    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    print(f"\n[batch] Building {len(items)} summarization request(s) for {model} ({provider})...")

    custom_id_map = {}
    jsonl_lines = []
    requests_list = []

    for idx, item in enumerate(items):
        custom_id = f"item-{idx}"
        custom_id_map[custom_id] = idx

        text = str(item) if item is not None else ""

        messages = build_text_summarization_prompt(
            response_text=text,
            input_description=input_description,
            summary_instructions=summary_instructions,
            max_length=max_length,
            focus=focus,
            chain_of_thought=chain_of_thought,
            context_prompt=context_prompt,
            step_back_prompt=step_back_prompt,
            stepback_insights=stepback_insights,
            model_name=model,
        )

        payload = client._build_payload(
            messages=messages,
            json_schema=json_schema,
            creativity=creativity,
        )

        line = _build_jsonl_line(provider, custom_id, payload, model)
        jsonl_lines.append(line)
        if provider in ("anthropic", "xai", "google"):
            requests_list.append(line)

    jsonl_bytes = b"\n".join(json.dumps(line).encode("utf-8") for line in jsonl_lines)

    # Upload file (OpenAI, Mistral only)
    file_id = None
    if provider in ("openai", "mistral"):
        print(f"[batch] Uploading JSONL ({len(jsonl_bytes)/1024:.1f} KB) to {provider}...")
        file_id = _upload_jsonl(provider, api_key, jsonl_bytes)
        print(f"[batch] File uploaded: {file_id}")

    # Create batch job
    print(f"[batch] Creating batch job for {model}...")
    job_id = _create_batch_job(
        provider=provider,
        api_key=api_key,
        model=model,
        file_id=file_id,
        requests_list=requests_list if provider in ("anthropic", "xai", "google") else None,
    )
    print(f"[batch] Job created: {job_id}")
    print(f"[batch] Polling every {batch_poll_interval}s (timeout={batch_timeout/3600:.1f}h)...")

    # Poll until complete
    status_data = _poll_batch_job(
        provider=provider,
        api_key=api_key,
        job_id=job_id,
        interval=batch_poll_interval,
        timeout=batch_timeout,
    )
    print(f"[batch] Job complete for {model}.")

    # Download results
    print(f"[batch] Downloading results for {model}...")
    raw_results = _download_batch_results(
        provider=provider,
        api_key=api_key,
        job_id=job_id,
        status_data=status_data,
    )

    # Parse results — use text mode since summaries are JSON-wrapped
    # but we still want extract_json since output is {"summary": "..."}
    return _parse_batch_results(
        provider=provider,
        raw_results=raw_results,
        custom_id_map=custom_id_map,
        client=client,
        parse_mode="json",
    )


def _run_one_sync_summarize_model(
    cfg: dict,
    items: list,
    prompt_params: dict,
) -> dict:
    """
    Summarize all items synchronously for one model (fallback for unsupported batch providers).
    Returns {item_index: (json_str_or_None, error_or_None)}.
    """
    from .text_functions_ensemble import build_text_summarization_prompt, build_summary_json_schema

    provider = cfg["provider"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    creativity = prompt_params.get("creativity")

    include_additional = provider != "google"
    json_schema = build_summary_json_schema(include_additional)

    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)
    item_results = {}

    print(f"\n[batch] Synchronous fallback for {model} ({provider}): {len(items)} item(s)...")

    for idx, item in enumerate(items):
        messages = build_text_summarization_prompt(
            response_text=str(item) if item is not None else "",
            input_description=prompt_params.get("input_description", ""),
            summary_instructions=prompt_params.get("summary_instructions", ""),
            max_length=prompt_params.get("max_length"),
            focus=prompt_params.get("focus"),
            chain_of_thought=prompt_params.get("chain_of_thought", False),
            context_prompt=prompt_params.get("context_prompt", False),
            step_back_prompt=prompt_params.get("step_back_prompt", False),
            stepback_insights=prompt_params.get("stepback_insights", {}),
            model_name=model,
        )
        try:
            raw, _err = client.complete(
                messages=messages,
                json_schema=json_schema,
                creativity=creativity,
            )
            item_results[idx] = (extract_json(raw), None)
        except Exception as e:
            item_results[idx] = (None, str(e))

    return item_results


def run_batch_summarize(
    items: list,
    cfg: dict,
    prompt_params: dict,
    filename: str = None,
    save_directory: str = None,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
    fail_strategy: str = "partial",
) -> "pd.DataFrame":
    """
    Run batch summarization for a single model.

    Returns a DataFrame with survey_input, summary, processing_status columns.
    """
    import math
    import pandas as pd
    from .text_functions_ensemble import extract_summary_from_json

    item_results = _run_one_batch_summarize_job(
        cfg=cfg,
        items=items,
        prompt_params=prompt_params,
        batch_poll_interval=batch_poll_interval,
        batch_timeout=batch_timeout,
    )

    rows = []
    for idx, item in enumerate(items):
        json_str, error = item_results.get(idx, (None, "Missing from batch results"))

        text = str(item) if item is not None else ""
        is_skipped = item is None or (isinstance(item, float) and math.isnan(item))

        if is_skipped:
            rows.append({"survey_input": text, "summary": "", "processing_status": "skipped"})
            continue

        if error:
            rows.append({"survey_input": text, "summary": "", "processing_status": "error"})
            continue

        if fail_strategy == "strict" and error:
            rows.append({"survey_input": text, "summary": "", "processing_status": "error"})
            continue

        is_valid, summary_text = extract_summary_from_json(json_str)
        if is_valid and summary_text:
            rows.append({"survey_input": text, "summary": summary_text, "processing_status": "success"})
        else:
            rows.append({"survey_input": text, "summary": "", "processing_status": "error"})

    df = pd.DataFrame(rows)

    if filename:
        import os
        save_path = os.path.join(save_directory, filename) if save_directory else filename
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")

    return df


def run_batch_ensemble_summarize(
    items: list,
    model_configs: list,
    prompt_params_per_model: dict,
    fail_strategy: str = "partial",
    filename: str = None,
    save_directory: str = None,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
    max_retries: int = 3,
) -> "pd.DataFrame":
    """
    Run batch summarization for multiple models concurrently, then synthesize.

    Each model submits its own batch job. After all complete, summaries are
    synthesized into a consensus using _synthesize_summaries().
    """
    import math
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .text_functions_ensemble import extract_summary_from_json, _synthesize_summaries

    batch_cfgs = [c for c in model_configs if c["provider"] not in UNSUPPORTED_BATCH_PROVIDERS]
    sync_cfgs  = [c for c in model_configs if c["provider"] in UNSUPPORTED_BATCH_PROVIDERS]

    if batch_cfgs:
        print(
            f"\n[batch ensemble] {len(batch_cfgs)} model(s) will use batch API: "
            f"{', '.join(c['model'] for c in batch_cfgs)}"
        )
    if sync_cfgs:
        print(
            f"[batch ensemble] {len(sync_cfgs)} model(s) will use synchronous fallback: "
            f"{', '.join(c['model'] for c in sync_cfgs)}"
        )

    all_model_results = {}

    def _run_cfg(cfg):
        model_key = cfg["sanitized_name"]
        pp = prompt_params_per_model[cfg["model"]]
        if cfg["provider"] in UNSUPPORTED_BATCH_PROVIDERS:
            return model_key, _run_one_sync_summarize_model(cfg, items, pp)
        else:
            return model_key, _run_one_batch_summarize_job(cfg, items, pp, batch_poll_interval, batch_timeout)

    with ThreadPoolExecutor(max_workers=len(model_configs)) as executor:
        futures = {executor.submit(_run_cfg, cfg): cfg for cfg in model_configs}
        for future in as_completed(futures):
            model_key, result = future.result()
            all_model_results[model_key] = result

    model_names = [cfg["sanitized_name"] for cfg in model_configs]

    rows = []
    for idx, item in enumerate(items):
        text = str(item) if item is not None else ""
        is_skipped = item is None or (isinstance(item, float) and math.isnan(item))

        if is_skipped:
            row = {"survey_input": text, "summary": "", "processing_status": "skipped"}
            for mn in model_names:
                row[f"summary_{mn}"] = ""
            row["failed_models"] = ""
            rows.append(row)
            continue

        summaries = {}
        errors = []
        for cfg in model_configs:
            mn = cfg["sanitized_name"]
            json_str, error = all_model_results[mn].get(idx, (None, "Missing from batch results"))
            if error:
                summaries[mn] = ""
                errors.append(mn)
            else:
                is_valid, summary_text = extract_summary_from_json(json_str)
                summaries[mn] = summary_text if is_valid else ""
                if not is_valid or not summary_text:
                    errors.append(mn)

        # fail_strategy="strict": blank everything if any model failed
        if fail_strategy == "strict" and errors:
            summaries = {k: "" for k in summaries}

        row = {"survey_input": text}
        for mn in model_names:
            row[f"summary_{mn}"] = summaries.get(mn, "")

        # Synthesize consensus
        valid_summaries = {k: v for k, v in summaries.items() if v}
        if valid_summaries:
            synthesis_cfg = model_configs[0]
            consensus = _synthesize_summaries(
                summaries=valid_summaries,
                original_text=text,
                synthesis_config=synthesis_cfg,
                max_retries=max_retries,
            )
            row["summary"] = consensus
        else:
            row["summary"] = ""

        row["failed_models"] = ",".join(errors) if errors else ""

        if all(not s for s in summaries.values()):
            row["processing_status"] = "error"
        elif any(not s for s in summaries.values()):
            row["processing_status"] = "partial"
        else:
            row["processing_status"] = "success"

        rows.append(row)

    df = pd.DataFrame(rows)

    if filename:
        import os
        save_path = os.path.join(save_directory, filename) if save_directory else filename
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")

    return df
