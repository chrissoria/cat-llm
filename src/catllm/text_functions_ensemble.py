"""
Ensemble text classification functions for CatLLM.

This module provides multi-model ensemble classification using parallel execution.
Multiple LLM models are called simultaneously and results are combined using
majority voting for more robust classification.

MODULE STRUCTURE:
=================
Helper Functions:
    - sanitize_model_name(): Convert model names to column-safe suffixes
    - prepare_model_configs(): Validate model configurations, check Ollama

Shared Utilities (reusable for image/pdf classification):
    - normalize_model_input(): Convert various input formats to list of tuples
    - gather_stepback_insights(): Get step-back insights from each model
    - prepare_json_schemas(): Build JSON schemas per provider
    - aggregate_results(): Majority voting consensus
    - build_output_dataframes(): Build output DataFrames

Text-Specific (swap for image classification):
    - build_text_classification_prompt(): Build prompt for text classification

Main Function:
    - multi_class_ensemble(): Main entry point
        - Supports single model (returns DataFrame) or multiple models (returns dict)
        - Supports categories="auto" for auto-detection

Chain of Verification (CoVe):
============================
CoVe is a 4-step prompting strategy to improve classification accuracy:
    Step 1: Initial classification (existing classify_single)
    Step 2: Generate verification questions about the classification
    Step 3: Answer each verification question (up to 5 questions)
    Step 4: Final corrected classification based on Q&A

Usage: Set chain_of_verification=True in multi_class_ensemble()
Note: CoVe requires ~4x API calls per response. Not recommended for ensemble mode
      due to cost, but supported for single-model usage.
"""

__all__ = ["classify_ensemble", "multi_class_ensemble", "summarize_ensemble"]

import json
import os
import re
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Union

from .text_functions import (
    UnifiedLLMClient,
    detect_provider,
    build_json_schema,
    extract_json,
    validate_classification_json,
    ollama_two_step_classify,
    check_ollama_running,
    check_ollama_model,
    pull_ollama_model,
    _get_stepback_insight,
    explore_common_categories,
)

# PDF utility imports
from .pdf_functions import (
    _load_pdf_files,
    _get_pdf_pages,
    _extract_page_as_image_bytes,
    _extract_page_as_pdf_bytes,
    _extract_page_text,
    _encode_bytes_to_base64,
)

# Image utility imports
from .image_functions import (
    _load_image_files,
    _encode_image,
)


# =============================================================================
# Consensus Threshold Helper
# =============================================================================

def _resolve_consensus_threshold(threshold: Union[str, float, int]) -> float:
    """
    Convert consensus threshold to a numeric value.

    Accepts both string aliases and numeric values for flexibility:
    - String values: "majority" (0.5), "two-thirds" (0.67), "unanimous" (1.0)
    - Numeric values: Any float between 0 and 1

    Args:
        threshold: Either a string alias or numeric value (0-1)

    Returns:
        float: The resolved threshold value

    Examples:
        >>> _resolve_consensus_threshold("majority")
        0.5
        >>> _resolve_consensus_threshold("two-thirds")
        0.67
        >>> _resolve_consensus_threshold(0.75)
        0.75
    """
    if isinstance(threshold, str):
        mapping = {
            "majority": 0.5,
            "two-thirds": 0.67,
            "two_thirds": 0.67,
            "twothirds": 0.67,
            "unanimous": 1.0,
        }
        resolved = mapping.get(threshold.lower().strip())
        if resolved is None:
            valid_options = ", ".join(f'"{k}"' for k in ["majority", "two-thirds", "unanimous"])
            raise ValueError(
                f"Invalid consensus_threshold string: '{threshold}'. "
                f"Valid options: {valid_options}, or a numeric value between 0 and 1."
            )
        return resolved
    else:
        value = float(threshold)
        if not 0 <= value <= 1:
            raise ValueError(f"consensus_threshold must be between 0 and 1, got {value}")
        return value


# =============================================================================
# Test Utilities (for debugging batch retry logic)
# =============================================================================

# Global flag and state for retry testing - set _TEST_FORCE_FAILURE = True to test
_TEST_FORCE_FAILURE = False
_test_attempted_pairs = set()


def _test_should_force_failure(response_text: str, model_name: str) -> bool:
    """
    Test helper: Returns True if this (response, model) pair should be forced to fail.

    Only forces failure on FIRST attempt. Subsequent attempts (retries) will proceed normally.

    Usage: Set _TEST_FORCE_FAILURE = True at module level to enable testing.
    """
    if not _TEST_FORCE_FAILURE:
        return False

    pair_key = (response_text[:50], model_name)
    if pair_key not in _test_attempted_pairs:
        _test_attempted_pairs.add(pair_key)
        print(f"  [TEST] Forcing error for: {model_name} on '{response_text[:30]}...'")
        return True
    return False


def _test_reset():
    """Reset test state between test runs."""
    global _test_attempted_pairs
    _test_attempted_pairs = set()


# =============================================================================
# Input Type Detection
# =============================================================================

# Supported image extensions (from image_functions.py)
_IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.svgz',
    '.avif', '.apng', '.tif', '.tiff', '.bmp', '.heif', '.heic',
    '.ico', '.psd', '.jfif', '.pjpeg', '.pjp', '.jpe'
}


def _detect_input_type(survey_input) -> str:
    """
    Detect if input is text strings, PDF files, or image files.

    Auto-detection logic:
    - If input ends in .pdf → PDF mode
    - If input ends in image extension (.png, .jpg, etc.) → Image mode
    - If input is a directory → Check first file to determine PDF or Image mode
    - Otherwise → Text mode

    Args:
        survey_input: Text strings, PDF paths, image paths, or directory path

    Returns:
        'text', 'pdf', or 'image'
    """
    # Handle single string input
    if isinstance(survey_input, (str, Path)):
        survey_str = str(survey_input)
        ext = os.path.splitext(survey_str)[1].lower()

        # Check for PDF
        if ext == '.pdf':
            return 'pdf'

        # Check for image
        if ext in _IMAGE_EXTENSIONS:
            return 'image'

        # Check if it's a directory (could contain PDFs or images)
        if os.path.isdir(survey_str):
            # Check first file to determine type
            try:
                for f in sorted(os.listdir(survey_str)):
                    f_ext = os.path.splitext(f)[1].lower()
                    if f_ext == '.pdf':
                        return 'pdf'
                    if f_ext in _IMAGE_EXTENSIONS:
                        return 'image'
            except OSError:
                pass
            # Default to PDF for directories (backward compatibility)
            return 'pdf'

        return 'text'

    # Handle list/series input
    if hasattr(survey_input, '__iter__'):
        for item in survey_input:
            if item is not None and not pd.isna(item):
                item_str = str(item)
                ext = os.path.splitext(item_str)[1].lower()
                if ext == '.pdf':
                    return 'pdf'
                if ext in _IMAGE_EXTENSIONS:
                    return 'image'
                # First non-null item is text
                return 'text'

    return 'text'


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_model_name(model: str) -> str:
    """
    Convert model name to a valid column suffix.

    Examples:
        gpt-4o -> gpt_4o
        claude-sonnet-4-5-20250929 -> claude_sonnet_4_5_20250929
        llama3.2:latest -> llama3_2_latest

    Args:
        model: The model name string

    Returns:
        Sanitized string suitable for use in column names
    """
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_').lower()
    return sanitized[:40]  # Truncate to reasonable length


def prepare_model_configs(models: list, auto_download: bool = False) -> list:
    """
    Validate and prepare model configurations.

    Args:
        models: List of tuples (model, provider, api_key)
        auto_download: If True, automatically download missing Ollama models

    Returns:
        List of config dicts with validated settings

    Raises:
        ValueError: If API key missing for non-Ollama provider
        ConnectionError: If Ollama not running when needed
        RuntimeError: If Ollama model not available and auto_download is False
    """
    configs = []
    has_ollama = False
    ollama_checked = False

    for model, provider, api_key in models:
        detected_provider = detect_provider(model, provider)

        if detected_provider == "ollama":
            has_ollama = True
            # Check Ollama running (once)
            if not ollama_checked:
                if not check_ollama_running():
                    raise ConnectionError(
                        "\n" + "="*60 + "\n"
                        "  OLLAMA NOT RUNNING\n"
                        "="*60 + "\n\n"
                        "Ollama must be running to use local models.\n\n"
                        "To start Ollama:\n"
                        "  macOS:   Open the Ollama app, or run 'ollama serve'\n"
                        "  Linux:   Run 'ollama serve' in terminal\n"
                        "  Windows: Open the Ollama app\n\n"
                        + "="*60
                    )
                ollama_checked = True

            # Check model availability
            if not check_ollama_model(model):
                if not pull_ollama_model(model, auto_confirm=auto_download):
                    raise RuntimeError(
                        f"Ollama model '{model}' not available. "
                        f"Run: ollama pull {model}"
                    )
        else:
            # Validate API key exists for cloud providers
            if not api_key:
                raise ValueError(
                    f"API key required for provider '{detected_provider}' (model: {model})"
                )

        configs.append({
            "model": model,
            "provider": detected_provider,
            "api_key": api_key,
            "use_two_step": (detected_provider == "ollama"),
            "sanitized_name": sanitize_model_name(model),
        })

    # Check for duplicate sanitized names
    sanitized_names = [c["sanitized_name"] for c in configs]
    if len(sanitized_names) != len(set(sanitized_names)):
        # Find duplicates and make unique by appending index
        seen = {}
        for cfg in configs:
            name = cfg["sanitized_name"]
            if name in seen:
                seen[name] += 1
                cfg["sanitized_name"] = f"{name}_{seen[name]}"
            else:
                seen[name] = 0

    return configs


def aggregate_results(
    model_results: dict,
    categories: list,
    consensus_threshold: Union[str, float],
    fail_strategy: str,
) -> dict:
    """
    Aggregate results from multiple models using majority voting.

    Args:
        model_results: Dict mapping model name to (json_str, error)
        categories: List of category names
        consensus_threshold: Threshold for majority vote. Can be:
            - "majority": 50% agreement (default)
            - "two-thirds": 67% agreement
            - "unanimous": 100% agreement
            - float: Custom threshold between 0 and 1
        fail_strategy: How to handle failures ("partial" or "strict")

    Returns:
        Dict with per_model results, consensus, agreement scores, and metadata
    """
    # Resolve string thresholds to numeric values
    threshold = _resolve_consensus_threshold(consensus_threshold)
    successful = {}
    failed_models = []

    for model_name, (json_str, error) in model_results.items():
        if error:
            failed_models.append(model_name)
        else:
            try:
                parsed = json.loads(json_str)
                successful[model_name] = parsed
            except json.JSONDecodeError:
                failed_models.append(model_name)

    # Handle failure strategies
    if fail_strategy == "strict" and failed_models:
        return {
            "per_model": {},
            "consensus": {},
            "agreement": {},
            "failed_models": failed_models,
            "error": f"Models failed (strict mode): {failed_models}",
        }

    if not successful:
        return {
            "per_model": {},
            "consensus": {},
            "agreement": {},
            "failed_models": failed_models,
            "error": "All models failed",
        }

    # Calculate consensus via majority vote
    consensus = {}
    agreement_scores = {}
    num_successful = len(successful)

    for i in range(1, len(categories) + 1):
        key = str(i)
        votes = []
        for model_name, parsed in successful.items():
            vote = parsed.get(key, "0")
            # Handle both string and int values
            try:
                votes.append(int(vote))
            except (ValueError, TypeError):
                votes.append(0)

        agreement = sum(votes) / num_successful if num_successful > 0 else 0
        consensus[key] = "1" if agreement >= threshold else "0"
        agreement_scores[key] = round(agreement, 3)

    return {
        "per_model": successful,
        "consensus": consensus,
        "agreement": agreement_scores,
        "failed_models": failed_models,
        "error": None,
    }


# =============================================================================
# Shared Utility Functions
# =============================================================================

def normalize_model_input(
    model: str = None,
    api_key: str = None,
    provider: str = "auto",
    models: list = None,
) -> list:
    """
    Normalize model input to a list of tuples.

    Supports three input formats:
    - Single model: model="gpt-4o", api_key="sk-...", provider="auto"
    - Single tuple: models=("gpt-4o", "openai", "sk-...")
    - List of tuples: models=[("gpt-4o", "openai", "sk-..."), ...]

    Args:
        model: Single model name
        api_key: API key for single model
        provider: Provider for single model (default "auto")
        models: List of tuples or single tuple

    Returns:
        List of tuples: [(model, provider, api_key), ...]

    Raises:
        ValueError: If no model specified
    """
    if models is None and model is not None:
        # Single model mode: model="gpt-4o", api_key="sk-...", provider="auto"
        return [(model, provider, api_key)]
    elif models is not None:
        # Check if it's a single tuple (not a list of tuples)
        if isinstance(models, tuple) and len(models) == 3 and isinstance(models[0], str):
            return [models]
        return models

    raise ValueError(
        "No model specified. Use either:\n"
        "  - Single model: model='gpt-4o', api_key='sk-...'\n"
        "  - Multiple models: models=[('gpt-4o', 'openai', 'sk-...'), ...]"
    )


def gather_stepback_insights(
    model_configs: list,
    survey_question: str,
    creativity: float = None,
) -> dict:
    """
    Gather step-back insights from each model.

    Step-back prompting first asks about underlying factors before classification.

    Args:
        model_configs: List of model configuration dicts
        survey_question: The survey question being analyzed
        creativity: Temperature setting

    Returns:
        Dict mapping model name to (stepback_question, insight) tuples
    """
    if not survey_question:
        raise TypeError(
            "survey_question is required when using step_back_prompt. "
            "Please provide the survey question you are analyzing."
        )

    stepback_question = f'What are the underlying factors or dimensions that explain how people typically answer "{survey_question}"?'

    print("Getting step-back insights for each model...")
    stepback_insights = {}

    for cfg in model_configs:
        if cfg["provider"] != "ollama":
            insight, added = _get_stepback_insight(
                cfg["provider"],
                stepback_question,
                cfg["api_key"],
                cfg["model"],
                creativity
            )
            if added:
                stepback_insights[cfg["model"]] = (stepback_question, insight)

    return stepback_insights


def prepare_json_schemas(
    model_configs: list,
    categories: list,
    use_json_schema: bool = True,
) -> dict:
    """
    Prepare JSON schemas for each model based on provider requirements.

    Args:
        model_configs: List of model configuration dicts
        categories: List of category names
        use_json_schema: Whether to use strict JSON schema

    Returns:
        Dict mapping model name to JSON schema (or None)
    """
    json_schemas = {}

    for cfg in model_configs:
        if use_json_schema:
            # Google doesn't support additionalProperties
            include_additional = (cfg["provider"] != "google")
            json_schemas[cfg["model"]] = build_json_schema(categories, include_additional)
        else:
            json_schemas[cfg["model"]] = None

    return json_schemas


# =============================================================================
# Chain of Verification (CoVe) Functions
# =============================================================================

def build_cove_prompts(original_task: str, response_text: str) -> tuple:
    """
    Build Chain of Verification prompts for the 4-step verification process.

    Args:
        original_task: The original classification prompt/task
        response_text: The survey response being classified

    Returns:
        Tuple of (step2_prompt, step3_prompt, step4_prompt)
    """
    step2_prompt = """You provided this initial categorization:
<<INITIAL_REPLY>>

Original task: {original_task}

Generate a focused list of 3-5 verification questions to fact-check your categorization. Each question should:
- Be concise and specific (one sentence)
- Address a distinct aspect of the categorization
- Be answerable independently

Focus on verifying:
- Whether each category assignment is accurate
- Whether the categories match the criteria in the original task
- Whether there are any logical inconsistencies

Provide only the verification questions as a numbered list.""".format(original_task=original_task)

    step3_prompt = """Answer the following verification question based on the survey response provided.

Survey response: {response_text}

Verification question: <<QUESTION>>

Provide a brief, direct answer (1-2 sentences maximum).

Answer:""".format(response_text=response_text)

    step4_prompt = """Original task: {original_task}
Initial categorization:
<<INITIAL_REPLY>>
Verification questions and answers:
<<VERIFICATION_QA>>
If no categories are present, assign "0" to all categories.
Provide the final corrected categorization in the same JSON format:""".format(original_task=original_task)

    return step2_prompt, step3_prompt, step4_prompt


def _remove_numbering(line: str) -> str:
    """
    Remove numbering/bullets from a line for CoVe question parsing.

    Handles formats like:
    - "1. Question"
    - "1) Question"
    - "- Question"
    - "• Question"
    """
    line = line.strip()
    if line.startswith('- '):
        return line[2:].strip()
    if line.startswith('• '):
        return line[2:].strip()
    if line and line[0].isdigit():
        i = 0
        while i < len(line) and line[i].isdigit():
            i += 1
        if i < len(line) and line[i] in '.)':
            return line[i+1:].strip()
    return line


def run_chain_of_verification(
    client,
    initial_reply: str,
    step2_prompt: str,
    step3_prompt: str,
    step4_prompt: str,
    json_schema: dict,
    creativity: float = None,
    max_retries: int = 5,
) -> str:
    """
    Run the Chain of Verification process.

    This is a 4-step process:
    1. Initial classification (already done, passed as initial_reply)
    2. Generate verification questions
    3. Answer each verification question
    4. Final corrected classification

    Args:
        client: UnifiedLLMClient instance
        initial_reply: The initial JSON classification result
        step2_prompt: Prompt template for generating questions
        step3_prompt: Prompt template for answering questions
        step4_prompt: Prompt template for final classification
        json_schema: JSON schema for the final classification
        creativity: Temperature setting
        max_retries: Maximum retry attempts for each API call

    Returns:
        Final corrected JSON classification string
    """
    # Step 2: Generate verification questions (text response, not JSON)
    step2_filled = step2_prompt.replace("<<INITIAL_REPLY>>", initial_reply)
    questions_reply, err = client.complete(
        messages=[{"role": "user", "content": step2_filled}],
        creativity=creativity,
        force_json=False,  # Text response
        max_retries=max_retries,
    )
    if err:
        return initial_reply  # Fall back to initial reply on error

    # Parse questions
    questions = [
        _remove_numbering(line)
        for line in questions_reply.strip().split('\n')
        if line.strip()
    ]

    # Step 3: Answer each verification question (text responses)
    qa_pairs = []
    for question in questions[:5]:  # Limit to 5 questions
        step3_filled = step3_prompt.replace("<<QUESTION>>", question)
        answer_reply, err = client.complete(
            messages=[{"role": "user", "content": step3_filled}],
            creativity=creativity,
            force_json=False,  # Text response
            max_retries=max_retries,
        )
        if not err:
            qa_pairs.append(f"Q: {question}\nA: {answer_reply.strip()}")

    verification_qa = "\n\n".join(qa_pairs)

    # Step 4: Final corrected categorization (JSON response)
    step4_filled = step4_prompt.replace(
        "<<INITIAL_REPLY>>", initial_reply
    ).replace(
        "<<VERIFICATION_QA>>", verification_qa
    )
    final_reply, err = client.complete(
        messages=[{"role": "user", "content": step4_filled}],
        json_schema=json_schema,
        creativity=creativity,
        max_retries=max_retries,
    )

    if err:
        return initial_reply
    return final_reply


# =============================================================================
# Text-Specific Functions (swap these for image classification)
# =============================================================================

def build_text_classification_prompt(
    response_text: str,
    categories_str: str,
    survey_question_context: str = "",
    examples_text: str = "",
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    stepback_insights: dict = None,
    model_name: str = None,
) -> list:
    """
    Build the classification prompt for a text response.

    This is the text-specific prompt builder. For image classification,
    a different function would be used.

    Args:
        response_text: The text to classify
        categories_str: Formatted string of categories (numbered list)
        survey_question_context: Context about the survey question
        examples_text: Few-shot examples text
        chain_of_thought: Whether to use step-by-step reasoning
        context_prompt: Whether to add expert context prefix
        step_back_prompt: Whether step-back prompting is enabled
        stepback_insights: Dict of step-back insights per model
        model_name: Current model name (for step-back lookup)

    Returns:
        List of message dicts for the LLM
    """
    if chain_of_thought:
        user_prompt = f"""{survey_question_context}

Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

{examples_text}

Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""
    else:
        user_prompt = f"""{survey_question_context}
Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}
{examples_text}
Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""

    # Add context prompt prefix if enabled
    if context_prompt:
        context = """You are an expert researcher in survey data categorization.
Apply multi-label classification and base decisions on explicit and implicit meanings.
When uncertain, prioritize precision over recall.

"""
        user_prompt = context + user_prompt

    # Build messages list
    messages = []

    # Add step-back insight if available for this model
    if step_back_prompt and stepback_insights and model_name in stepback_insights:
        sb_question, sb_insight = stepback_insights[model_name]
        messages.append({"role": "user", "content": sb_question})
        messages.append({"role": "assistant", "content": sb_insight})

    messages.append({"role": "user", "content": user_prompt})

    return messages


# =============================================================================
# Summarization Functions
# =============================================================================

def build_summary_json_schema(include_additional_properties: bool = True) -> dict:
    """
    Build JSON schema for summary output.

    Args:
        include_additional_properties: Whether to include additionalProperties: false.
            Should be False for Google (not supported).

    Returns:
        JSON schema dict for structured summary output
    """
    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the input text"
            }
        },
        "required": ["summary"],
    }
    if include_additional_properties:
        schema["additionalProperties"] = False
    return schema


def build_text_summarization_prompt(
    response_text: str,
    input_description: str = "",
    summary_instructions: str = "",
    max_length: int = None,
    focus: str = None,
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    stepback_insights: dict = None,
    model_name: str = None,
) -> list:
    """
    Build the summarization prompt for a text input.

    Args:
        response_text: The text to summarize
        input_description: Description of what the text contains
        summary_instructions: Specific instructions (e.g., "bullet points", "one sentence")
        max_length: Maximum summary length in words
        focus: What to focus on (e.g., "main arguments", "emotional content")
        chain_of_thought: Whether to use step-by-step reasoning
        context_prompt: Whether to add expert context prefix
        step_back_prompt: Whether step-back prompting is enabled
        stepback_insights: Dict of step-back insights per model
        model_name: Current model name (for step-back lookup)

    Returns:
        List of message dicts for the LLM
    """
    # Build focus instruction if provided
    focus_instruction = ""
    if focus:
        focus_instruction = f", focusing on {focus}"

    # Build description context if provided
    description_context = ""
    if input_description:
        description_context = f"The following text is: {input_description}\n\n"

    # Build length instruction if provided
    length_instruction = ""
    if max_length:
        length_instruction = f"\n\nKeep the summary under {max_length} words."

    # Build custom instructions if provided
    custom_instructions = ""
    if summary_instructions:
        custom_instructions = f"\n\nAdditional instructions: {summary_instructions}"

    if chain_of_thought:
        user_prompt = f"""{description_context}Summarize the following text{focus_instruction}:

"{response_text}"

Let's think step by step:
1. First, identify the main topic or theme
2. Then, extract the key points
3. Finally, synthesize into a concise summary{length_instruction}{custom_instructions}

Provide your answer in JSON format: {{"summary": "your summary here"}}"""
    else:
        user_prompt = f"""{description_context}Summarize the following text{focus_instruction}:

"{response_text}"{length_instruction}{custom_instructions}

Provide your answer in JSON format: {{"summary": "your summary here"}}"""

    # Add context prompt prefix if enabled
    if context_prompt:
        context = """You are an expert at synthesizing key insights from text.
Focus on accuracy, clarity, and identifying the most important themes.
Provide concise summaries that capture essential information.

"""
        user_prompt = context + user_prompt

    # Build messages list
    messages = []

    # Add step-back insight if available for this model
    if step_back_prompt and stepback_insights and model_name in stepback_insights:
        sb_question, sb_insight = stepback_insights[model_name]
        messages.append({"role": "user", "content": sb_question})
        messages.append({"role": "assistant", "content": sb_insight})

    messages.append({"role": "user", "content": user_prompt})

    return messages


def extract_summary_from_json(json_str: str) -> tuple:
    """
    Extract summary from JSON response.

    Args:
        json_str: JSON string containing {"summary": "..."}

    Returns:
        Tuple of (is_valid, summary_text or None)
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "summary" in data:
            summary = data["summary"]
            if isinstance(summary, str) and summary.strip():
                return True, summary.strip()
        return False, None
    except (json.JSONDecodeError, TypeError):
        return False, None


def build_pdf_summarization_prompt(
    page_data: dict,
    input_description: str = "",
    summary_instructions: str = "",
    max_length: int = None,
    focus: str = None,
    provider: str = "openai",
    pdf_mode: str = "image",
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    stepback_insights: dict = None,
    model_name: str = None,
) -> list:
    """
    Build the summarization prompt for a PDF page.

    This is the PDF-specific prompt builder, parallel to build_pdf_classification_prompt()
    but for summarization instead of classification.

    Args:
        page_data: Dict containing:
            - pdf_path: Path to source PDF
            - page_index: Page number (0-indexed)
            - page_label: Label like "document_p1"
            - image_bytes: PNG bytes (for image mode)
            - pdf_bytes: PDF bytes (for native PDF providers)
            - text: Extracted text (for text mode)
        input_description: Description of what the PDF documents contain
        summary_instructions: Specific instructions (e.g., "bullet points")
        max_length: Maximum summary length in words
        focus: What to focus on in the summary
        provider: Provider name for format-specific handling
        pdf_mode: "image", "text", or "both"
        chain_of_thought: Whether to use step-by-step reasoning
        context_prompt: Whether to add expert context prefix
        step_back_prompt: Whether step-back prompting is enabled
        stepback_insights: Dict of step-back insights per model
        model_name: Current model name (for step-back lookup)

    Returns:
        List of message content parts for the LLM (format varies by provider)
    """
    # Build focus instruction if provided
    focus_instruction = ""
    if focus:
        focus_instruction = f", focusing on {focus}"

    # Build examine instruction based on mode
    if pdf_mode == "text":
        examine_instruction = "Examine the following text extracted from a PDF page"
    elif pdf_mode == "both":
        examine_instruction = "Examine the attached PDF page AND the extracted text below"
    else:  # image mode
        examine_instruction = "Examine the attached PDF page"

    # Build length instruction if provided
    length_instruction = ""
    if max_length:
        length_instruction = f"\n\nKeep the summary under {max_length} words."

    # Build custom instructions if provided
    custom_instructions = ""
    if summary_instructions:
        custom_instructions = f"\n\nAdditional instructions: {summary_instructions}"

    if chain_of_thought:
        base_text = f"""You are a document summarization assistant.
Task: {examine_instruction} and provide a concise summary{focus_instruction}.

{f'Document context: {input_description}' if input_description else ''}

Let's analyze step by step:
1. First, identify the main topic or theme of this page
2. Then, extract the key points and important information
3. Finally, synthesize into a concise summary{length_instruction}{custom_instructions}

Provide your answer in JSON format: {{"summary": "your summary here"}}"""
    else:
        base_text = f"""You are a document summarization assistant.
Task: {examine_instruction} and provide a concise summary{focus_instruction}.

{f'Document context: {input_description}' if input_description else ''}{length_instruction}{custom_instructions}

Provide your answer in JSON format: {{"summary": "your summary here"}}"""

    # Add extracted text for text and both modes
    if page_data.get("text") and pdf_mode in ("text", "both"):
        base_text += f"\n\n--- EXTRACTED TEXT FROM PAGE ---\n{page_data['text']}\n--- END OF EXTRACTED TEXT ---"

    # Add context prompt prefix if enabled
    if context_prompt:
        context = """You are an expert at synthesizing key insights from documents.
Focus on accuracy, clarity, and identifying the most important themes.
Provide concise summaries that capture essential information.

"""
        base_text = context + base_text

    # Build messages based on provider and mode
    messages = []

    # Add step-back insight if available
    if step_back_prompt and stepback_insights and model_name in stepback_insights:
        sb_question, sb_insight = stepback_insights[model_name]
        messages.append({"role": "user", "content": sb_question})
        messages.append({"role": "assistant", "content": sb_insight})

    # TEXT-ONLY MODE: No image/PDF attachment
    if pdf_mode == "text":
        messages.append({"role": "user", "content": base_text})
        return messages

    # IMAGE/BOTH MODE: Include visual content
    # Format depends on provider

    if provider in _NATIVE_PDF_PROVIDERS and page_data.get("pdf_bytes"):
        # Anthropic or Google with native PDF
        encoded_pdf = _encode_bytes_to_base64(page_data["pdf_bytes"])

        if provider == "anthropic":
            content = [
                {"type": "text", "text": base_text},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": encoded_pdf
                    }
                }
            ]
            messages.append({"role": "user", "content": content})

        elif provider == "google":
            # Google uses a special format
            content = [
                {"type": "text", "text": base_text},
                {
                    "type": "inline_data",
                    "mime_type": "application/pdf",
                    "data": encoded_pdf
                }
            ]
            messages.append({"role": "user", "content": content})

    elif page_data.get("image_bytes"):
        # Providers requiring image conversion (OpenAI, Mistral, xAI, etc.)
        encoded_image = _encode_bytes_to_base64(page_data["image_bytes"])
        encoded_image_url = f"data:image/png;base64,{encoded_image}"

        content = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": encoded_image_url, "detail": "high"}}
        ]
        messages.append({"role": "user", "content": content})

    else:
        # Fallback to text-only if no visual data available
        messages.append({"role": "user", "content": base_text})

    return messages


# =============================================================================
# PDF-Specific Functions
# =============================================================================

# Provider-specific PDF support info
_NATIVE_PDF_PROVIDERS = {"anthropic", "google"}  # Send native PDF bytes
_IMAGE_PROVIDERS = {"openai", "mistral", "xai", "perplexity", "huggingface"}  # Convert to image
_TEXT_ONLY_PROVIDERS = {"ollama"}  # Text extraction only


def build_pdf_classification_prompt(
    page_data: dict,
    categories_str: str,
    input_description: str = "",
    provider: str = "openai",
    pdf_mode: str = "image",
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    stepback_insights: dict = None,
    model_name: str = None,
    example_json: str = None,
) -> list:
    """
    Build the classification prompt for a PDF page.

    This is the PDF-specific prompt builder, parallel to build_text_classification_prompt().

    Args:
        page_data: Dict containing:
            - pdf_path: Path to source PDF
            - page_index: Page number (0-indexed)
            - page_label: Label like "document_p1"
            - image_bytes: PNG bytes (for image mode)
            - pdf_bytes: PDF bytes (for native PDF providers)
            - text: Extracted text (for text mode)
        categories_str: Formatted string of categories (numbered list)
        input_description: Description of what the PDF documents contain
        provider: Provider name for format-specific handling
        pdf_mode: "image", "text", or "both"
        chain_of_thought: Whether to use step-by-step reasoning
        context_prompt: Whether to add expert context prefix
        step_back_prompt: Whether step-back prompting is enabled
        stepback_insights: Dict of step-back insights per model
        model_name: Current model name (for step-back lookup)
        example_json: Example JSON output format

    Returns:
        List of message content parts for the LLM (format varies by provider)
    """
    # Build the base text prompt
    if pdf_mode == "text":
        examine_instruction = "Examine the following text extracted from a PDF page"
    elif pdf_mode == "both":
        examine_instruction = "Examine the attached PDF page AND the extracted text below"
    else:  # image mode
        examine_instruction = "Examine the attached PDF page"

    if chain_of_thought:
        base_text = f"""You are a document-tagging assistant.
Task: {examine_instruction} and decide, **for each category below**, whether it is PRESENT (1) or NOT PRESENT (0).

{f'Document page is expected to contain: {input_description}' if input_description else ''}

Categories:
{categories_str}

Let's analyze step by step:
1. First, identify the key content elements in the document page
2. Then, match each element to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Output format: Respond with **only** a JSON object whose keys are the quoted category numbers ('1', '2', ...) and whose values are 1 or 0. No additional keys, comments, or text.

{f'Example JSON format: {example_json}' if example_json else ''}"""
    else:
        base_text = f"""You are a document-tagging assistant.
Task: {examine_instruction} and decide, **for each category below**, whether it is PRESENT (1) or NOT PRESENT (0).

{f'Document page is expected to contain: {input_description}' if input_description else ''}

Categories:
{categories_str}

Output format: Respond with **only** a JSON object whose keys are the quoted category numbers ('1', '2', ...) and whose values are 1 or 0. No additional keys, comments, or text.

{f'Example JSON format: {example_json}' if example_json else ''}"""

    # Add extracted text for text and both modes
    if page_data.get("text") and pdf_mode in ("text", "both"):
        base_text += f"\n\n--- EXTRACTED TEXT FROM PAGE ---\n{page_data['text']}\n--- END OF EXTRACTED TEXT ---"

    # Add context prompt prefix if enabled
    if context_prompt:
        context = """You are an expert document analyst specializing in page categorization.
Apply multi-label classification based on explicit and implicit content cues.
When uncertain, prioritize precision over recall.

"""
        base_text = context + base_text

    # Build messages based on provider and mode
    messages = []

    # Add step-back insight if available
    if step_back_prompt and stepback_insights and model_name in stepback_insights:
        sb_question, sb_insight = stepback_insights[model_name]
        messages.append({"role": "user", "content": sb_question})
        messages.append({"role": "assistant", "content": sb_insight})

    # TEXT-ONLY MODE: No image/PDF attachment
    if pdf_mode == "text":
        messages.append({"role": "user", "content": base_text})
        return messages

    # IMAGE/BOTH MODE: Include visual content
    # Format depends on provider

    if provider in _NATIVE_PDF_PROVIDERS and page_data.get("pdf_bytes"):
        # Anthropic or Google with native PDF
        encoded_pdf = _encode_bytes_to_base64(page_data["pdf_bytes"])

        if provider == "anthropic":
            content = [
                {"type": "text", "text": base_text},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": encoded_pdf
                    }
                }
            ]
            messages.append({"role": "user", "content": content})

        elif provider == "google":
            # Google uses a special format - return dict for google-specific handling
            content = [
                {"type": "text", "text": base_text},
                {
                    "type": "inline_data",
                    "mime_type": "application/pdf",
                    "data": encoded_pdf
                }
            ]
            messages.append({"role": "user", "content": content})

    elif page_data.get("image_bytes"):
        # Providers requiring image conversion (OpenAI, Mistral, xAI, etc.)
        encoded_image = _encode_bytes_to_base64(page_data["image_bytes"])
        encoded_image_url = f"data:image/png;base64,{encoded_image}"

        content = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": encoded_image_url, "detail": "high"}}
        ]
        messages.append({"role": "user", "content": content})

    else:
        # Fallback to text-only if no visual data available
        messages.append({"role": "user", "content": base_text})

    return messages


def _prepare_page_data(
    pdf_path: str,
    page_index: int,
    page_label: str,
    pdf_mode: str,
    provider: str,
    pdf_dpi: int = 150,
) -> dict:
    """
    Prepare page data for classification based on mode and provider.

    Args:
        pdf_path: Path to the PDF file
        page_index: Page number (0-indexed)
        page_label: Label for the page (e.g., "document_p1")
        pdf_mode: "image", "text", or "both"
        provider: Provider name for determining format
        pdf_dpi: DPI for image extraction

    Returns:
        Dict with page data ready for classification
    """
    page_data = {
        "pdf_path": pdf_path,
        "page_index": page_index,
        "page_label": page_label,
        "text": None,
        "image_bytes": None,
        "pdf_bytes": None,
        "error": None,
    }

    # Extract text if needed
    if pdf_mode in ("text", "both"):
        text, is_valid, error = _extract_page_text(pdf_path, page_index)
        if is_valid:
            page_data["text"] = text
        elif pdf_mode == "text":
            # Text mode requires text
            page_data["error"] = f"Failed to extract text: {error}"
            return page_data

    # Extract visual content if needed
    if pdf_mode in ("image", "both"):
        if provider in _NATIVE_PDF_PROVIDERS:
            # Extract as PDF bytes for native PDF providers
            pdf_bytes, is_valid = _extract_page_as_pdf_bytes(pdf_path, page_index)
            if is_valid:
                page_data["pdf_bytes"] = pdf_bytes
            else:
                # Fallback to image
                image_bytes, is_valid = _extract_page_as_image_bytes(pdf_path, page_index, dpi=pdf_dpi)
                if is_valid:
                    page_data["image_bytes"] = image_bytes
                else:
                    page_data["error"] = "Failed to extract page as PDF or image"
        else:
            # Extract as image for other providers
            image_bytes, is_valid = _extract_page_as_image_bytes(pdf_path, page_index, dpi=pdf_dpi)
            if is_valid:
                page_data["image_bytes"] = image_bytes
            else:
                page_data["error"] = "Failed to extract page as image"

    return page_data


# =============================================================================
# Image-Specific Functions
# =============================================================================

def build_image_classification_prompt(
    image_data: dict,
    categories_str: str,
    input_description: str = "",
    provider: str = "openai",
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    stepback_insights: dict = None,
    model_name: str = None,
    example_json: str = None,
) -> list:
    """
    Build the classification prompt for an image.

    This is the image-specific prompt builder, parallel to build_pdf_classification_prompt().

    Args:
        image_data: Dict containing:
            - image_path: Path to source image
            - image_label: Label for the image (filename without extension)
            - encoded_image: Base64 encoded image
            - extension: Image file extension (without dot)
        categories_str: Formatted string of categories (numbered list)
        input_description: Description of what the images contain
        provider: Provider name for format-specific handling
        chain_of_thought: Whether to use step-by-step reasoning
        context_prompt: Whether to add expert context prefix
        step_back_prompt: Whether step-back prompting is enabled
        stepback_insights: Dict of step-back insights per model
        model_name: Current model name (for step-back lookup)
        example_json: Example JSON output format

    Returns:
        List of message content parts for the LLM (format varies by provider)
    """
    # Build the base text prompt
    if chain_of_thought:
        base_text = f"""You are an image-tagging assistant.
Task: Examine the attached image and decide, **for each category below**, whether it is PRESENT (1) or NOT PRESENT (0).

{f'The image is expected to contain: {input_description}' if input_description else ''}

Categories:
{categories_str}

Let's analyze step by step:
1. First, identify the key visual elements in the image
2. Then, match each element to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Output format: Respond with **only** a JSON object whose keys are the quoted category numbers ('1', '2', ...) and whose values are 1 or 0. No additional keys, comments, or text.

{f'Example JSON format: {example_json}' if example_json else ''}"""
    else:
        base_text = f"""You are an image-tagging assistant.
Task: Examine the attached image and decide, **for each category below**, whether it is PRESENT (1) or NOT PRESENT (0).

{f'The image is expected to contain: {input_description}' if input_description else ''}

Categories:
{categories_str}

Output format: Respond with **only** a JSON object whose keys are the quoted category numbers ('1', '2', ...) and whose values are 1 or 0. No additional keys, comments, or text.

{f'Example JSON format: {example_json}' if example_json else ''}"""

    # Add context prompt prefix if enabled
    if context_prompt:
        context = """You are an expert visual analyst specializing in image categorization.
Apply multi-label classification based on explicit and implicit visual cues.
When uncertain, prioritize precision over recall.

"""
        base_text = context + base_text

    # Build messages based on provider
    messages = []

    # Add step-back insight if available
    if step_back_prompt and stepback_insights and model_name in stepback_insights:
        sb_question, sb_insight = stepback_insights[model_name]
        messages.append({"role": "user", "content": sb_question})
        messages.append({"role": "assistant", "content": sb_insight})

    # Get encoded image and extension
    encoded = image_data.get("encoded_image", "")
    ext = image_data.get("extension", "png")

    # Format depends on provider
    if provider == "anthropic":
        # Anthropic uses explicit media_type
        content = [
            {"type": "text", "text": base_text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{ext}",
                    "data": encoded
                }
            }
        ]
        messages.append({"role": "user", "content": content})

    elif provider == "google":
        # Google uses inline_data format
        content = [
            {"type": "text", "text": base_text},
            {
                "type": "inline_data",
                "mime_type": f"image/{ext}",
                "data": encoded
            }
        ]
        messages.append({"role": "user", "content": content})

    else:
        # OpenAI, Mistral, xAI, etc. use image_url format
        encoded_url = f"data:image/{ext};base64,{encoded}"
        content = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": encoded_url, "detail": "high"}}
        ]
        messages.append({"role": "user", "content": content})

    return messages


def _prepare_image_data(
    image_path: str,
    image_label: str,
) -> dict:
    """
    Prepare image data for classification.

    Args:
        image_path: Path to the image file
        image_label: Label for the image (typically filename without extension)

    Returns:
        Dict with image_path, image_label, encoded_image, extension, error
    """
    image_data = {
        "image_path": image_path,
        "image_label": image_label,
        "encoded_image": None,
        "extension": None,
        "error": None,
    }

    try:
        # _encode_image returns (encoded_data, extension, is_valid)
        encoded, ext_from_encode, is_valid = _encode_image(image_path)
        if not is_valid:
            image_data["error"] = "Failed to encode image"
            return image_data

        # Normalize extension
        ext = ext_from_encode.lower() if ext_from_encode else os.path.splitext(image_path)[1].lower().replace('.', '')
        if ext in ('jpg', 'jpe', 'jfif', 'pjpeg', 'pjp'):
            ext = 'jpeg'

        image_data["encoded_image"] = encoded
        image_data["extension"] = ext
    except Exception as e:
        image_data["error"] = str(e)

    return image_data


def _save_partial_results(
    all_results: list,
    model_configs: list,
    categories: list,
    filename: str,
    save_directory: str,
) -> None:
    """
    Save partial results to file for safety/incremental saves.

    This is a simplified version of build_output_dataframes that only
    saves the combined DataFrame without building per-model DataFrames.
    """
    model_names = [cfg["sanitized_name"] for cfg in model_configs]
    num_categories = len(categories)

    # Check if any results have PDF or image metadata
    has_pdf_metadata = any(result.get("pdf_path") is not None for result in all_results)
    has_image_metadata = any(result.get("image_path") is not None for result in all_results)

    # Build partial data
    rows = []
    for result in all_results:
        # Determine processing status
        if result.get("skipped"):
            status = "skipped"
        elif result["aggregated"]["error"]:
            status = "error"
        elif result["aggregated"]["failed_models"]:
            status = "partial"
        else:
            status = "success"

        row = {
            "survey_input": result["response"],
            "processing_status": status,
            "failed_models": ",".join(result["aggregated"]["failed_models"]) if result["aggregated"]["failed_models"] else "",
        }

        # Add PDF metadata columns if present
        if has_pdf_metadata:
            row["pdf_path"] = result.get("pdf_path", "")
            row["page_index"] = result.get("page_index", "")

        # Add image metadata columns if present
        if has_image_metadata:
            row["image_path"] = result.get("image_path", "")

        # Per-model results
        for model_name in model_names:
            parsed = result["aggregated"]["per_model"].get(model_name, {})
            for i in range(1, num_categories + 1):
                key = str(i)
                value = parsed.get(key, None)
                if value is not None:
                    try:
                        row[f"category_{i}_{model_name}"] = int(value)
                    except (ValueError, TypeError):
                        row[f"category_{i}_{model_name}"] = None
                else:
                    row[f"category_{i}_{model_name}"] = None

        # Consensus results
        for i in range(1, num_categories + 1):
            key = str(i)
            consensus_val = result["aggregated"]["consensus"].get(key, None)
            agreement_val = result["aggregated"]["agreement"].get(key, None)

            if consensus_val is not None:
                try:
                    row[f"category_{i}_consensus"] = int(consensus_val)
                except (ValueError, TypeError):
                    row[f"category_{i}_consensus"] = None
            else:
                row[f"category_{i}_consensus"] = None

            row[f"category_{i}_agreement"] = agreement_val

        rows.append(row)

    # Create DataFrame and save
    partial_df = pd.DataFrame(rows)

    save_path = filename
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, filename)

    partial_df.to_csv(save_path, index=False)


def classify_ensemble(
    survey_input,
    categories,
    # Single model mode (like original multi_class)
    model: str = None,
    api_key: str = None,
    provider: str = "auto",
    # Multi-model mode
    models: list = None,
    # Common parameters
    survey_question: str = "",
    example1: str = None,
    example2: str = None,
    example3: str = None,
    example4: str = None,
    example5: str = None,
    example6: str = None,
    creativity: float = None,
    chain_of_thought: bool = True,
    chain_of_verification: bool = False,
    step_back_prompt: bool = False,
    context_prompt: bool = False,
    thinking_budget: int = 0,
    use_json_schema: bool = True,
    max_workers: int = None,
    consensus_threshold: Union[str, float] = "majority",
    fail_strategy: str = "partial",
    safety: bool = False,
    max_retries: int = 5,
    batch_retries: int = 2,
    retry_delay: float = 1.0,
    filename: str = None,
    save_directory: str = None,
    progress_callback: Callable = None,
    # Auto-category detection parameters
    max_categories: int = 12,
    categories_per_chunk: int = 10,
    divisions: int = 10,
    research_question: str = None,
    # PDF-specific parameters (only used when survey_input contains PDFs)
    pdf_mode: str = "image",
    pdf_dpi: int = 150,
    input_description: str = "",
    # Ollama parameters
    auto_download: bool = False,
):
    """
    Multi-class classification with support for text AND PDF inputs, single or multiple LLM models.

    This unified function auto-detects whether the input is text or PDF and processes accordingly.

    Input type detection:
    - If survey_input is a directory path -> PDF mode
    - If survey_input contains .pdf file paths -> PDF mode
    - Otherwise -> Text mode

    This function can work in multiple modes:
    1. Single model mode: Like the original multi_class function
    2. Ensemble mode: Call multiple models in parallel with majority voting
    3. PDF mode: Classify PDF pages instead of text responses

    Args:
        survey_input: Text responses OR PDF paths (auto-detected)
            - Text mode: List or Series of text strings to classify
            - PDF mode: Directory path, single PDF path, or list of PDF paths

        categories: List of category names, or "auto" to auto-detect categories

        # Single model mode (use these for simple single-model classification):
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
        api_key: API key for the provider
        provider: Provider name or "auto" to detect from model name

        # Multi-model mode (use this for ensemble classification):
        models: List of tuples (model_name, provider, api_key), or a single tuple
            Example: [("gpt-4o", "openai", "sk-..."), ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-...")]

        # Classification parameters:
        survey_question: Context about what question was asked (required for categories="auto")
        example1-6: Optional few-shot examples for classification
        creativity: Temperature setting (None for provider default)
        chain_of_thought: If True, uses step-by-step reasoning in prompt
        chain_of_verification: If True, uses 4-step verification to improve accuracy
            (Note: ~4x API calls per response - expensive for ensemble mode)
        step_back_prompt: If True, first asks about underlying factors before classifying
        context_prompt: If True, adds expert context prefix to prompts
        thinking_budget: Token budget for Google's extended thinking (0 to disable)
        use_json_schema: Whether to use strict JSON schema (vs just json_object mode)

        # Ensemble parameters:
        max_workers: Maximum parallel workers (default: min(len(models), 8))
        consensus_threshold: Threshold for majority vote. Can be:
            - "majority": 50% agreement (default)
            - "two-thirds": 67% agreement
            - "unanimous": 100% agreement
            - float: Custom threshold between 0 and 1 (e.g., 0.75 for 75%)
        fail_strategy: How to handle model failures:
            - "partial": Continue with successful models
            - "strict": Fail row if any model fails
        safety: If True, saves results incrementally during processing to prevent
            data loss. Requires filename to be set.
        max_retries: Maximum retry attempts for each API call (handles rate limits,
            server errors, timeouts). Default 5.
        batch_retries: Maximum retry passes for failed (row, model) pairs after
            the batch completes. Default 2 means up to 3 total attempts. Set to 0
            to disable batch-level retries.
        retry_delay: Seconds to wait between batch retry passes.

        # Output parameters:
        filename: Optional CSV filename to save combined results (required if safety=True)
        save_directory: Optional directory for saved files
        progress_callback: Optional callback(response_idx, model_name, success, total, completed)

        # Auto-category detection parameters (used when categories="auto"):
        max_categories: Maximum number of categories to discover (default 12)
        categories_per_chunk: Categories to extract per data chunk (default 10)
        divisions: Number of chunks to divide data into (default 10)
        research_question: Optional research context for category discovery

        # PDF-specific parameters (only used when survey_input contains PDFs):
        pdf_mode: How to process PDF pages. Options:
            - "image": Render pages as images (best for visual elements)
            - "text": Extract text only (faster/cheaper for text-heavy docs)
            - "both": Send both text and image (most comprehensive)
        pdf_dpi: Resolution for PDF to image conversion (default 150)
        input_description: Description of what the PDF documents contain

        # Ollama parameters:
        auto_download: If True, automatically download missing Ollama models

    Returns:
        - Single model: Returns DataFrame directly (backward compatible with multi_class)
        - Multiple models: Returns dict containing:
            - "combined": DataFrame with all per-model and consensus columns
            - "consensus": DataFrame with only consensus results
            - "<model_name>": Individual DataFrame for each model

        DataFrame columns:
        - survey_input: Text string OR page label (e.g., "document_p1")
        - category_N_<model>: Per-model results (0/1)
        - category_N_consensus: Majority vote result
        - category_N_agreement: Model agreement score
        - processing_status: "success", "partial", "error", "skipped"
        - failed_models: List of failed models

        Additional columns (PDF mode only):
        - pdf_path: Source PDF file path
        - page_index: Page number (0-indexed)

    Examples:
        # TEXT MODE - Single model (returns DataFrame directly):
        df = multi_class_ensemble(
            survey_input=["I moved for a new job"],
            categories=["Employment", "Family", "Housing"],
            model="gpt-4o",
            api_key="sk-...",
            survey_question="Why did you move?",
        )

        # TEXT MODE - Ensemble with multiple models (returns dict):
        results = multi_class_ensemble(
            survey_input=["I moved for a new job"],
            categories=["Employment", "Family", "Housing"],
            models=[
                ("gpt-4o", "openai", "sk-..."),
                ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
            ],
            survey_question="Why did you move?",
        )
        combined_df = results["combined"]

        # PDF MODE - Single model (auto-detected from .pdf paths):
        df = multi_class_ensemble(
            survey_input="reports/",  # Directory of PDFs
            categories=["Has Chart", "Has Table", "Financial Summary"],
            model="gpt-4o",
            api_key="sk-...",
            pdf_mode="image",
            input_description="Financial reports with charts and tables",
        )

        # PDF MODE - Ensemble with native PDF support:
        results = multi_class_ensemble(
            survey_input=["doc1.pdf", "doc2.pdf"],
            categories=["Diagnosis", "Treatment Plan"],
            models=[
                ("gpt-4o", "openai", "sk-..."),
                ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),  # Native PDF
            ],
            pdf_mode="both",
            consensus_threshold=0.5,
        )
    """
    # Normalize model input to list of tuples
    models = normalize_model_input(model, api_key, provider, models)

    # Validate safety parameter
    if safety and filename is None:
        raise TypeError(
            "filename is required when using safety=True. "
            "Please provide a filename to save incremental results to."
        )

    # Handle categories="auto" - auto-detect categories from the data
    if categories == "auto":
        if survey_question == "":
            raise TypeError(
                "survey_question is required when using categories='auto'. "
                "Please provide the survey question you are analyzing."
            )

        # Use first model for category discovery
        first_model, first_provider, first_api_key = models[0]
        detected_provider = detect_provider(first_model, first_provider)

        print(f"Auto-detecting categories using {first_model}...")
        auto_result = explore_common_categories(
            survey_question=survey_question,
            survey_input=survey_input,
            research_question=research_question,
            api_key=first_api_key,
            model_source=detected_provider,
            user_model=first_model,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions
        )
        categories = auto_result["top_categories"]
        print(f"Discovered {len(categories)} categories: {categories}")

    if not isinstance(categories, list) or len(categories) == 0:
        raise ValueError("categories must be a non-empty list of category names, or 'auto'")

    # Prepare model configurations
    print(f"Validating {len(models)} model configuration(s)...")
    model_configs = prepare_model_configs(models, auto_download=auto_download)

    # Print model info
    print(f"\nModels to use:")
    for cfg in model_configs:
        print(f"  - {cfg['model']} ({cfg['provider']}) -> column suffix: {cfg['sanitized_name']}")

    # =============================================================================
    # DETECT INPUT TYPE: Text vs PDF vs Image
    # =============================================================================
    input_type = _detect_input_type(survey_input)
    print(f"\nInput type detected: {input_type.upper()}")

    # Initialize processing variables
    items_to_process = []
    is_pdf_mode = (input_type == 'pdf')
    is_image_mode = (input_type == 'image')

    # Build example JSON for visual modes (PDF/image)
    category_dict = {str(i+1): "0" for i in range(len(categories))}
    example_json = json.dumps(category_dict, indent=2)

    if is_image_mode:
        # =================================================================
        # IMAGE MODE: Load images
        # =================================================================
        print(f"Loading images...")

        # Load image files
        image_files = _load_image_files(survey_input)

        if not image_files:
            raise ValueError("No images found in the provided input.")

        print(f"Total images to process: {len(image_files)}")

        # items_to_process is list of (image_path, image_label) tuples
        items_to_process = [
            (img_path, os.path.splitext(os.path.basename(img_path))[0])
            for img_path in image_files
        ]

    elif is_pdf_mode:
        # =================================================================
        # PDF MODE: Load PDFs and extract all pages
        # =================================================================
        # Validate pdf_mode parameter
        pdf_mode = pdf_mode.lower()
        if pdf_mode not in {"image", "text", "both"}:
            raise ValueError(f"pdf_mode must be 'image', 'text', or 'both', got: {pdf_mode}")

        print(f"PDF processing mode: {pdf_mode}")

        # Load PDF files
        pdf_files = _load_pdf_files(survey_input)

        # Extract all pages from all PDFs
        all_pages = []
        for pdf_path in pdf_files:
            pages = _get_pdf_pages(pdf_path)
            all_pages.extend(pages)

        if not all_pages:
            raise ValueError("No pages found in the provided PDF files.")

        print(f"Total pages to process: {len(all_pages)}")

        # items_to_process is list of (pdf_path, page_index, page_label) tuples
        items_to_process = all_pages

    else:
        # =================================================================
        # TEXT MODE: survey_input is the items to process
        # =================================================================
        items_to_process = survey_input

    # Set max workers
    effective_workers = max_workers or min(len(models), 8)
    print(f"\nParallel workers: {effective_workers}")

    # Warn about CoVe cost with ensemble
    if chain_of_verification:
        print("\n[Chain of Verification enabled]")
        print("  - ~4x API calls per response per model")
        if len(models) > 1:
            print("  - WARNING: CoVe with ensemble is expensive. Consider single-model mode.")

    # Build shared prompt components
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))

    examples = [example1, example2, example3, example4, example5, example6]
    examples_text = "\n".join(
        f"Example {i}: {ex}" for i, ex in enumerate(examples, 1) if ex is not None
    )

    survey_question_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # Print categories
    print(f"\nCategories to classify ({len(categories)} total):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print()

    # Get step-back insights per model (if enabled)
    stepback_insights = {}
    if step_back_prompt:
        stepback_insights = gather_stepback_insights(model_configs, survey_question, creativity)

    # Build JSON schemas per provider
    json_schemas = prepare_json_schemas(model_configs, categories, use_json_schema)

    # Build original task prompt for CoVe (if enabled)
    cove_original_task = ""
    if chain_of_verification:
        cove_original_task = f"""{survey_question_context}
Categorize survey responses into the following categories:
{categories_str}
Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""

    # Classification function for single model + single item (text, PDF page, or image)
    def classify_single(cfg: dict, item) -> tuple:
        """
        Classify one item (text response, PDF page, or image) with one model.

        Args:
            cfg: Model configuration dict
            item: Either:
                - Text string (text mode)
                - Tuple (pdf_path, page_index, page_label) (PDF mode)
                - Tuple (image_path, image_label) (image mode)

        Returns:
            tuple: (model_name, json_result, error)
        """
        # Determine item type and identifier
        if is_image_mode and isinstance(item, tuple) and len(item) == 2:
            # Image mode: item is (image_path, image_label)
            image_path, image_label = item
            item_identifier = image_label
        elif is_pdf_mode and isinstance(item, tuple) and len(item) == 3:
            # PDF mode: item is (pdf_path, page_index, page_label)
            pdf_path, page_index, page_label = item
            item_identifier = page_label
        else:
            # Text mode: item is text string
            item_identifier = str(item) if item else ""

        # Test hook for debugging batch retries (only active when _TEST_FORCE_FAILURE = True)
        if _test_should_force_failure(item_identifier, cfg["sanitized_name"]):
            return (cfg["sanitized_name"], '{"1":"e"}', "TEST: Forced first-attempt failure")

        try:
            client = UnifiedLLMClient(
                provider=cfg["provider"],
                api_key=cfg["api_key"],
                model=cfg["model"]
            )

            # =================================================================
            # PDF MODE: Build PDF-specific prompt
            # =================================================================
            if is_pdf_mode and isinstance(item, tuple):
                pdf_path, page_index, page_label = item

                # Prepare page data based on mode and provider
                page_data = _prepare_page_data(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    page_label=page_label,
                    pdf_mode=pdf_mode,
                    provider=cfg["provider"],
                    pdf_dpi=pdf_dpi,
                )

                # Check for extraction errors
                if page_data.get("error"):
                    return (cfg["sanitized_name"], '{"1":"e"}', page_data["error"])

                # Build PDF classification prompt
                messages = build_pdf_classification_prompt(
                    page_data=page_data,
                    categories_str=categories_str,
                    input_description=input_description,
                    provider=cfg["provider"],
                    pdf_mode=pdf_mode,
                    chain_of_thought=chain_of_thought,
                    context_prompt=context_prompt,
                    step_back_prompt=step_back_prompt,
                    stepback_insights=stepback_insights,
                    model_name=cfg["model"],
                    example_json=example_json,
                )

                # Handle Google API separately (different format)
                if cfg["provider"] == "google":
                    # Google needs special handling for multimodal content
                    reply, error = _call_google_multimodal(
                        client=client,
                        messages=messages,
                        json_schema=json_schemas[cfg["model"]],
                        creativity=creativity,
                        thinking_budget=thinking_budget,
                        max_retries=max_retries,
                    )
                else:
                    reply, error = client.complete(
                        messages=messages,
                        json_schema=json_schemas[cfg["model"]],
                        creativity=creativity,
                        thinking_budget=thinking_budget if cfg["provider"] == "google" else None,
                        max_retries=max_retries,
                    )

                if error:
                    json_result = '{"1":"e"}'
                else:
                    json_result = extract_json(reply)

                # Note: CoVe for PDF mode is not yet implemented
                # (would require re-attaching PDF/image to verification prompts)

            # =================================================================
            # IMAGE MODE: Build image-specific prompt
            # =================================================================
            elif is_image_mode and isinstance(item, tuple):
                image_path, image_label = item

                # Prepare image data
                image_data = _prepare_image_data(image_path, image_label)

                # Check for encoding errors
                if image_data.get("error"):
                    return (cfg["sanitized_name"], '{"1":"e"}', image_data["error"])

                # Build image classification prompt
                messages = build_image_classification_prompt(
                    image_data=image_data,
                    categories_str=categories_str,
                    input_description=input_description,
                    provider=cfg["provider"],
                    chain_of_thought=chain_of_thought,
                    context_prompt=context_prompt,
                    step_back_prompt=step_back_prompt,
                    stepback_insights=stepback_insights,
                    model_name=cfg["model"],
                    example_json=example_json,
                )

                # Handle Google API separately (different format)
                if cfg["provider"] == "google":
                    # Google needs special handling for multimodal content
                    reply, error = _call_google_multimodal(
                        client=client,
                        messages=messages,
                        json_schema=json_schemas[cfg["model"]],
                        creativity=creativity,
                        thinking_budget=thinking_budget,
                        max_retries=max_retries,
                    )
                else:
                    reply, error = client.complete(
                        messages=messages,
                        json_schema=json_schemas[cfg["model"]],
                        creativity=creativity,
                        thinking_budget=thinking_budget if cfg["provider"] == "google" else None,
                        max_retries=max_retries,
                    )

                if error:
                    json_result = '{"1":"e"}'
                else:
                    json_result = extract_json(reply)

                # Note: CoVe for image mode is not yet implemented

            # =================================================================
            # TEXT MODE: Original text classification logic
            # =================================================================
            else:
                response_text = item

                if cfg["use_two_step"]:  # Ollama
                    json_result, error = ollama_two_step_classify(
                        client=client,
                        response_text=response_text,
                        categories=categories,
                        categories_str=categories_str,
                        survey_question=survey_question,
                        creativity=creativity,
                        max_retries=max_retries,
                    )
                    # CoVe not supported for Ollama two-step (already has verification)
                else:
                    messages = build_text_classification_prompt(
                        response_text=response_text,
                        categories_str=categories_str,
                        survey_question_context=survey_question_context,
                        examples_text=examples_text,
                        chain_of_thought=chain_of_thought,
                        context_prompt=context_prompt,
                        step_back_prompt=step_back_prompt,
                        stepback_insights=stepback_insights,
                        model_name=cfg["model"],
                    )
                    reply, error = client.complete(
                        messages=messages,
                        json_schema=json_schemas[cfg["model"]],
                        creativity=creativity,
                        thinking_budget=thinking_budget if cfg["provider"] == "google" else None,
                        max_retries=max_retries,
                    )
                    if error:
                        json_result = '{"1":"e"}'
                    else:
                        json_result = extract_json(reply)

                        # Run Chain of Verification if enabled
                        if chain_of_verification and not error:
                            step2, step3, step4 = build_cove_prompts(
                                cove_original_task, response_text
                            )
                            json_result = run_chain_of_verification(
                                client=client,
                                initial_reply=json_result,
                                step2_prompt=step2,
                                step3_prompt=step3,
                                step4_prompt=step4,
                                json_schema=json_schemas[cfg["model"]],
                                creativity=creativity,
                                max_retries=max_retries,
                            )

            return (cfg["sanitized_name"], json_result, error)

        except Exception as e:
            return (cfg["sanitized_name"], '{"1":"e"}', str(e))

    # Helper function for Google multimodal API calls
    def _call_google_multimodal(client, messages, json_schema, creativity, thinking_budget, max_retries):
        """
        Handle Google's multimodal API format for PDF/image content.
        """
        import requests

        # Extract the content from messages
        user_msg = messages[-1]  # Last message should be the user message
        content = user_msg.get("content", [])

        # Build Google-format parts
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"text": part["text"]})
            elif part.get("type") == "inline_data":
                parts.append({
                    "inline_data": {
                        "mime_type": part["mime_type"],
                        "data": part["data"]
                    }
                })
            elif part.get("type") == "image_url":
                # Convert image URL to inline_data format
                url = part["image_url"]["url"]
                if url.startswith("data:image/png;base64,"):
                    data = url.replace("data:image/png;base64,", "")
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": data
                        }
                    })

        # Get model name from client
        model_name = client.model

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {
            "x-goog-api-key": client.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({"temperature": creativity} if creativity is not None else {}),
                **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget else {})
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()

                if "candidates" in result and result["candidates"]:
                    reply = result["candidates"][0]["content"]["parts"][0]["text"]
                    return reply, None
                else:
                    return None, "No response generated"

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                retryable_errors = [429, 500, 502, 503, 504]

                if status_code in retryable_errors and attempt < max_retries - 1:
                    import time
                    wait_time = 10 * (2 ** attempt) if status_code == 429 else 2 * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return None, f"HTTP error {status_code}: {str(e)}"

            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 * (2 ** attempt))
                else:
                    return None, str(e)

        return None, "Max retries exceeded"

    # Process all items (text responses, PDF pages, or images)
    all_results = []
    completed_calls = [0]  # Mutable for closure
    total_calls = len(items_to_process) * len(model_configs)

    # Set progress description based on mode
    if is_image_mode:
        progress_desc = "Classifying images"
    elif is_pdf_mode:
        progress_desc = "Classifying PDF pages"
    else:
        progress_desc = "Classifying responses"

    # Disable tqdm when progress_callback is provided (e.g., for Streamlit/GUI apps)
    use_tqdm = progress_callback is None
    for idx, item in enumerate(tqdm(items_to_process, desc=progress_desc, disable=not use_tqdm)):
        skipped_nan = False

        # Determine the display identifier and metadata for this item
        if is_image_mode and isinstance(item, tuple) and len(item) == 2:
            image_path, image_label = item
            display_id = image_label
            pdf_metadata = None
            image_metadata = {"image_path": image_path}
        elif is_pdf_mode and isinstance(item, tuple) and len(item) == 3:
            pdf_path, page_index, page_label = item
            display_id = page_label
            pdf_metadata = {"pdf_path": pdf_path, "page_index": page_index}
            image_metadata = None
        else:
            display_id = item
            pdf_metadata = None
            image_metadata = None

        # Check for NaN (text mode only)
        if not is_pdf_mode and not is_image_mode and pd.isna(item):
            # Handle NaN - mark as skipped, not error
            skipped_nan = True
            model_results = {
                cfg["sanitized_name"]: ('{"1":"e"}', "Skipped NaN input")
                for cfg in model_configs
            }
        else:
            # Parallel classification across models
            model_results = {}
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(classify_single, cfg, item): cfg["sanitized_name"]
                    for cfg in model_configs
                }

                for future in as_completed(futures):
                    model_name, json_result, error = future.result()
                    model_results[model_name] = (json_result, error)

                    # Update progress (for multi-model detailed callbacks only)
                    completed_calls[0] += 1

        # Aggregate results with majority voting
        aggregated = aggregate_results(
            model_results,
            categories,
            consensus_threshold,
            fail_strategy
        )

        # Build result entry
        result_entry = {
            "response": display_id,  # Page label for PDF, text for text mode
            "model_results": model_results,
            "aggregated": aggregated,
            "skipped": skipped_nan,
        }

        # Add PDF metadata if in PDF mode
        if pdf_metadata:
            result_entry["pdf_path"] = pdf_metadata["pdf_path"]
            result_entry["page_index"] = pdf_metadata["page_index"]

        # Add image metadata if in image mode
        if image_metadata:
            result_entry["image_path"] = image_metadata["image_path"]

        # Store the original item for retry logic
        result_entry["_original_item"] = item

        all_results.append(result_entry)

        # Call progress callback with simple item-level signature
        # This is for UI integrations (like Streamlit) that want per-item updates
        if progress_callback:
            try:
                # Try simple signature: (current_idx, total, item_label)
                progress_callback(idx, len(items_to_process), display_id)
            except TypeError:
                # Fallback: callback might expect keyword args (old signature)
                pass

        # Safety incremental save
        if safety:
            _save_partial_results(
                all_results,
                model_configs,
                categories,
                filename,
                save_directory,
            )

    # Retry logic for failed (row, model) pairs
    if batch_retries > 0:
        for retry_num in range(1, batch_retries + 1):
            # Find all failed (row_idx, model_config) pairs
            failed_pairs = []
            for row_idx, result in enumerate(all_results):
                # Skip rows that were NaN inputs
                if result.get("skipped"):
                    continue
                # Check each model for this row
                for cfg in model_configs:
                    model_name = cfg["sanitized_name"]
                    json_str, error = result["model_results"].get(model_name, (None, "Missing"))
                    if error is not None:
                        failed_pairs.append((row_idx, cfg))
                    else:
                        # Also check if JSON parsing would fail
                        try:
                            parsed = json.loads(json_str)
                        except (json.JSONDecodeError, TypeError):
                            failed_pairs.append((row_idx, cfg))

            if not failed_pairs:
                break  # All successful, no retries needed

            print(f"\n[Batch retry {retry_num}/{batch_retries}] Retrying {len(failed_pairs)} failed (row, model) pairs...")

            # Wait before retrying
            if retry_delay > 0:
                time.sleep(retry_delay)

            # Retry failed pairs
            successes_this_round = 0
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(classify_single, cfg, all_results[row_idx]["_original_item"]): (row_idx, cfg)
                    for row_idx, cfg in failed_pairs
                }

                for future in as_completed(futures):
                    row_idx, cfg = futures[future]
                    model_name, json_result, error = future.result()

                    # Update the result in place
                    all_results[row_idx]["model_results"][model_name] = (json_result, error)

                    if error is None:
                        # Verify JSON is valid
                        try:
                            json.loads(json_result)
                            successes_this_round += 1
                        except (json.JSONDecodeError, TypeError):
                            pass

            # Recalculate aggregation for all affected rows
            affected_rows = set(row_idx for row_idx, _ in failed_pairs)
            for row_idx in affected_rows:
                all_results[row_idx]["aggregated"] = aggregate_results(
                    all_results[row_idx]["model_results"],
                    categories,
                    consensus_threshold,
                    fail_strategy
                )

            print(f"  -> {successes_this_round}/{len(failed_pairs)} pairs succeeded on retry")

            # Safety save after retry
            if safety:
                _save_partial_results(
                    all_results,
                    model_configs,
                    categories,
                    filename,
                    save_directory,
                )

            # Early exit if ALL retries failed (likely server down or out of credits)
            if successes_this_round == 0:
                print(f"  -> All retries failed. Stopping retry loop (possible server issue).")
                break

    # Build output DataFrames
    print("\nBuilding output DataFrames...")
    return build_output_dataframes(
        all_results,
        model_configs,
        categories,
        filename,
        save_directory,
    )


def build_output_dataframes(
    all_results: list,
    model_configs: list,
    categories: list,
    filename: str,
    save_directory: str,
) -> dict:
    """
    Build the output DataFrames from classification results.

    Returns:
        Dict with "combined", "consensus", and per-model DataFrames
    """
    model_names = [cfg["sanitized_name"] for cfg in model_configs]
    num_categories = len(categories)

    # Check if any results have PDF metadata
    has_pdf_metadata = any(result.get("pdf_path") is not None for result in all_results)

    # Check if any results have image metadata
    has_image_metadata = any(result.get("image_path") is not None for result in all_results)

    # Initialize data structures
    combined_data = {
        "survey_input": [],
        "processing_status": [],
        "failed_models": [],
    }

    # Add PDF metadata columns if present
    if has_pdf_metadata:
        combined_data["pdf_path"] = []
        combined_data["page_index"] = []

    # Add image metadata columns if present
    if has_image_metadata:
        combined_data["image_path"] = []

    # Add columns for each model and each category
    for model_name in model_names:
        for i in range(1, num_categories + 1):
            combined_data[f"category_{i}_{model_name}"] = []

    # Add consensus and agreement columns
    for i in range(1, num_categories + 1):
        combined_data[f"category_{i}_consensus"] = []
        combined_data[f"category_{i}_agreement"] = []

    # Populate data
    for result in all_results:
        combined_data["survey_input"].append(result["response"])
        aggregated = result["aggregated"]

        # Add PDF metadata if present
        if has_pdf_metadata:
            combined_data["pdf_path"].append(result.get("pdf_path", ""))
            combined_data["page_index"].append(result.get("page_index", ""))

        # Add image metadata if present
        if has_image_metadata:
            combined_data["image_path"].append(result.get("image_path", ""))

        # Determine processing status
        if result.get("skipped"):
            combined_data["processing_status"].append("skipped")
        elif aggregated["error"]:
            combined_data["processing_status"].append("error")
        elif aggregated["failed_models"]:
            combined_data["processing_status"].append("partial")
        else:
            combined_data["processing_status"].append("success")

        combined_data["failed_models"].append(
            ",".join(aggregated["failed_models"]) if aggregated["failed_models"] else ""
        )

        # Per-model results
        for model_name in model_names:
            parsed = aggregated["per_model"].get(model_name, {})
            for i in range(1, num_categories + 1):
                key = str(i)
                col_name = f"category_{i}_{model_name}"
                value = parsed.get(key, None)
                if value is not None:
                    try:
                        combined_data[col_name].append(int(value))
                    except (ValueError, TypeError):
                        combined_data[col_name].append(None)
                else:
                    combined_data[col_name].append(None)

        # Consensus results
        for i in range(1, num_categories + 1):
            key = str(i)
            consensus_val = aggregated["consensus"].get(key, None)
            agreement_val = aggregated["agreement"].get(key, None)

            if consensus_val is not None:
                try:
                    combined_data[f"category_{i}_consensus"].append(int(consensus_val))
                except (ValueError, TypeError):
                    combined_data[f"category_{i}_consensus"].append(None)
            else:
                combined_data[f"category_{i}_consensus"].append(None)

            combined_data[f"category_{i}_agreement"].append(agreement_val)

    # Create combined DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Convert category columns to Int64 (nullable integer)
    cat_cols = [c for c in combined_df.columns if c.startswith("category_") and not c.endswith("_agreement")]
    for col in cat_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').astype('Int64')

    # Create consensus-only DataFrame
    consensus_cols = ["survey_input", "processing_status", "failed_models"]
    # Add PDF columns if present
    if has_pdf_metadata:
        consensus_cols += ["pdf_path", "page_index"]
    # Add image columns if present
    if has_image_metadata:
        consensus_cols += ["image_path"]
    consensus_cols += [c for c in combined_df.columns if "_consensus" in c or "_agreement" in c]
    consensus_df = combined_df[consensus_cols].copy()

    # Create per-model DataFrames
    output = {
        "combined": combined_df,
        "consensus": consensus_df,
    }

    for model_name in model_names:
        model_cols = ["survey_input", "processing_status"]
        # Add PDF columns if present
        if has_pdf_metadata:
            model_cols += ["pdf_path", "page_index"]
        # Add image columns if present
        if has_image_metadata:
            model_cols += ["image_path"]
        model_cols += [c for c in combined_df.columns if f"_{model_name}" in c]
        output[model_name] = combined_df[model_cols].copy()

    # Save to file if requested
    if filename:
        save_path = filename
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, filename)
        combined_df.to_csv(save_path, index=False)
        print(f"\nCombined results saved to {save_path}")

    # If only one model, return simplified DataFrame (backward compatible with multi_class)
    if len(model_names) == 1:
        model_name = model_names[0]
        simplified_df = combined_df.copy()

        # Remove consensus/agreement/failed_models columns (redundant for single model)
        cols_to_drop = [c for c in simplified_df.columns if "_consensus" in c or "_agreement" in c]
        if "failed_models" in simplified_df.columns:
            cols_to_drop.append("failed_models")
        simplified_df = simplified_df.drop(columns=cols_to_drop)

        # Rename category columns to remove model suffix: category_1_model_name -> category_1
        rename_map = {}
        for col in simplified_df.columns:
            if col.startswith("category_") and f"_{model_name}" in col:
                # Extract category number and create simple name
                new_name = col.replace(f"_{model_name}", "")
                rename_map[col] = new_name
        simplified_df = simplified_df.rename(columns=rename_map)

        return simplified_df

    # For multiple models, return the combined DataFrame (contains all model results + consensus)
    return combined_df


# Backward compatibility alias
multi_class_ensemble = classify_ensemble


# =============================================================================
# Summarization Ensemble Function
# =============================================================================

def summarize_ensemble(
    survey_input,
    api_key: str = None,
    input_description: str = "",
    summary_instructions: str = "",
    max_length: int = None,
    focus: str = None,
    user_model: str = "gpt-4o",
    model_source: str = "auto",
    pdf_mode: str = "image",
    pdf_dpi: int = 150,
    creativity: float = None,
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    max_retries: int = 5,
    batch_retries: int = 2,
    retry_delay: float = 1.0,
    safety: bool = False,
    filename: str = None,
    save_directory: str = None,
    progress_callback: Optional[Callable] = None,
    # Multi-model parameters
    models: list = None,
) -> pd.DataFrame:
    """
    Summarize text or PDF inputs using LLMs with optional multi-model ensemble.

    Supports single-model and multi-model modes. In multi-model mode,
    summaries from all models are collected and synthesized into a consensus
    summary using an LLM. Input type is auto-detected from the data.

    Args:
        survey_input: Data to summarize. Can be:
            - Text: list of strings, pandas Series, or single string
            - PDF: directory path, single PDF path, or list of PDF paths
        api_key: API key for single-model mode
        input_description: Description of what the content contains (provides context)
        summary_instructions: Specific summarization instructions (e.g., "bullet points")
        max_length: Maximum summary length in words
        focus: What to focus on (e.g., "main arguments", "emotional content")
        user_model: Model to use (default "gpt-4o")
        model_source: Provider - "auto", "openai", "anthropic", "google", etc.
        pdf_mode: PDF processing mode (only used for PDF input):
            - "image" (default): Render pages as images
            - "text": Extract text only
            - "both": Send both image and extracted text
        pdf_dpi: DPI for PDF page rendering (default 150)
        creativity: Temperature setting (None uses provider default)
        chain_of_thought: Enable step-by-step reasoning (default True)
        context_prompt: Add expert context prefix
        step_back_prompt: Enable step-back prompting
        max_retries: Max retries per API call
        batch_retries: Number of batch retry passes for failed items
        retry_delay: Delay between retries in seconds
        safety: Save progress after each item
        filename: Output CSV filename
        save_directory: Directory to save results
        progress_callback: Optional callback for progress updates
        models: For multi-model mode, list of (model, provider, api_key) tuples

    Returns:
        DataFrame with columns:
        - survey_input: Original text or page label (for PDFs)
        - summary: Generated summary (or consensus summary for multi-model)
        - summary_<model>: Per-model summaries (multi-model only)
        - processing_status: "success", "error", "skipped"
        - failed_models: Comma-separated list (multi-model only)
        - pdf_path: Path to source PDF (PDF mode only)
        - page_index: Page number, 0-indexed (PDF mode only)

    Examples:
        >>> import catllm as cat
        >>>
        >>> # Single model text summarization
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key=api_key
        ... )
        >>>
        >>> # PDF summarization (auto-detected)
        >>> results = cat.summarize(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     mode="image",
        ...     api_key=api_key
        ... )
        >>>
        >>> # Multi-model with synthesis
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     models=[
        ...         ("gpt-4o", "openai", "sk-..."),
        ...         ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ...     ],
        ... )
    """
    # Detect input type: Text vs PDF
    input_type = _detect_input_type(survey_input)
    is_pdf_mode = (input_type == 'pdf')

    if is_pdf_mode:
        # Validate pdf_mode parameter
        pdf_mode = pdf_mode.lower()
        if pdf_mode not in {"image", "text", "both"}:
            raise ValueError(f"pdf_mode must be 'image', 'text', or 'both', got: {pdf_mode}")

        print(f"\nInput type detected: PDF")
        print(f"PDF processing mode: {pdf_mode}")

        # Load PDF files
        pdf_files = _load_pdf_files(survey_input)

        # Extract all pages from all PDFs
        all_pages = []
        for pdf_path in pdf_files:
            pages = _get_pdf_pages(pdf_path)
            all_pages.extend(pages)

        if not all_pages:
            raise ValueError("No pages found in the provided PDF files.")

        items_to_process = all_pages
        print(f"Total PDF pages to summarize: {len(items_to_process)}")
    else:
        # TEXT MODE: Normalize input to list
        print(f"\nInput type detected: TEXT")
        if isinstance(survey_input, str):
            survey_input = [survey_input]
        elif hasattr(survey_input, 'tolist'):
            survey_input = survey_input.tolist()
        else:
            survey_input = list(survey_input)

        items_to_process = survey_input
        print(f"Total texts to summarize: {len(items_to_process)}")

    # Normalize model input to list of tuples
    models = normalize_model_input(user_model, api_key, model_source, models)

    # Validate and prepare model configs
    print(f"Validating {len(models)} model configuration(s)...")
    model_configs = prepare_model_configs(models)

    if not model_configs:
        raise ValueError("No valid model configurations found.")

    model_names = [cfg["sanitized_name"] for cfg in model_configs]
    print(f"\nModels to use:")
    for cfg in model_configs:
        print(f"  - {cfg['model']} ({cfg['provider']}) -> column suffix: {cfg['sanitized_name']}")

    # Build JSON schemas per provider (for summary output)
    json_schemas = {}
    for cfg in model_configs:
        provider = cfg["provider"]
        include_additional = provider != "google"
        json_schemas[cfg["sanitized_name"]] = build_summary_json_schema(include_additional)

    # Example JSON for prompt
    example_json = '{"summary": "Your summary here"}'

    # Gather step-back insights if enabled
    stepback_insights = {}
    if step_back_prompt:
        print("\nGathering step-back insights...")
        stepback_insights = gather_stepback_insights(
            model_configs=model_configs,
            context=input_description or "text summarization",
            question=f"What are the key factors to consider when summarizing text{f' with a focus on {focus}' if focus else ''}?"
        )

    # Initialize results storage
    all_results = []  # List of dicts, one per input item
    failed_pairs = []  # List of (idx, model_name) pairs that failed

    # Define the summarization function for a single item
    def summarize_single_item(item, idx, cfg):
        """Summarize a single text item or PDF page with a single model."""
        model_name = cfg["sanitized_name"]

        # Determine if this is a PDF page or text
        if is_pdf_mode and isinstance(item, tuple) and len(item) == 3:
            # PDF mode: item is (pdf_path, page_index, page_label)
            pdf_path, page_index, page_label = item

            try:
                # Prepare page data based on mode and provider
                page_data = _prepare_page_data(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    page_label=page_label,
                    pdf_mode=pdf_mode,
                    provider=cfg["provider"],
                    pdf_dpi=pdf_dpi,
                )

                # Check for extraction errors
                if page_data.get("error"):
                    return (model_name, '{"summary": ""}', page_data["error"])

                # Build PDF summarization prompt
                messages = build_pdf_summarization_prompt(
                    page_data=page_data,
                    input_description=input_description,
                    summary_instructions=summary_instructions,
                    max_length=max_length,
                    focus=focus,
                    provider=cfg["provider"],
                    pdf_mode=pdf_mode,
                    chain_of_thought=chain_of_thought,
                    context_prompt=context_prompt,
                    step_back_prompt=step_back_prompt,
                    stepback_insights=stepback_insights,
                    model_name=model_name,
                )

                # Create client and make API call
                client = UnifiedLLMClient(
                    provider=cfg["provider"],
                    api_key=cfg["api_key"],
                    model=cfg["model"],
                )

                json_schema = json_schemas[model_name]

                # Handle Google multimodal differently
                if cfg["provider"] == "google" and pdf_mode != "text":
                    response = _call_google_multimodal(
                        client=client,
                        messages=messages,
                        json_schema=json_schema,
                        creativity=creativity,
                        thinking_budget=0,
                        max_retries=max_retries,
                    )
                else:
                    response = client.complete(
                        messages=messages,
                        json_schema=json_schema,
                        creativity=creativity,
                        max_retries=max_retries,
                    )

                # Extract JSON from response
                json_str = extract_json(response)

                return (model_name, json_str, None)

            except Exception as e:
                error_msg = str(e)
                return (model_name, '{"summary": ""}', error_msg)

        else:
            # TEXT MODE: Original text handling
            # Skip empty/null items
            if item is None or (isinstance(item, str) and not item.strip()) or pd.isna(item):
                return (model_name, '{"summary": ""}', "skipped")

            try:
                # Build the prompt
                messages = build_text_summarization_prompt(
                    response_text=str(item),
                    input_description=input_description,
                    summary_instructions=summary_instructions,
                    max_length=max_length,
                    focus=focus,
                    chain_of_thought=chain_of_thought,
                    context_prompt=context_prompt,
                    step_back_prompt=step_back_prompt,
                    stepback_insights=stepback_insights,
                    model_name=model_name,
                )

                # Create client and make API call
                client = UnifiedLLMClient(
                    provider=cfg["provider"],
                    api_key=cfg["api_key"],
                    model=cfg["model"],
                )

                json_schema = json_schemas[model_name]

                response = client.complete(
                    messages=messages,
                    json_schema=json_schema,
                    creativity=creativity,
                    max_retries=max_retries,
                )

                # Extract JSON from response
                json_str = extract_json(response)

                return (model_name, json_str, None)

            except Exception as e:
                error_msg = str(e)
                return (model_name, '{"summary": ""}', error_msg)

    # Process all items
    progress_desc = "Summarizing PDF pages" if is_pdf_mode else "Summarizing texts"
    print(f"\n{progress_desc}...")

    # Determine number of workers
    max_workers = min(len(model_configs), 4)

    # Progress tracking
    total_items = len(items_to_process)

    for idx, item in enumerate(tqdm(items_to_process, desc=progress_desc)):
        item_results = {}
        item_errors = {}

        # Process with each model (in parallel if multiple models)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(summarize_single_item, item, idx, cfg): cfg["sanitized_name"]
                for cfg in model_configs
            }

            for future in as_completed(futures):
                model_name, json_result, error = future.result()
                item_results[model_name] = json_result
                if error and error != "skipped":
                    item_errors[model_name] = error
                    failed_pairs.append((idx, model_name))

        # Store results for this item
        result_entry = {
            "idx": idx,
            "survey_input": item,
            "model_results": item_results,
            "errors": item_errors,
        }
        all_results.append(result_entry)

        # Progress callback
        if progress_callback:
            progress_callback(idx + 1, total_items)

    # Batch retries for failed pairs
    for retry_pass in range(batch_retries):
        if not failed_pairs:
            break

        print(f"\n[Batch retry {retry_pass + 1}/{batch_retries}] Retrying {len(failed_pairs)} failed (row, model) pairs...")
        time.sleep(retry_delay)

        retry_success = 0
        still_failed = []

        for idx, model_name in failed_pairs:
            cfg = next((c for c in model_configs if c["sanitized_name"] == model_name), None)
            if not cfg:
                continue

            item = items_to_process[idx]
            model_name_result, json_result, error = summarize_single_item(item, idx, cfg)

            if error and error != "skipped":
                still_failed.append((idx, model_name))
            else:
                # Update the stored result
                all_results[idx]["model_results"][model_name] = json_result
                if model_name in all_results[idx]["errors"]:
                    del all_results[idx]["errors"][model_name]
                retry_success += 1

        failed_pairs = still_failed
        print(f"  -> {retry_success}/{len(failed_pairs) + retry_success} pairs succeeded on retry")

    # Build output DataFrame
    print("\nBuilding output DataFrame...")

    rows = []
    for entry in all_results:
        # Handle PDF mode: extract metadata from tuple
        item = entry["survey_input"]
        if is_pdf_mode and isinstance(item, tuple) and len(item) == 3:
            pdf_path, page_index, page_label = item
            row = {
                "survey_input": page_label,
                "pdf_path": pdf_path,
                "page_index": page_index,
            }
            original_text_for_synthesis = page_label  # Use page label for synthesis context
        else:
            row = {"survey_input": item}
            original_text_for_synthesis = item

        # Extract summaries from each model
        summaries = {}
        for model_name, json_str in entry["model_results"].items():
            is_valid, summary_text = extract_summary_from_json(json_str)
            if is_valid:
                summaries[model_name] = summary_text
            else:
                summaries[model_name] = ""

        # For multi-model: synthesize consensus
        if len(model_configs) > 1:
            # Add individual model summaries
            for model_name in model_names:
                row[f"summary_{model_name}"] = summaries.get(model_name, "")

            # Synthesize consensus summary using the first successful model
            valid_summaries = {k: v for k, v in summaries.items() if v}
            if valid_summaries:
                # Use the first model config for synthesis
                synthesis_cfg = model_configs[0]
                consensus = _synthesize_summaries(
                    summaries=valid_summaries,
                    original_text=str(original_text_for_synthesis),
                    synthesis_config=synthesis_cfg,
                    max_retries=max_retries,
                )
                row["summary"] = consensus
            else:
                row["summary"] = ""

            # Track failed models
            row["failed_models"] = ",".join(entry["errors"].keys()) if entry["errors"] else ""

        else:
            # Single model: just use the summary directly
            model_name = model_names[0]
            row["summary"] = summaries.get(model_name, "")

        # Processing status
        if all(not s for s in summaries.values()):
            # For PDF mode, check if it's a valid tuple (never skip PDFs)
            if is_pdf_mode:
                row["processing_status"] = "error"
            elif item is None or (isinstance(item, str) and not item.strip()):
                row["processing_status"] = "skipped"
            else:
                row["processing_status"] = "error"
        elif any(not s for s in summaries.values()):
            row["processing_status"] = "partial"
        else:
            row["processing_status"] = "success"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to file if requested
    if filename:
        save_path = os.path.join(save_directory, filename) if save_directory else filename
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")

    return df


def _synthesize_summaries(
    summaries: dict,
    original_text: str,
    synthesis_config: dict,
    max_retries: int = 3,
) -> str:
    """
    Synthesize multiple model summaries into one consensus summary.

    Args:
        summaries: Dict of {model_name: summary_text}
        original_text: The original text that was summarized
        synthesis_config: Model config to use for synthesis
        max_retries: Max retries for synthesis call

    Returns:
        Synthesized consensus summary string
    """
    if len(summaries) == 1:
        return list(summaries.values())[0]

    # Build synthesis prompt
    summaries_text = "\n".join([
        f"- {model}: \"{summary}\""
        for model, summary in summaries.items()
    ])

    # Truncate original text if too long
    max_original_len = 500
    original_display = original_text[:max_original_len]
    if len(original_text) > max_original_len:
        original_display += "..."

    synthesis_prompt = f"""You are synthesizing multiple AI-generated summaries of the same text into one optimal summary.

Original text: "{original_display}"

Summaries from different models:
{summaries_text}

Create a single, comprehensive summary that captures the best insights from all summaries.
Resolve any contradictions by focusing on accuracy.

Provide your answer in JSON format: {{"summary": "your synthesized summary"}}"""

    try:
        client = UnifiedLLMClient(
            provider=synthesis_config["provider"],
            api_key=synthesis_config["api_key"],
            model=synthesis_config["model"],
        )

        json_schema = build_summary_json_schema(
            include_additional_properties=synthesis_config["provider"] != "google"
        )

        response = client.complete(
            messages=[{"role": "user", "content": synthesis_prompt}],
            json_schema=json_schema,
            creativity=0.3,  # Low creativity for synthesis
            max_retries=max_retries,
        )

        json_str = extract_json(response)
        is_valid, summary = extract_summary_from_json(json_str)

        if is_valid:
            return summary
        else:
            # Fallback: return the longest summary
            return max(summaries.values(), key=len)

    except Exception as e:
        print(f"Warning: Synthesis failed ({e}), using longest summary as fallback")
        return max(summaries.values(), key=len)
