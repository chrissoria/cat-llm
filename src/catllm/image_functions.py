import warnings

from .text_functions import _detect_model_source
from .calls.image_stepback import get_image_stepback_insight

# Exported names (excludes deprecated image_multi_class)
__all__ = [
    "_load_image_files",
    "_encode_image",
    "image_score_drawing",
    "image_features",
    "explore_image_categories",
]
from .calls.image_CoVe import (
    image_chain_of_verification_openai,
    image_chain_of_verification_anthropic,
    image_chain_of_verification_google,
    image_chain_of_verification_mistral
)


def _load_image_files(image_input):
    """Load image files from directory path, single file path, or return list as-is."""
    import os
    import glob

    image_extensions = [
        '*.png', '*.jpg', '*.jpeg',
        '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
        '*.tif', '*.tiff', '*.bmp',
        '*.heif', '*.heic', '*.ico',
        '*.psd'
    ]

    if isinstance(image_input, list):
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")
    elif os.path.isfile(image_input):
        # Single file path
        image_files = [image_input]
        print(f"Provided 1 image file.")
    elif os.path.isdir(image_input):
        # Directory path - glob for images
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))
        print(f"Found {len(image_files)} images in directory.")
    else:
        raise FileNotFoundError(f"Image input not found: {image_input}")

    return image_files


def _encode_image(img_path):
    """Encode an image file to base64. Returns (encoded_data, extension, is_valid)."""
    import os
    import base64
    from pathlib import Path

    if img_path is None or not os.path.exists(img_path):
        return None, None, False

    if os.path.isdir(img_path):
        return None, None, False

    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(img_path).suffix.lstrip(".").lower()
        return encoded, ext, True
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None, None, False


# image multi-class (binary) function
def image_multi_class(
    image_description,
    image_input,
    categories,
    api_key,
    user_model="gpt-4o",
    creativity=None,
    safety=False,
    chain_of_verification=False,
    chain_of_thought=True,
    step_back_prompt=False,
    context_prompt=False,
    thinking_budget=0,
    example1=None,
    example2=None,
    example3=None,
    example4=None,
    example5=None,
    example6=None,
    filename=None,
    save_directory=None,
    model_source="auto"
):
    """
    Classify images using LLMs.

    .. deprecated::
        Use :func:`catllm.classify` instead. This function will be removed in a future version.
    """
    warnings.warn(
        "image_multi_class() is deprecated and will be removed in a future version. "
        "Use catllm.classify() instead, which auto-detects image input.",
        DeprecationWarning,
        stacklevel=2,
    )

    import os
    import json
    import pandas as pd
    import regex
    import time
    from tqdm import tqdm

    if save_directory is not None and not os.path.isdir(save_directory):
        raise FileNotFoundError(f"Directory {save_directory} doesn't exist")

    model_source = _detect_model_source(user_model, model_source)

    image_files = _load_image_files(image_input)

    # Handle "auto" categories - extract categories first
    if categories == "auto":
        if not image_description:
            raise ValueError("image_description is required when using categories='auto'")

        print("\nAuto-extracting categories from images...")
        auto_result = explore_image_categories(
            image_input=image_input,
            api_key=api_key,
            image_description=image_description,
            user_model=user_model,
            model_source=model_source,
            creativity=creativity
        )
        categories = auto_result["top_categories"]
        print(f"Extracted {len(categories)} categories: {categories}\n")

    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    print(f"\nCategories to classify by {model_source} {user_model}:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")

    # Build examples text from provided examples
    examples = [example1, example2, example3, example4, example5, example6]
    examples = [ex for ex in examples if ex is not None]
    if examples:
        examples_text = "Here are some examples of how to categorize:\n" + "\n".join(examples)
    else:
        examples_text = ""

    # Helper function for CoVe
    def remove_numbering(line):
        line = line.strip()
        if line.startswith('- '):
            return line[2:].strip()
        if line.startswith('• '):
            return line[2:].strip()
        if line and line[0].isdigit():
            i = 0
            while i < len(line) and line[i].isdigit():
                i += 1
            if i < len(line) and line[i] in '.':
                return line[i+1:].strip()
            elif i < len(line) and line[i] in ')':
                return line[i+1:].strip()
        return line

    # Step-back insight initialization
    if step_back_prompt:
        stepback = f"""What are the key visual features or patterns that typically indicate the presence of these categories in images showing "{image_description}"?

Categories to consider:
{categories_str}

Provide a brief analysis of what visual cues to look for when categorizing such images."""

        stepback_insight, step_back_added = get_image_stepback_insight(
            model_source, stepback, api_key, user_model, creativity
        )
    else:
        stepback_insight = None
        step_back_added = False

    link1 = []
    extracted_jsons = []

    def _build_base_prompt_text():
        """Build the base text portion of the prompt."""
        if chain_of_thought:
            base_text = (
                f"You are an image-tagging assistant.\n"
                f"Task ► Examine the attached image and decide, **for each category below**, "
                f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                f"Image is expected to show: {image_description}\n\n"
                f"Categories:\n{categories_str}\n\n"
                f"Let's analyze step by step:\n"
                f"1. First, identify the key visual elements in the image\n"
                f"2. Then, match each element to the relevant categories\n"
                f"3. Finally, assign 1 to matching categories and 0 to non-matching categories\n\n"
                f"{examples_text}\n\n"
                f"Output format ► Respond with **only** a JSON object whose keys are the "
                f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                f"No additional keys, comments, or text.\n\n"
                f"Example (three categories):\n"
                f"{example_JSON}"
            )
        else:
            base_text = (
                f"You are an image-tagging assistant.\n"
                f"Task ► Examine the attached image and decide, **for each category below**, "
                f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                f"Image is expected to show: {image_description}\n\n"
                f"Categories:\n{categories_str}\n\n"
                f"{examples_text}\n\n"
                f"Output format ► Respond with **only** a JSON object whose keys are the "
                f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                f"No additional keys, comments, or text.\n\n"
                f"Example (three categories):\n"
                f"{example_JSON}"
            )

        if context_prompt:
            context = (
                "You are an expert visual analyst specializing in image categorization. "
                "Apply multi-label classification based on explicit and implicit visual cues. "
                "When uncertain, prioritize precision over recall.\n\n"
            )
            base_text = context + base_text

        return base_text

    def _build_cove_prompts(base_prompt_text):
        """Build chain of verification prompts for images."""
        step2_prompt = f"""You provided this initial categorization:
<<INITIAL_REPLY>>

Original task: {base_prompt_text}

Generate a focused list of 3-5 verification questions to fact-check your categorization. Each question should:
- Be concise and specific (one sentence)
- Address a distinct visual element or category assignment
- Be answerable by re-examining the image

Focus on verifying:
- Whether each category assignment matches what's visible in the image
- Whether any visual elements were missed or misinterpreted
- Whether there are any logical inconsistencies

Provide only the verification questions as a numbered list."""

        step3_prompt = f"""Re-examine the attached image and answer the following verification question.

Image description: {image_description}

Verification question: <<QUESTION>>

Provide a brief, direct answer (1-2 sentences maximum) based on what you observe in the image.

Answer:"""

        step4_prompt = f"""Original task: {base_prompt_text}
Initial categorization:
<<INITIAL_REPLY>>
Verification questions and answers:
<<VERIFICATION_QA>>
Based on this verification, provide the final corrected categorization.
If no categories are present, assign "0" to all categories.
Provide the final categorization in the same JSON format:"""

        return step2_prompt, step3_prompt, step4_prompt

    def _build_prompt_openai_mistral(encoded, ext, base_text):
        """Build prompt for OpenAI/Mistral format."""
        encoded_image = f"data:image/{ext};base64,{encoded}"
        return [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": encoded_image, "detail": "high"}},
        ]

    def _build_prompt_anthropic(encoded, ext, base_text):
        """Build prompt for Anthropic format."""
        media_type = f"image/{ext}" if ext else "image/jpeg"
        return [
            {"type": "text", "text": base_text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded
                }
            }
        ]

    def _build_prompt_google(encoded, ext, base_text):
        """Build prompt for Google format."""
        return {
            "text_prompt": base_text,
            "image_data": encoded,
            "mime_type": f"image/{ext}" if ext else "image/jpeg"
        }

    def _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt, image_content):
        """Handle OpenAI-compatible API calls (OpenAI, Perplexity, HuggingFace, xAI)."""
        import requests as req

        # Determine the base URL based on model source
        if model_source == "huggingface":
            from catllm.text_functions import _detect_huggingface_endpoint
            base_url = _detect_huggingface_endpoint(api_key, user_model)
        elif model_source == "huggingface-together":
            base_url = "https://router.huggingface.co/together/v1"
        elif model_source == "perplexity":
            base_url = "https://api.perplexity.ai"
        elif model_source == "xai":
            base_url = "https://api.x.ai/v1"
        else:
            base_url = "https://api.openai.com/v1"

        endpoint = f"{base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        max_retries = 8
        delay = 2

        for attempt in range(max_retries):
            try:
                # Build messages with optional stepback
                messages = []
                if step_back_prompt and step_back_added:
                    messages.append({'role': 'user', 'content': stepback})
                    messages.append({'role': 'assistant', 'content': stepback_insight})
                messages.append({'role': 'user', 'content': prompt})

                payload = {
                    "model": user_model,
                    "messages": messages,
                }
                if creativity is not None:
                    payload["temperature"] = creativity

                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]

                if chain_of_verification:
                    reply = image_chain_of_verification_openai(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=None,  # Not used anymore, CoVe needs refactoring too
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        image_content=image_content
                    )

                return reply, None

            except req.exceptions.HTTPError as e:
                error_str = str(e).lower()
                status_code = e.response.status_code if e.response else None

                if status_code == 400 and "json_validate_failed" in error_str and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ JSON validation failed. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif status_code == 404:
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
                elif status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

            except Exception as e:
                if ("500" in str(e) or "504" in str(e)) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

        return """{"1":"e"}""", "Max retries exceeded"

    def _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt, image_content):
        """Handle Anthropic API calls using direct HTTP requests."""
        import requests as req

        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        try:
            # Build messages with optional stepback
            messages = []
            if step_back_prompt and step_back_added:
                messages.append({'role': 'user', 'content': stepback})
                messages.append({'role': 'assistant', 'content': stepback_insight})
            messages.append({'role': 'user', 'content': prompt})

            payload = {
                "model": user_model,
                "max_tokens": 1024,
                "messages": messages,
            }
            if creativity is not None:
                payload["temperature"] = creativity

            response = req.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            content = result.get("content", [])
            if content and content[0].get("type") == "text":
                reply = content[0].get("text", "")
            else:
                return """{"1":"e"}""", "No text content in response"

            if chain_of_verification:
                reply = image_chain_of_verification_anthropic(
                    initial_reply=reply,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    client=None,  # No longer using SDK client
                    user_model=user_model,
                    creativity=creativity,
                    remove_numbering=remove_numbering,
                    image_content=image_content,
                    api_key=api_key  # Pass api_key for HTTP calls
                )

            return reply, None

        except req.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_google(prompt_data, step2_prompt, step3_prompt, step4_prompt, base_prompt_text):
        """Handle Google API calls."""
        import requests

        def make_google_request(url, headers, payload, max_retries=8):
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code
                    retryable_errors = [429, 500, 502, 503, 504]

                    if status_code in retryable_errors and attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt) if status_code == 429 else 2 * (2 ** attempt)
                        error_type = "Rate limited" if status_code == 429 else f"Server error {status_code}"
                        print(f"⚠️ {error_type}. Attempt {attempt + 1}/{max_retries}")
                        print(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        # Build parts with optional stepback context
        parts = []
        if step_back_prompt and step_back_added:
            parts.append({"text": f"Context from step-back analysis:\n{stepback_insight}\n\n"})
        parts.append({"text": prompt_data["text_prompt"]})
        parts.append({
            "inline_data": {
                "mime_type": prompt_data["mime_type"],
                "data": prompt_data["image_data"]
            }
        })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({"temperature": creativity} if creativity is not None else {}),
                **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget else {})
            }
        }

        try:
            result = make_google_request(url, headers, payload)

            if "candidates" in result and result["candidates"]:
                reply = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response generated", None

            if chain_of_verification:
                reply = image_chain_of_verification_google(
                    initial_reply=reply,
                    prompt=base_prompt_text,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    url=url,
                    headers=headers,
                    creativity=creativity,
                    remove_numbering=remove_numbering,
                    make_google_request=make_google_request,
                    image_data=prompt_data["image_data"],
                    mime_type=prompt_data["mime_type"]
                )

            return reply, None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"❌ Model '{user_model}' not found. Please check the model name and try again.") from e
            elif e.response.status_code in [401, 403]:
                raise ValueError(f"❌ Authentication failed. Please check your Google API key.") from e
            else:
                print(f"HTTP error occurred: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt, image_content):
        """Handle Mistral API calls - uses requests directly."""
        import requests as req

        endpoint = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        max_retries = 8
        delay = 2

        for attempt in range(max_retries):
            try:
                # Build messages with optional stepback
                messages = []
                if step_back_prompt and step_back_added:
                    messages.append({'role': 'user', 'content': stepback})
                    messages.append({'role': 'assistant', 'content': stepback_insight})
                messages.append({'role': 'user', 'content': prompt})

                payload = {
                    "model": user_model,
                    "messages": messages,
                }
                if creativity is not None:
                    payload["temperature"] = creativity

                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]

                if chain_of_verification:
                    reply = image_chain_of_verification_mistral(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=None,  # Not used anymore, CoVe needs refactoring too
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        image_content=image_content
                    )

                return reply, None

            except req.exceptions.HTTPError as e:
                error_str = str(e).lower()
                status_code = e.response.status_code if e.response else None

                if status_code == 404 or "invalid_model" in error_str or "invalid model" in error_str:
                    raise ValueError(f"❌ Model '{user_model}' not found.") from e
                elif status_code == 401 or "unauthorized" in error_str:
                    raise ValueError(f"❌ Authentication failed. Please check your Mistral API key.") from e
                elif status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ Server error {status_code}. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"

        return """{"1":"e"}""", "Max retries exceeded"

    def _process_single_image(img_path):
        """Process a single image and return (reply, error_msg)."""
        encoded, ext, is_valid = _encode_image(img_path)

        if not is_valid:
            return None, "Invalid image path or encoding failed"

        base_prompt_text = _build_base_prompt_text()

        if chain_of_verification:
            step2_prompt, step3_prompt, step4_prompt = _build_cove_prompts(base_prompt_text)
        else:
            step2_prompt = step3_prompt = step4_prompt = None

        if model_source in ["openai", "perplexity", "huggingface", "xai"]:
            prompt = _build_prompt_openai_mistral(encoded, ext, base_prompt_text)
            # Image content for CoVe (just the image part)
            encoded_image = f"data:image/{ext};base64,{encoded}"
            image_content = {"type": "image_url", "image_url": {"url": encoded_image, "detail": "high"}}
            return _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt, image_content)

        elif model_source == "anthropic":
            prompt = _build_prompt_anthropic(encoded, ext, base_prompt_text)
            media_type = f"image/{ext}" if ext else "image/jpeg"
            image_content = {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded}
            }
            return _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt, image_content)

        elif model_source == "google":
            prompt_data = _build_prompt_google(encoded, ext, base_prompt_text)
            return _call_google(prompt_data, step2_prompt, step3_prompt, step4_prompt, base_prompt_text)

        elif model_source == "mistral":
            prompt = _build_prompt_openai_mistral(encoded, ext, base_prompt_text)
            encoded_image = f"data:image/{ext};base64,{encoded}"
            image_content = {"type": "image_url", "image_url": {"url": encoded_image, "detail": "high"}}
            return _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt, image_content)

        else:
            raise ValueError("Unknown source! Choose from OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral")

    def _extract_json(reply):
        """Extract JSON from model reply."""
        if reply is None:
            return """{"1":"e"}"""

        if reply == "invalid image path":
            return """{"no_valid_path": 1}"""

        extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
        if extracted_json:
            return extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
        else:
            print("""{"1":"e"}""")
            return """{"1":"e"}"""

    # Main processing loop
    for idx, img_path in enumerate(tqdm(image_files, desc="Categorizing images")):
        if img_path is None:
            link1.append("Skipped NaN input")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue

        reply, error_msg = _process_single_image(img_path)

        if error_msg:
            link1.append(error_msg)
            if "Invalid image" in error_msg:
                extracted_jsons.append("""{"no_valid_path": 1}""")
            else:
                extracted_jsons.append(_extract_json(reply))
        else:
            link1.append(reply)
            extracted_jsons.append(_extract_json(reply))

        # --- Safety Save ---
        if safety:
            if filename is None:
                raise TypeError("filename is required when using safety. Please provide the filename.")

            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)

            temp_df = pd.DataFrame({
                'image_input': image_files[:idx+1],
                'model_response': link1,
                'json': extracted_jsons
            })
            temp_df = pd.concat([temp_df, normalized_data], axis=1)

            save_path = os.path.join(save_directory, filename) if save_directory else filename
            temp_df.to_csv(save_path, index=False)

    # --- Final DataFrame ---
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed_obj = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed_obj))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    categorized_data = pd.DataFrame({
        'image_input': pd.Series(image_files),
        'model_response': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    categorized_data = categorized_data.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

    # Identify rows with invalid strings (like "e")
    cat_cols = [col for col in categorized_data.columns if col.startswith('category_')]
    has_invalid_strings = categorized_data[cat_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce').isna() & col.notna()
    ).any(axis=1)

    categorized_data['processing_status'] = (~has_invalid_strings).map({True: 'success', False: 'error'})
    categorized_data.loc[has_invalid_strings, cat_cols] = pd.NA

    for col in cat_cols:
        categorized_data[col] = pd.to_numeric(categorized_data[col], errors='coerce')

    categorized_data.loc[~has_invalid_strings, cat_cols] = (
        categorized_data.loc[~has_invalid_strings, cat_cols].fillna(0)
    )
    categorized_data[cat_cols] = categorized_data[cat_cols].astype('Int64')

    # Create categories_id (comma-separated binary values for each category)
    categorized_data['categories_id'] = categorized_data[cat_cols].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1
    )

    if filename:
        save_path = os.path.join(save_directory, filename) if save_directory else filename
        categorized_data.to_csv(save_path, index=False)

    return categorized_data


# image score function
def image_score_drawing(
    reference_image_description,
    image_input,
    reference_image,
    api_key,
    columns="numbered",
    user_model="gpt-4o-2024-11-20",
    creativity=None,
    to_csv=False,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import glob
    import base64
    from pathlib import Path

    if save_directory is not None and not os.path.isdir(save_directory):
    # Directory doesn't exist - raise an exception to halt execution
        raise FileNotFoundError(f"Directory {save_directory} doesn't exist")

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

    model_source = model_source.lower() # eliminating case sensitivity

    if not isinstance(image_input, list):
        # If image_input is a filepath (string)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))

        print(f"Found {len(image_files)} images.")
    else:
        # If image_files is already a list
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")

    with open(reference_image, 'rb') as f:
        reference = base64.b64encode(f.read()).decode('utf-8')
        reference_image = f"data:image/{reference_image.split('.')[-1]};base64,{reference}"

    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Categorising images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration

    # Only open the file if path is valid
        if os.path.isdir(img_path):
            encoded = "Not a Valid Image, contains file path"
        else:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                    encoded = f"Error: {str(e)}"
    # Handle extension safely
        if encoded.startswith("Error:") or encoded == "Not a Valid Image, contains file path":
            encoded_image = encoded
            valid_image = False

        else:
            ext = Path(img_path).suffix.lstrip(".").lower()
            encoded_image = f"data:image/{ext};base64,{encoded}"
            valid_image = True

    # Handle extension safely
        ext = Path(img_path).suffix.lstrip(".").lower()
        encoded_image = f"data:image/{ext};base64,{encoded}"

        if model_source == "openai" or model_source == "mistral":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual similarity assessment system.\n"
                        f"Task ► Compare these two images:\n"
                        f"1. REFERENCE (left): {reference_image_description}\n"
                        f"2. INPUT (right): User-provided drawing\n\n"
                        f"Rating criteria:\n"
                        f"1: No meaningful similarity (fundamentally different)\n"
                        f"2: Barely recognizable similarity (25% match)\n"
                        f"3: Partial match (50% key features)\n"
                        f"4: Strong alignment (75% features)\n"
                        f"5: Near-perfect match (90%+ similarity)\n\n"
                        f"Output format ► Return ONLY:\n"
                        "{\n"
                        '  "score": [1-5],\n'
                        '  "summary": "reason you scored"\n'
                        "}\n\n"
                        f"Critical rules:\n"
                        f"- Score must reflect shape, proportions, and key details\n"
                        f"- List only concrete matching elements from reference\n"
                        f"- No markdown or additional text"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": reference_image, "detail": "high"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": encoded_image, "detail": "high"}
                }
            ]

        elif model_source == "anthropic":  # Changed to elif
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual similarity assessment system.\n"
                        f"Task ► Compare these two images:\n"
                        f"1. REFERENCE (left): {reference_image_description}\n"
                        f"2. INPUT (right): User-provided drawing\n\n"
                        f"Rating criteria:\n"
                        f"1: No meaningful similarity (fundamentally different)\n"
                        f"2: Barely recognizable similarity (25% match)\n"
                        f"3: Partial match (50% key features)\n"
                        f"4: Strong alignment (75% features)\n"
                        f"5: Near-perfect match (90%+ similarity)\n\n"
                        f"Output format ► Return ONLY:\n"
                        "{\n"
                        '  "score": [1-5],\n'
                        '  "summary": "reason you scored"\n'
                        "}\n\n"
                        f"Critical rules:\n"
                        f"- Score must reflect shape, proportions, and key details\n"
                        f"- List only concrete matching elements from reference\n"
                        f"- No markdown or additional text"
                    )
                },
                {
                    "type": "image",  # Added missing type
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": reference
                    }
                },
                {
                    "type": "image",  # Added missing type
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                }
            ]


        if model_source == "openai":
            import requests as req
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": user_model,
                "messages": [{'role': 'user', 'content': prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                link1.append(reply)
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "anthropic":
            import requests as req
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": user_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                content = result.get("content", [])
                if content and content[0].get("type") == "text":
                    reply = content[0].get("text", "")
                    link1.append(reply)
                else:
                    link1.append("Error processing input: No text content in response")
            except req.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    raise ValueError(f"Invalid Anthropic model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "mistral":
            import requests as req
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": user_model,
                "messages": [{'role': 'user', 'content': prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                link1.append(reply)
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    raise ValueError(f"Invalid Mistral model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")
        #if no valid image path is provided
        elif  valid_image == False:
            reply = "invalid image path"
            print("Skipped NaN input or invalid path")
            #extracted_jsons.append("""{"no_valid_path": 1}""")
            link1.append("Error processing input: {e}")
        else:
            raise ValueError("Unknown source! Choose from OpenAI, Perplexity, or Mistral")
            # in situation that no JSON is found
        if reply is not None:
            if reply == "invalid image path":
                extracted_jsons.append("""{"no_valid_path": 1}""")
            else:
                extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
                if extracted_json:
                    cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                    extracted_jsons.append(cleaned_json)
                else:
                    error_message = """{"1":"e"}"""
                    extracted_jsons.append(error_message)
                    print(error_message)
        else:
            error_message = """{"1":"e"}"""
            extracted_jsons.append(error_message)
            print(error_message)

        # --- Safety Save ---
        if safety:
            # Save progress so far
            temp_df = pd.DataFrame({
                'image_input': image_files[:i+1],
                'model_response': link1,
                'json': extracted_jsons
            })
            # Normalize processed jsons so far
            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)
            temp_df = pd.concat([temp_df, normalized_data], axis=1)
            # Save to CSV
            if save_directory is None:
                save_directory = os.getcwd()
            temp_df.to_csv(os.path.join(save_directory, filename), index=False)

    # --- Final DataFrame ---
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed_obj = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed_obj))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    categorized_data = pd.DataFrame({
        'image_input': (
            image_files.reset_index(drop=True) if isinstance(image_files, (pd.DataFrame, pd.Series))
            else pd.Series(image_files)
        ),
        'link1': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)

    if to_csv:
        if save_directory is None:
            save_directory = os.getcwd()
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)

    return categorized_data

# image features function
def image_features(
    image_description,
    image_input,
    features_to_extract,
    api_key,
    user_model="gpt-4o-2024-11-20",
    creativity=None,
    to_csv=False,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import glob
    import base64
    from pathlib import Path

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

    model_source = model_source.lower() # eliminating case sensitivity

    if not isinstance(image_input, list):
        # If image_input is a filepath (string)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))

        print(f"Found {len(image_files)} images.")
    else:
        # If image_files is already a list
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")

    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(features_to_extract))
    cat_num = len(features_to_extract)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Scoring images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration

    # Only open the file if path is valid
        if os.path.isdir(img_path):
            encoded = "Not a Valid Image, contains file path"
        else:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                    encoded = f"Error: {str(e)}"
    # Handle extension safely
        if encoded.startswith("Error:") or encoded == "Not a Valid Image, contains file path":
            encoded_image = encoded
            valid_image = False

        else:
            ext = Path(img_path).suffix.lstrip(".").lower()
            encoded_image = f"data:image/{ext};base64,{encoded}"
            valid_image = True

        if model_source == "openai" or model_source == "mistral":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual question answering assistant.\n"
                        f"Task ► Analyze the attached image and answer these specific questions:\n\n"
                        f"Image context: {image_description}\n\n"
                        f"Questions to answer:\n{categories_str}\n\n"
                        f"Output format ► Return **only** a JSON object where:\n"
                        f"- Keys are question numbers ('1', '2', ...)\n"
                        f"- Values are concise answers (numbers, short phrases)\n\n"
                        f"Example for 3 questions:\n"
                        "{\n"
                        '  "1": "4",\n'
                        '  "2": "blue",\n'
                        '  "3": "yes"\n'
                        "}\n\n"
                        f"Important rules:\n"
                        f"1. Answer directly - no explanations\n"
                        f"2. Use exact numerical values when possible\n"
                        f"3. For yes/no questions, use 'yes' or 'no'\n"
                        f"4. Never add extra keys or formatting"
                        ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_image, "detail": "high"},
                            },
            ]
        elif model_source == "anthropic":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual question answering assistant.\n"
                        f"Task ► Analyze the attached image and answer these specific questions:\n\n"
                        f"Image context: {image_description}\n\n"
                        f"Questions to answer:\n{categories_str}\n\n"
                        f"Output format ► Return **only** a JSON object where:\n"
                        f"- Keys are question numbers ('1', '2', ...)\n"
                        f"- Values are concise answers (numbers, short phrases)\n\n"
                        f"Example for 3 questions:\n"
                        "{\n"
                        '  "1": "4",\n'
                        '  "2": "blue",\n'
                        '  "3": "yes"\n'
                        "}\n\n"
                        f"Important rules:\n"
                        f"1. Answer directly - no explanations\n"
                        f"2. Use exact numerical values when possible\n"
                        f"3. For yes/no questions, use 'yes' or 'no'\n"
                        f"4. Never add extra keys or formatting"
                    )
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                }
            ]
        if model_source == "openai":
            import requests as req
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": user_model,
                "messages": [{'role': 'user', 'content': prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                link1.append(reply)
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "perplexity":
            import requests as req
            endpoint = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": user_model,
                "messages": [{'role': 'user', 'content': prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                link1.append(reply)
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    raise ValueError(f"Invalid Perplexity model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "anthropic":
            import requests as req
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": user_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                content = result.get("content", [])
                if content and content[0].get("type") == "text":
                    reply = content[0].get("text", "")
                    link1.append(reply)
                else:
                    link1.append("Error processing input: No text content in response")
            except req.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    raise ValueError(f"Invalid Anthropic model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "mistral":
            import requests as req
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": user_model,
                "messages": [{'role': 'user', 'content': prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            try:
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                link1.append(reply)
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    raise ValueError(f"Invalid Mistral model '{user_model}': {e}")
                else:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif  valid_image == False:
            print("Skipped NaN input or invalid path")
            reply = None
            link1.append(f"Error processing input: invalid image")
        else:
            raise ValueError("Unknown source! Choose from OpenAI, Anthropic, Perplexity, or Mistral")
            # in situation that no JSON is found
        if reply is not None:
            extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
            if extracted_json:
                cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                extracted_jsons.append(cleaned_json)
                #print(cleaned_json)
            else:
                error_message = """{"1":"e"}"""
                extracted_jsons.append(error_message)
                print(error_message)
        else:
            error_message = """{"1":"e"}"""
            extracted_jsons.append(error_message)
            #print(error_message)

        # --- Safety Save ---
        if safety:
            #print(f"Saving CSV to: {save_directory}")
            # Save progress so far
            temp_df = pd.DataFrame({
                'image_input': image_files[:i+1],
                'link1': link1,
                'json': extracted_jsons
            })
            # Normalize processed jsons so far
            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)
            temp_df = pd.concat([temp_df, normalized_data], axis=1)
            # Save to CSV
            if save_directory is None:
                save_directory = os.getcwd()
            temp_df.to_csv(os.path.join(save_directory, filename), index=False)

    # --- Final DataFrame ---
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed_obj = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed_obj))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    categorized_data = pd.DataFrame({
        'image_input': (
            image_files.reset_index(drop=True) if isinstance(image_files, (pd.DataFrame, pd.Series))
            else pd.Series(image_files)
        ),
        'model_response': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)

    if to_csv:
        if save_directory is None:
            save_directory = os.getcwd()
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)

    return categorized_data


def explore_image_categories(
    image_input,
    api_key,
    image_description="",
    max_categories=12,
    categories_per_chunk=10,
    divisions=5,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    mode="image",
    filename=None,
    model_source="auto",
    iterations=3,
    random_state=None,
    progress_callback=None,
):
    """
    Explore and extract common categories from a collection of images.

    Modes:
        - "image" (default): Samples random images and sends them directly to
          a vision model for category extraction. Best for visual categorization.

        - "both": Samples random images, uses vision model to describe each
          image's content (including any text), then extracts categories from
          those descriptions. Best for images that contain text or mixed content.

    Args:
        image_input: Path to image file, directory of images, or list of image paths
        api_key: API key for the model provider
        image_description: Description of what the images contain
        max_categories: Maximum number of final categories to return
        categories_per_chunk: Categories to extract per chunk of images
        divisions: Number of chunks to divide images into
        user_model: Model to use (must support vision)
        creativity: Temperature setting (None for default)
        specificity: "broad" or "specific" category granularity
        research_question: Optional research context
        mode: "image" or "both"
        filename: Optional CSV filename to save results
        model_source: "auto", "openai", "anthropic", "google", "mistral"
        iterations: Number of passes over the data
        random_state: Random seed for reproducibility
        progress_callback: Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step
    """
    import os
    import re
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    model_source = _detect_model_source(user_model, model_source)

    # Load all images
    image_files = _load_image_files(image_input)
    if not image_files:
        raise ValueError("No image files found in the specified input.")

    n = len(image_files)
    if n == 0:
        raise ValueError("No images found.")

    # Auto-adjust divisions for small datasets
    # Images can have multiple categories each, so we can use fewer divisions
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 2))  # At least 2 images per chunk
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} images.")

    # Chunk sizing - images often contain multiple categories each
    chunk_size = int(round(max(1, n / divisions), 0))
    # Don't reduce categories_per_chunk as aggressively for images since each image can yield many categories
    if chunk_size < 2:
        # Only reduce if we have very few images
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(5, chunk_size * 4)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    print(
        f"Exploring categories in images: '{image_description}'.\n"
        f"          {n} total images, {categories_per_chunk * divisions} categories to extract, "
        f"{max_categories} final categories. Mode: {mode}\n"
    )

    # RNG for reproducible sampling
    rng = np.random.default_rng(random_state)

    # Validate model_source (clients initialized per-call using requests)
    import requests as req
    if model_source not in ["openai", "huggingface", "huggingface-together", "xai", "anthropic", "google", "mistral"]:
        raise ValueError(f"Unsupported model_source: {model_source}")

    # Determine base URL for OpenAI-compatible providers
    if model_source == "huggingface":
        from catllm.text_functions import _detect_huggingface_endpoint
        openai_base_url = _detect_huggingface_endpoint(api_key, user_model)
    elif model_source == "huggingface-together":
        openai_base_url = "https://router.huggingface.co/together/v1"
    elif model_source == "xai":
        openai_base_url = "https://api.x.ai/v1"
    elif model_source == "openai":
        openai_base_url = "https://api.openai.com/v1"
    else:
        openai_base_url = None  # Not an OpenAI-compatible provider

    def make_image_prompt() -> str:
        """Build prompt for image mode - direct category extraction."""
        return (
            f"Identify {categories_per_chunk} {specificity} categories of content found in this image. "
            f"The image is: {image_description}. "
            f"{'Research context: ' + research_question if research_question else ''}\n\n"
            f"Number your categories from 1 through {categories_per_chunk} and provide concise labels only (no descriptions)."
        )

    def make_describe_prompt() -> str:
        """Build prompt for 'both' mode - describe image content."""
        return (
            f"Describe the content of this image in detail. "
            f"Include all visual elements, text, objects, people, and any other content. "
            f"The image is: {image_description}. "
            f"{'Research context: ' + research_question if research_question else ''}\n\n"
            f"Provide a comprehensive text description that captures both visual and textual content."
        )

    def make_text_prompt(text_blob: str) -> str:
        """Build prompt for extracting categories from text descriptions."""
        return (
            f"Identify {categories_per_chunk} {specificity} categories of content found in this description. "
            f"The content is: {image_description}. "
            f"{'Research context: ' + research_question + '. ' if research_question else ''}"
            f"The description is contained within triple backticks: ```{text_blob}``` "
            f"Number your categories from 1 through {categories_per_chunk} and provide concise labels only (no descriptions)."
        )

    def call_model_with_image(img_path, prompt_text):
        """Send an image to the model and get category extraction."""
        encoded, ext, is_valid = _encode_image(img_path)
        if not is_valid:
            return None

        try:
            if model_source in ["openai", "huggingface", "huggingface-together", "xai"]:
                endpoint = f"{openai_base_url}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded}"}}
                    ]
                }]
                payload = {"model": user_model, "messages": messages}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

            elif model_source == "anthropic":
                endpoint = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                media_type = f"image/{ext}" if ext else "image/jpeg"
                content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}
                ]
                payload = {
                    "model": user_model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": content}],
                }
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                resp_content = result.get("content", [])
                if resp_content and resp_content[0].get("type") == "text":
                    return resp_content[0].get("text", "")
                return None

            elif model_source == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
                mime_type = f"image/{ext}" if ext else "image/jpeg"
                parts = [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": mime_type, "data": encoded}}
                ]
                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {**({"temperature": creativity} if creativity is not None else {})}
                }
                response = req.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                return None

            elif model_source == "mistral":
                endpoint = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded}"}}
                    ]
                }]
                payload = {"model": user_model, "messages": messages}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    def describe_image_with_vision(img_path):
        """Use vision model to describe an image's content as text."""
        encoded, ext, is_valid = _encode_image(img_path)
        if not is_valid:
            return None
        prompt_text = make_describe_prompt()

        try:
            if model_source in ["openai", "huggingface", "huggingface-together", "xai"]:
                endpoint = f"{openai_base_url}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded}"}}
                    ]
                }]
                payload = {"model": user_model, "messages": messages}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

            elif model_source == "anthropic":
                endpoint = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                media_type = f"image/{ext}" if ext else "image/jpeg"
                content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}
                ]
                payload = {
                    "model": user_model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": content}],
                }
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                resp_content = result.get("content", [])
                if resp_content and resp_content[0].get("type") == "text":
                    return resp_content[0].get("text", "")
                return None

            elif model_source == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
                mime_type = f"image/{ext}" if ext else "image/jpeg"
                parts = [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": mime_type, "data": encoded}}
                ]
                payload = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {**({"temperature": creativity} if creativity is not None else {})}
                }
                response = req.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                return None

            elif model_source == "mistral":
                endpoint = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded}"}}
                    ]
                }]
                payload = {"model": user_model, "messages": messages}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error describing image {img_path}: {e}")
            return None

    def call_model_with_text(prompt_text):
        """Send text to the model for category extraction."""
        try:
            if model_source in ["openai", "huggingface", "huggingface-together", "xai"]:
                endpoint = f"{openai_base_url}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                payload = {"model": user_model, "messages": [{"role": "user", "content": prompt_text}]}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

            elif model_source == "anthropic":
                endpoint = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": user_model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt_text}],
                }
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                resp_content = result.get("content", [])
                if resp_content and resp_content[0].get("type") == "text":
                    return resp_content[0].get("text", "")
                return None

            elif model_source == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {**({"temperature": creativity} if creativity is not None else {})}
                }
                response = req.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                return None

            elif model_source == "mistral":
                endpoint = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                payload = {"model": user_model, "messages": [{"role": "user", "content": prompt_text}]}
                if creativity is not None:
                    payload["temperature"] = creativity
                response = req.post(endpoint, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error in text mode: {e}")
            return None

    # Parse numbered list pattern
    line_pat = re.compile(r"^\s*\d+\s*[\.\)\-]\s*(.+)$")

    all_items = []

    # Calculate total steps for progress tracking: (iterations * divisions) + 1 for final merge
    total_steps = (iterations * divisions) + 1
    current_step = 0

    for pass_idx in range(iterations):
        # Sample images for this pass
        image_indices = list(range(n))
        rng.shuffle(image_indices)

        # Create chunks
        chunks = [image_indices[i:i + chunk_size] for i in range(0, len(image_indices), chunk_size)][:divisions]

        for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"Processing chunks (pass {pass_idx+1}/{iterations})")):
            if not chunk:
                continue

            # Sample one random image from the full pool
            random_idx = rng.choice(image_indices)
            img_path = image_files[random_idx]

            if mode == "image":
                # IMAGE MODE: Send image directly for category extraction
                prompt = make_image_prompt()
                reply = call_model_with_image(img_path, prompt)

            elif mode == "both":
                # BOTH MODE: Describe image first, then extract categories from description
                image_description_text = describe_image_with_vision(img_path)
                if not image_description_text:
                    continue

                prompt = make_text_prompt(image_description_text)
                reply = call_model_with_text(prompt)

            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'image' or 'both'.")

            if reply:
                # Extract numbered items
                items = []
                for raw_line in reply.splitlines():
                    m = line_pat.match(raw_line.strip())
                    if m:
                        items.append(m.group(1).strip())
                # Fallback for unnumbered lines
                if not items:
                    for raw_line in reply.splitlines():
                        s = raw_line.strip()
                        if s:
                            items.append(s)
                all_items.extend(items)

            # Progress callback
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"Pass {pass_idx+1}/{iterations}, chunk {chunk_idx+1}/{len(chunks)}")

    # Normalize and count
    def normalize_category(cat):
        terms = sorted([t.strip().lower() for t in str(cat).split("/")])
        return "/".join(terms)

    flat_list = [str(x).strip() for x in all_items if str(x).strip()]
    if not flat_list:
        raise ValueError("No categories were extracted from the images.")

    df = pd.DataFrame(flat_list, columns=["Category"])
    df["normalized"] = df["Category"].map(normalize_category)

    result = (
        df.groupby("normalized")
          .agg(Category=("Category", lambda x: x.value_counts().index[0]),
               counts=("Category", "size"))
          .sort_values("counts", ascending=False)
          .reset_index(drop=True)
    )

    # Second-pass semantic merge
    seed_list = result["Category"].head(max_categories * 3).tolist()

    second_prompt = f"""
You are a data analyst reviewing categorized image data.

Task: From the provided categories, identify and return the top {max_categories} CONCEPTUALLY UNIQUE categories.

Critical Instructions:
1) Exact duplicates are already removed.
2) Merge SEMANTIC duplicates (same concept, different wording).
3) When merging:
   - Combine frequencies mentally
   - Keep the most frequent OR clearest label
   - Each concept appears ONLY ONCE
4) Keep category names {specificity}.
5) Return ONLY a numbered list of {max_categories} categories. No extra text.

Pre-processed Categories (sorted by frequency, top sample):
{seed_list}

Output:
1. category
2. category
...
{max_categories}. category
""".strip()

    try:
        if model_source in ["openai", "huggingface", "huggingface-together", "xai"]:
            endpoint = f"{openai_base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {"model": user_model, "messages": [{"role": "user", "content": second_prompt}]}
            if creativity is not None:
                payload["temperature"] = creativity
            response = req.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            top_categories_text = result["choices"][0]["message"]["content"]
        elif model_source == "anthropic":
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": user_model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": second_prompt}],
            }
            if creativity is not None:
                payload["temperature"] = creativity
            response = req.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            resp_content = result.get("content", [])
            if resp_content and resp_content[0].get("type") == "text":
                top_categories_text = resp_content[0].get("text", "")
            else:
                top_categories_text = ""
        elif model_source == "google":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
            headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": second_prompt}]}],
                "generationConfig": {**({"temperature": creativity} if creativity is not None else {})}
            }
            response = req.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            res = response.json()
            top_categories_text = res["candidates"][0]["content"]["parts"][0]["text"]
        elif model_source == "mistral":
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {"model": user_model, "messages": [{"role": "user", "content": second_prompt}]}
            if creativity is not None:
                payload["temperature"] = creativity
            response = req.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            top_categories_text = result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in second-pass merge: {e}")
        top_categories_text = ""

    # Final progress callback for the merge step
    if progress_callback:
        progress_callback(total_steps, total_steps, "Merging categories")

    # Parse final list
    final = []
    for line in top_categories_text.splitlines():
        m = line_pat.match(line.strip())
        if m:
            final.append(m.group(1).strip())
    if not final:
        final = [l.strip("-*• ").strip() for l in top_categories_text.splitlines() if l.strip()]

    print("\nTop categories:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(final[:max_categories])))

    if filename:
        result.to_csv(filename, index=False)

    return {
        "counts_df": result,
        "top_categories": final[:max_categories],
        "raw_top_text": top_categories_text
    }
