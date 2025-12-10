from .text_functions import _detect_model_source
from .calls.image_stepback import get_image_stepback_insight
from .calls.image_CoVe import (
    image_chain_of_verification_openai,
    image_chain_of_verification_anthropic,
    image_chain_of_verification_google,
    image_chain_of_verification_mistral
)


def _load_image_files(image_input):
    """Load image files from directory path or return list as-is."""
    import os
    import glob

    image_extensions = [
        '*.png', '*.jpg', '*.jpeg',
        '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
        '*.tif', '*.tiff', '*.bmp',
        '*.heif', '*.heic', '*.ico',
        '*.psd'
    ]

    if not isinstance(image_input, list):
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))
        print(f"Found {len(image_files)} images.")
    else:
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")

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
        from openai import OpenAI, BadRequestError

        base_url = (
            "https://api.perplexity.ai" if model_source == "perplexity"
            else "https://router.huggingface.co/v1" if model_source == "huggingface"
            else "https://api.x.ai/v1" if model_source == "xai"
            else None
        )
        client = OpenAI(api_key=api_key, base_url=base_url)

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

                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=messages,
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content

                if chain_of_verification:
                    reply = image_chain_of_verification_openai(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        image_content=image_content
                    )

                return reply, None

            except BadRequestError as e:
                if "json_validate_failed" in str(e) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ JSON validation failed. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e

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
        """Handle Anthropic API calls."""
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        try:
            # Build messages with optional stepback
            messages = []
            if step_back_prompt and step_back_added:
                messages.append({'role': 'user', 'content': stepback})
                messages.append({'role': 'assistant', 'content': stepback_insight})
            messages.append({'role': 'user', 'content': prompt})

            response_obj = client.messages.create(
                model=user_model,
                max_tokens=1024,
                messages=messages,
                **({"temperature": creativity} if creativity is not None else {})
            )
            reply = response_obj.content[0].text

            if chain_of_verification:
                reply = image_chain_of_verification_anthropic(
                    initial_reply=reply,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    client=client,
                    user_model=user_model,
                    creativity=creativity,
                    remove_numbering=remove_numbering,
                    image_content=image_content
                )

            return reply, None

        except anthropic.NotFoundError as e:
            raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
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
        """Handle Mistral API calls."""
        from mistralai import Mistral
        from mistralai.models import SDKError, MistralError

        client = Mistral(api_key=api_key)
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

                response = client.chat.complete(
                    model=user_model,
                    messages=messages,
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response.choices[0].message.content

                if chain_of_verification:
                    reply = image_chain_of_verification_mistral(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        image_content=image_content
                    )

                return reply, None

            except SDKError as e:
                error_str = str(e).lower()

                if "invalid_model" in error_str or "invalid model" in error_str:
                    raise ValueError(f"❌ Model '{user_model}' not found.") from e
                elif "401" in str(e) or "unauthorized" in error_str:
                    raise ValueError(f"❌ Authentication failed. Please check your Mistral API key.") from e

                retryable_errors = ["500", "502", "503", "504"]
                if any(code in str(e) for code in retryable_errors) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ Server error detected. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

            except MistralError as e:
                if hasattr(e, 'status_code') and e.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ Server error {e.status_code}. Attempt {attempt + 1}/{max_retries}")
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
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = message.content[0].text  # Anthropic returns content as list
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "mistral":
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            try:
                response = client.chat.complete(
                    model=user_model,
                    messages=[
                    {'role': 'user', 'content': prompt}
                ],
                **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")
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
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "perplexity":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = message.content[0].text  # Anthropic returns content as list
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "mistral":
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            try:
                response = client.chat.complete(
                    model=user_model,
                    messages=[
                    {'role': 'user', 'content': prompt}
                ],
                **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif  valid_image == False:
            print("Skipped NaN input or invalid path")
            reply = None
            link1.append("Error processing input: {e}")
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
