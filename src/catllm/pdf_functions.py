from .text_functions import _detect_model_source
from .calls.pdf_stepback import get_pdf_stepback_insight
from .calls.pdf_CoVe import (
    pdf_chain_of_verification_openai,
    pdf_chain_of_verification_anthropic,
    pdf_chain_of_verification_google,
    pdf_chain_of_verification_mistral
)


def _load_pdf_files(pdf_input):
    """Load PDF files from directory path or return list as-is."""
    import os
    import glob

    if not isinstance(pdf_input, list):
        pdf_files = glob.glob(os.path.join(pdf_input, '*.pdf'))
        pdf_files.extend(glob.glob(os.path.join(pdf_input, '*.PDF')))
        # Remove duplicates (case-insensitive systems)
        seen = set()
        unique_files = []
        for f in pdf_files:
            if f.lower() not in seen:
                seen.add(f.lower())
                unique_files.append(f)
        pdf_files = unique_files
        print(f"Found {len(pdf_files)} PDFs.")
    else:
        pdf_files = pdf_input
        print(f"Provided a list of {len(pdf_input)} PDFs.")

    return pdf_files


def _get_pdf_pages(pdf_path):
    """
    Extract all pages from a PDF as separate page objects.
    Returns list of tuples: [(page_index, page_label), ...]

    For 'document.pdf' with 3 pages:
    [(0, "document_p1"), (1, "document_p2"), (2, "document_p3")]

    The actual page data is extracted later based on provider needs.
    """
    import os
    from pathlib import Path

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF processing. "
            "Install it with: pip install PyMuPDF"
        )

    pdf_name = Path(pdf_path).stem  # filename without extension

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()

        if page_count == 0:
            print(f"Warning: {pdf_path} has no pages")
            return []

        pages = []
        for i in range(page_count):
            page_label = f"{pdf_name}_p{i+1}"
            pages.append((pdf_path, i, page_label))

        return pages

    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []


def _extract_page_as_pdf_bytes(pdf_path, page_index):
    """
    Extract a single page from a PDF as PDF bytes.
    Used for providers with native PDF support (Anthropic, Google).
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]

        # Create a new PDF with just this page
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)

        pdf_bytes = new_doc.tobytes()
        new_doc.close()
        doc.close()

        return pdf_bytes, True

    except Exception as e:
        print(f"Error extracting page {page_index} from {pdf_path}: {e}")
        return None, False


def _extract_page_as_image_bytes(pdf_path, page_index, dpi=150):
    """
    Extract a single page from a PDF as PNG image bytes.
    Used for providers without native PDF support (OpenAI, Mistral, etc.).
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]

        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default PDF DPI
        pix = page.get_pixmap(matrix=mat)

        # Get PNG bytes
        image_bytes = pix.tobytes("png")
        doc.close()

        return image_bytes, True

    except Exception as e:
        print(f"Error rendering page {page_index} from {pdf_path}: {e}")
        return None, False


def _encode_bytes_to_base64(data_bytes):
    """Encode bytes to base64 string."""
    import base64
    return base64.b64encode(data_bytes).decode("utf-8")


def _extract_page_text(pdf_path, page_index):
    """
    Extract text content from a single PDF page.
    Used for text-based processing mode.
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        text = page.get_text("text")
        doc.close()

        if not text.strip():
            return None, False, "Page contains no extractable text"

        return text.strip(), True, None

    except Exception as e:
        print(f"Error extracting text from page {page_index} of {pdf_path}: {e}")
        return None, False, str(e)


# PDF multi-class (binary) function
def pdf_multi_class(
    pdf_description,
    pdf_input,
    categories,
    api_key,
    user_model="gpt-4o",
    mode="image",
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
    Categorize PDF pages using LLMs with multi-label classification.

    Each page of each PDF is processed separately, with output labeled as
    {pdf_name}_p{page_number} (e.g., "report_p1", "report_p2").

    Args:
        pdf_description (str): Description of the PDF documents being categorized.
        pdf_input (str or list): Directory path containing PDFs, or list of PDF file paths.
        categories (list): List of category names for classification.
        api_key (str): API key for the model provider.
        user_model (str): Model name to use. Default "gpt-4o".
        mode (str): How to process PDF pages. Options:
            - "image": Render pages as images (best for visual elements like charts/tables)
            - "text": Extract text only (best for text-heavy documents, faster/cheaper)
            - "both": Send both text and image (most comprehensive but slower/costlier)
            Default is "image".
        creativity (float): Temperature setting. None uses model default.
        safety (bool): If True, saves progress after each page.
        chain_of_verification (bool): Enable Chain of Verification for accuracy.
        chain_of_thought (bool): Enable step-by-step reasoning. Default True.
        step_back_prompt (bool): Enable step-back prompting for abstract thinking.
        context_prompt (bool): Add expert context to prompts.
        thinking_budget (int): Token budget for thinking (Google models).
        example1-6 (str): Example categorizations for few-shot learning.
        filename (str): Output filename for CSV.
        save_directory (str): Directory to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
                          "mistral", "perplexity", "huggingface", "xai".

    Returns:
        pd.DataFrame: Results with columns:
            - pdf_input: Page label (e.g., "report_p1")
            - model_response: Raw model response
            - json: Extracted JSON
            - category_1, category_2, ...: Binary category assignments
            - processing_status: "success" or "error"

    Example:
        >>> import catllm as cat
        >>> # Image mode (default) - good for documents with charts/tables
        >>> results = cat.pdf_multi_class(
        ...     pdf_description="financial reports",
        ...     pdf_input="/path/to/pdfs/",
        ...     categories=["has_chart", "has_table", "is_summary"],
        ...     api_key="your-api-key",
        ...     mode="image"
        ... )
        >>> # Text mode - good for text-heavy documents, faster and cheaper
        >>> results = cat.pdf_multi_class(
        ...     pdf_description="research papers",
        ...     pdf_input="/path/to/pdfs/",
        ...     categories=["discusses_methodology", "has_results"],
        ...     api_key="your-api-key",
        ...     mode="text"
        ... )
    """
    import os
    import json
    import pandas as pd
    import regex
    import time
    from tqdm import tqdm

    if save_directory is not None and not os.path.isdir(save_directory):
        raise FileNotFoundError(f"Directory {save_directory} doesn't exist")

    # Validate mode parameter
    mode = mode.lower()
    if mode not in {"image", "text", "both"}:
        raise ValueError(f"mode must be 'image', 'text', or 'both', got: {mode}")

    model_source = _detect_model_source(user_model, model_source)

    # Providers with native PDF support (only used in image/both modes)
    native_pdf_providers = {"anthropic", "google"}

    print(f"Processing mode: {mode}")

    # Load PDF files
    pdf_files = _load_pdf_files(pdf_input)

    # Extract all pages from all PDFs
    all_pages = []  # List of (pdf_path, page_index, page_label)
    for pdf_path in pdf_files:
        pages = _get_pdf_pages(pdf_path)
        all_pages.extend(pages)

    print(f"Total pages to process: {len(all_pages)}")

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
        stepback = f"""What are the key content patterns or elements that typically indicate the presence of these categories in document pages showing "{pdf_description}"?

Categories to consider:
{categories_str}

Provide a brief analysis of what content cues to look for when categorizing such document pages."""

        stepback_insight, step_back_added = get_pdf_stepback_insight(
            model_source, stepback, api_key, user_model, creativity
        )
    else:
        stepback_insight = None
        step_back_added = False

    page_labels = []
    link1 = []
    extracted_jsons = []

    def _build_base_prompt_text(page_text=None):
        """Build the base text portion of the prompt based on mode."""
        # Determine instruction based on mode
        if mode == "text":
            examine_instruction = "Examine the following text extracted from a PDF page"
        elif mode == "both":
            examine_instruction = "Examine the attached PDF page image AND the extracted text below"
        else:  # image mode
            examine_instruction = "Examine the attached PDF page"

        if chain_of_thought:
            base_text = (
                f"You are a document-tagging assistant.\n"
                f"Task ► {examine_instruction} and decide, **for each category below**, "
                f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                f"Document page is expected to contain: {pdf_description}\n\n"
                f"Categories:\n{categories_str}\n\n"
                f"Let's analyze step by step:\n"
                f"1. First, identify the key content elements in the document page\n"
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
                f"You are a document-tagging assistant.\n"
                f"Task ► {examine_instruction} and decide, **for each category below**, "
                f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                f"Document page is expected to contain: {pdf_description}\n\n"
                f"Categories:\n{categories_str}\n\n"
                f"{examples_text}\n\n"
                f"Output format ► Respond with **only** a JSON object whose keys are the "
                f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                f"No additional keys, comments, or text.\n\n"
                f"Example (three categories):\n"
                f"{example_JSON}"
            )

        # Add extracted text for text and both modes
        if page_text and mode in ("text", "both"):
            base_text += f"\n\n--- EXTRACTED TEXT FROM PAGE ---\n{page_text}\n--- END OF EXTRACTED TEXT ---"

        if context_prompt:
            context = (
                "You are an expert document analyst specializing in page categorization. "
                "Apply multi-label classification based on explicit and implicit content cues. "
                "When uncertain, prioritize precision over recall.\n\n"
            )
            base_text = context + base_text

        return base_text

    def _build_cove_prompts(base_prompt_text):
        """Build chain of verification prompts for PDF pages."""
        step2_prompt = f"""You provided this initial categorization:
<<INITIAL_REPLY>>

Original task: {base_prompt_text}

Generate a focused list of 3-5 verification questions to fact-check your categorization. Each question should:
- Be concise and specific (one sentence)
- Address a distinct content element or category assignment
- Be answerable by re-examining the document page

Focus on verifying:
- Whether each category assignment matches what's visible in the page
- Whether any content elements were missed or misinterpreted
- Whether there are any logical inconsistencies

Provide only the verification questions as a numbered list."""

        step3_prompt = f"""Re-examine the attached document page and answer the following verification question.

Document description: {pdf_description}

Verification question: <<QUESTION>>

Provide a brief, direct answer (1-2 sentences maximum) based on what you observe in the page.

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

    def _build_prompt_openai_mistral(encoded_image, base_text):
        """Build prompt for OpenAI/Mistral format (PDF converted to image)."""
        encoded_image_url = f"data:image/png;base64,{encoded_image}"
        return [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": encoded_image_url, "detail": "high"}},
        ]

    def _build_prompt_anthropic_pdf(encoded_pdf, base_text):
        """Build prompt for Anthropic format with native PDF support."""
        return [
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

    def _build_prompt_google_pdf(encoded_pdf, base_text):
        """Build prompt data for Google format with native PDF support."""
        return {
            "text_prompt": base_text,
            "pdf_data": encoded_pdf,
            "mime_type": "application/pdf"
        }

    def _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content):
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
                    reply = pdf_chain_of_verification_openai(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        pdf_content=pdf_content
                    )

                return reply, None

            except BadRequestError as e:
                if "json_validate_failed" in str(e) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ JSON validation failed. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found.") from e

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

    def _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content):
        """Handle Anthropic API calls with native PDF support."""
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
                reply = pdf_chain_of_verification_anthropic(
                    initial_reply=reply,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    client=client,
                    user_model=user_model,
                    creativity=creativity,
                    remove_numbering=remove_numbering,
                    pdf_content=pdf_content
                )

            return reply, None

        except anthropic.NotFoundError as e:
            raise ValueError(f"❌ Model '{user_model}' on {model_source} not found.") from e
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_google(prompt_data, step2_prompt, step3_prompt, step4_prompt, base_prompt_text):
        """Handle Google API calls with native PDF support."""
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
                "data": prompt_data["pdf_data"]
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
                reply = pdf_chain_of_verification_google(
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
                    pdf_data=prompt_data["pdf_data"],
                    mime_type=prompt_data["mime_type"]
                )

            return reply, None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"❌ Model '{user_model}' not found.") from e
            elif e.response.status_code in [401, 403]:
                raise ValueError(f"❌ Authentication failed.") from e
            else:
                print(f"HTTP error occurred: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content):
        """Handle Mistral API calls (PDF converted to image)."""
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
                    reply = pdf_chain_of_verification_mistral(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering,
                        pdf_content=pdf_content
                    )

                return reply, None

            except SDKError as e:
                error_str = str(e).lower()

                if "invalid_model" in error_str or "invalid model" in error_str:
                    raise ValueError(f"❌ Model '{user_model}' not found.") from e
                elif "401" in str(e) or "unauthorized" in error_str:
                    raise ValueError(f"❌ Authentication failed.") from e

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

    def _build_prompt_text_only(base_text):
        """Build text-only prompt for providers (no image attachment)."""
        return [{"type": "text", "text": base_text}]

    def _call_openai_text_only(prompt_text, step2_prompt, step3_prompt, step4_prompt):
        """Handle OpenAI-compatible API calls with text-only prompt."""
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
                messages = []
                if step_back_prompt and step_back_added:
                    messages.append({'role': 'user', 'content': stepback})
                    messages.append({'role': 'assistant', 'content': stepback_insight})
                messages.append({'role': 'user', 'content': prompt_text})

                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=messages,
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content
                return reply, None

            except BadRequestError as e:
                if "json_validate_failed" in str(e) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ JSON validation failed. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found.") from e

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

    def _call_anthropic_text_only(prompt_text, step2_prompt, step3_prompt, step4_prompt):
        """Handle Anthropic API calls with text-only prompt."""
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        try:
            messages = []
            if step_back_prompt and step_back_added:
                messages.append({'role': 'user', 'content': stepback})
                messages.append({'role': 'assistant', 'content': stepback_insight})
            messages.append({'role': 'user', 'content': prompt_text})

            response_obj = client.messages.create(
                model=user_model,
                max_tokens=1024,
                messages=messages,
                **({"temperature": creativity} if creativity is not None else {})
            )
            reply = response_obj.content[0].text
            return reply, None

        except anthropic.NotFoundError as e:
            raise ValueError(f"❌ Model '{user_model}' on {model_source} not found.") from e
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_google_text_only(prompt_text, step2_prompt, step3_prompt, step4_prompt):
        """Handle Google API calls with text-only prompt."""
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

        parts = []
        if step_back_prompt and step_back_added:
            parts.append({"text": f"Context from step-back analysis:\n{stepback_insight}\n\n"})
        parts.append({"text": prompt_text})

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

            return reply, None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"❌ Model '{user_model}' not found.") from e
            elif e.response.status_code in [401, 403]:
                raise ValueError(f"❌ Authentication failed.") from e
            else:
                print(f"HTTP error occurred: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_mistral_text_only(prompt_text, step2_prompt, step3_prompt, step4_prompt):
        """Handle Mistral API calls with text-only prompt."""
        from mistralai import Mistral
        from mistralai.models import SDKError, MistralError

        client = Mistral(api_key=api_key)
        max_retries = 8
        delay = 2

        for attempt in range(max_retries):
            try:
                messages = []
                if step_back_prompt and step_back_added:
                    messages.append({'role': 'user', 'content': stepback})
                    messages.append({'role': 'assistant', 'content': stepback_insight})
                messages.append({'role': 'user', 'content': prompt_text})

                response = client.chat.complete(
                    model=user_model,
                    messages=messages,
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response.choices[0].message.content
                return reply, None

            except SDKError as e:
                error_str = str(e).lower()

                if "invalid_model" in error_str or "invalid model" in error_str:
                    raise ValueError(f"❌ Model '{user_model}' not found.") from e
                elif "401" in str(e) or "unauthorized" in error_str:
                    raise ValueError(f"❌ Authentication failed.") from e

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

    def _process_single_page(pdf_path, page_index, page_label):
        """Process a single PDF page and return (reply, error_msg)."""

        # Extract text if needed for text or both modes
        page_text = None
        if mode in ("text", "both"):
            page_text, text_valid, text_error = _extract_page_text(pdf_path, page_index)
            if mode == "text" and not text_valid:
                # Text mode requires text - fail if extraction failed
                return None, f"Failed to extract text: {text_error}"
            # For "both" mode, we continue even if text extraction fails

        # Build prompt with text if available
        base_prompt_text = _build_base_prompt_text(page_text)

        if chain_of_verification:
            step2_prompt, step3_prompt, step4_prompt = _build_cove_prompts(base_prompt_text)
        else:
            step2_prompt = step3_prompt = step4_prompt = None

        # TEXT-ONLY MODE: No image/PDF attachment needed
        if mode == "text":
            if model_source == "anthropic":
                return _call_anthropic_text_only(base_prompt_text, step2_prompt, step3_prompt, step4_prompt)
            elif model_source == "google":
                return _call_google_text_only(base_prompt_text, step2_prompt, step3_prompt, step4_prompt)
            elif model_source in ["openai", "perplexity", "huggingface", "xai"]:
                return _call_openai_text_only(base_prompt_text, step2_prompt, step3_prompt, step4_prompt)
            elif model_source == "mistral":
                return _call_mistral_text_only(base_prompt_text, step2_prompt, step3_prompt, step4_prompt)
            else:
                raise ValueError(f"Unknown source! Choose from OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral")

        # IMAGE or BOTH MODE: Include image/PDF attachment
        # Handle providers with native PDF support
        if model_source == "anthropic":
            pdf_bytes, is_valid = _extract_page_as_pdf_bytes(pdf_path, page_index)
            if not is_valid:
                return None, "Failed to extract PDF page"

            encoded_pdf = _encode_bytes_to_base64(pdf_bytes)
            prompt = _build_prompt_anthropic_pdf(encoded_pdf, base_prompt_text)
            pdf_content = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": encoded_pdf
                }
            }
            return _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content)

        elif model_source == "google":
            pdf_bytes, is_valid = _extract_page_as_pdf_bytes(pdf_path, page_index)
            if not is_valid:
                return None, "Failed to extract PDF page"

            encoded_pdf = _encode_bytes_to_base64(pdf_bytes)
            prompt_data = _build_prompt_google_pdf(encoded_pdf, base_prompt_text)
            return _call_google(prompt_data, step2_prompt, step3_prompt, step4_prompt, base_prompt_text)

        # Handle providers requiring image conversion
        else:
            image_bytes, is_valid = _extract_page_as_image_bytes(pdf_path, page_index)
            if not is_valid:
                return None, "Failed to render PDF page to image"

            encoded_image = _encode_bytes_to_base64(image_bytes)
            prompt = _build_prompt_openai_mistral(encoded_image, base_prompt_text)

            # PDF content for CoVe (as image)
            encoded_image_url = f"data:image/png;base64,{encoded_image}"
            pdf_content = {"type": "image_url", "image_url": {"url": encoded_image_url, "detail": "high"}}

            if model_source in ["openai", "perplexity", "huggingface", "xai"]:
                return _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content)
            elif model_source == "mistral":
                return _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt, pdf_content)
            else:
                raise ValueError(f"Unknown source! Choose from OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral")

    def _extract_json(reply):
        """Extract JSON from model reply."""
        if reply is None:
            return """{"1":"e"}"""

        extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
        if extracted_json:
            return extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
        else:
            print("""{"1":"e"}""")
            return """{"1":"e"}"""

    # Main processing loop
    for idx, (pdf_path, page_index, page_label) in enumerate(tqdm(all_pages, desc="Categorizing PDF pages")):
        page_labels.append(page_label)

        reply, error_msg = _process_single_page(pdf_path, page_index, page_label)

        if error_msg:
            link1.append(error_msg)
            extracted_jsons.append("""{"1":"e"}""")
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
                'pdf_input': page_labels,
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
        'pdf_input': pd.Series(page_labels),
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
