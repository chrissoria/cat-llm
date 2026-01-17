"""
Shared utilities for CatLLM.

This module provides utility functions for JSON handling, file loading,
encoding, and other common operations used across the package.
"""

import json
import regex

__all__ = [
    # JSON utilities
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
    "ollama_two_step_classify",
    # Stepback utilities
    "_get_stepback_insight",
    # Image utilities
    "_load_image_files",
    "_encode_image",
    # PDF utilities
    "_anthropic_supports_pdf",
    "_load_pdf_files",
    "_get_pdf_pages",
    "_extract_page_as_pdf_bytes",
    "_extract_page_as_image_bytes",
    "_encode_bytes_to_base64",
    "_extract_page_text",
]


# =============================================================================
# JSON Schema Functions
# =============================================================================

def build_json_schema(categories: list, include_additional_properties: bool = True) -> dict:
    """Build a JSON schema for the classification output.

    Args:
        categories: List of category names
        include_additional_properties: If True, includes additionalProperties: false
                                       (required by OpenAI strict mode, not supported by Google)
    """
    properties = {}
    for i, cat in enumerate(categories, 1):
        properties[str(i)] = {
            "type": "string",
            "enum": ["0", "1"],
            "description": cat,
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }

    if include_additional_properties:
        schema["additionalProperties"] = False

    return schema


def extract_json(reply: str) -> str:
    """Extract JSON from model reply."""
    if reply is None:
        return '{"1":"e"}'

    extracted = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
    if extracted:
        # Clean up the JSON string
        return extracted[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '')
    else:
        return '{"1":"e"}'


def validate_classification_json(json_str: str, num_categories: int) -> tuple[bool, dict | None]:
    """
    Validate that a JSON string contains valid classification output.

    Args:
        json_str: The JSON string to validate
        num_categories: Expected number of categories

    Returns:
        tuple: (is_valid, parsed_dict or None)
    """
    try:
        parsed = json.loads(json_str)

        if not isinstance(parsed, dict):
            return False, None

        # Check that all expected keys are present and values are "0" or "1"
        for i in range(1, num_categories + 1):
            key = str(i)
            if key not in parsed:
                return False, None
            val = str(parsed[key]).strip()
            if val not in ("0", "1"):
                return False, None

        # Normalize values to strings
        normalized = {str(i): str(parsed[str(i)]).strip() for i in range(1, num_categories + 1)}
        return True, normalized

    except (json.JSONDecodeError, KeyError, TypeError):
        return False, None


def ollama_two_step_classify(
    client,
    response_text: str,
    categories: list,
    categories_str: str,
    survey_question: str = "",
    creativity: float = None,
    max_retries: int = 5,
) -> tuple[str, str | None]:
    """
    Two-step classification for Ollama models.

    Step 1: Classify the response (natural language output OK)
    Step 2: Convert classification to strict JSON format

    This approach is more reliable for local models that struggle with
    simultaneous reasoning and JSON formatting.

    Args:
        client: UnifiedLLMClient instance
        response_text: The survey response to classify
        categories: List of category names
        categories_str: Pre-formatted category string
        survey_question: Optional context
        creativity: Temperature setting
        max_retries: Number of retry attempts for JSON validation

    Returns:
        tuple: (json_string, error_message or None)
    """
    num_categories = len(categories)
    survey_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # ==========================================================================
    # Step 1: Classification (natural language - focus on accuracy)
    # ==========================================================================
    step1_messages = [
        {
            "role": "system",
            "content": "You are an expert at categorizing survey responses. Focus on accurate classification."
        },
        {
            "role": "user",
            "content": f"""{survey_context}

Analyze this survey response and determine which categories apply:

Response: "{response_text}"

Categories:
{categories_str}

For each category, explain briefly whether it applies (YES) or not (NO) to this response.
Format your answer as:
1. [Category name]: YES/NO - [brief reason]
2. [Category name]: YES/NO - [brief reason]
...and so on for all categories."""
        }
    ]

    step1_reply, step1_error = client.complete(
        messages=step1_messages,
        json_schema=None,  # No JSON requirement for step 1
        creativity=creativity,
    )

    if step1_error:
        return '{"1":"e"}', f"Step 1 failed: {step1_error}"

    # ==========================================================================
    # Step 2: JSON Formatting with validation and retry
    # ==========================================================================
    example_json = json.dumps({str(i): "0" for i in range(1, num_categories + 1)})

    for attempt in range(max_retries):
        step2_messages = [
            {
                "role": "system",
                "content": "You convert classification results to JSON. Output ONLY valid JSON, nothing else."
            },
            {
                "role": "user",
                "content": f"""Convert this classification to JSON format.

Classification results:
{step1_reply}

Rules:
- Output ONLY a JSON object, no other text
- Use category numbers as keys (1, 2, 3, etc.)
- Use "1" if the category was marked YES, "0" if NO
- Include ALL {num_categories} categories

Example format:
{example_json}

Your JSON output:"""
            }
        ]

        step2_reply, step2_error = client.complete(
            messages=step2_messages,
            json_schema=None,  # Ollama doesn't support strict schema anyway
            creativity=0.1,  # Low temperature for formatting task
        )

        if step2_error:
            if attempt < max_retries - 1:
                continue
            return '{"1":"e"}', f"Step 2 failed: {step2_error}"

        # Extract and validate JSON
        extracted = extract_json(step2_reply)
        is_valid, normalized = validate_classification_json(extracted, num_categories)

        if is_valid:
            return json.dumps(normalized), None

        # If invalid, try again with more explicit instructions
        if attempt < max_retries - 1:
            step1_reply = f"""Previous attempt produced invalid JSON.

Original classification:
{step1_reply}

Please be more careful to output EXACTLY {num_categories} categories numbered 1 through {num_categories}."""

    # All retries exhausted - try to salvage what we can
    extracted = extract_json(step2_reply) if step2_reply else '{"1":"e"}'
    return extracted, f"JSON validation failed after {max_retries} attempts"


# =============================================================================
# Stepback Insight Utility
# =============================================================================

def _get_stepback_insight(model_source, stepback, api_key, user_model, creativity):
    """Get step-back insight using the appropriate provider."""
    from .calls.stepback import (
        get_stepback_insight_openai,
        get_stepback_insight_anthropic,
        get_stepback_insight_google,
        get_stepback_insight_mistral
    )

    stepback_functions = {
        "openai": get_stepback_insight_openai,
        "perplexity": get_stepback_insight_openai,
        "huggingface": get_stepback_insight_openai,
        "huggingface-together": get_stepback_insight_openai,
        "xai": get_stepback_insight_openai,
        "anthropic": get_stepback_insight_anthropic,
        "google": get_stepback_insight_google,
        "mistral": get_stepback_insight_mistral,
    }

    func = stepback_functions.get(model_source)
    if func is None:
        return None, False

    return func(
        stepback=stepback,
        api_key=api_key,
        user_model=user_model,
        model_source=model_source,
        creativity=creativity
    )


# =============================================================================
# Image File Utilities
# =============================================================================

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


# =============================================================================
# PDF File Utilities
# =============================================================================

def _anthropic_supports_pdf(model_name):
    """Check if the Anthropic model supports native PDF input.

    PDF support is available for Claude 3.5 Sonnet, Claude 3 Opus, and Claude 3 Sonnet,
    but NOT for Claude 3 Haiku.
    """
    model_lower = model_name.lower()
    # Haiku models don't support PDF
    if "haiku" in model_lower:
        return False
    # Sonnet, Opus support PDF
    if any(x in model_lower for x in ["sonnet", "opus"]):
        return True
    # Default to False for unknown models to be safe
    return False


def _load_pdf_files(pdf_input):
    """Load PDF files from directory path, single file path, or return list as-is."""
    import os
    import glob

    if isinstance(pdf_input, list):
        pdf_files = pdf_input
        print(f"Provided a list of {len(pdf_input)} PDFs.")
    elif os.path.isfile(pdf_input):
        # Single file path
        pdf_files = [pdf_input]
        print(f"Provided 1 PDF file.")
    elif os.path.isdir(pdf_input):
        # Directory path - glob for PDFs
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
        print(f"Found {len(pdf_files)} PDFs in directory.")
    else:
        raise FileNotFoundError(f"PDF input not found: {pdf_input}")

    return pdf_files


def _get_pdf_pages(pdf_path):
    """
    Extract all pages from a PDF as separate page objects.
    Returns list of tuples: [(pdf_path, page_index, page_label), ...]

    For 'document.pdf' with 3 pages:
    [(pdf_path, 0, "document_p1"), (pdf_path, 1, "document_p2"), (pdf_path, 2, "document_p3")]

    The actual page data is extracted later based on provider needs.
    """
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
