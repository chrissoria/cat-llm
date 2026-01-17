"""
Classification functions for CatLLM.

This module provides unified classification for text, image, and PDF inputs,
supporting both single-model and multi-model (ensemble) classification.
"""

import warnings

__all__ = [
    # Main entry point
    "classify",
    # Ensemble function
    "classify_ensemble",
    # Deprecated functions (kept for backward compatibility)
    "multi_class",
    "image_multi_class",
    "pdf_multi_class",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions_ensemble import (
    classify_ensemble,
)

# Import deprecated functions for backward compatibility
from .text_functions import multi_class
from .image_functions import image_multi_class
from .pdf_functions import pdf_multi_class


def classify(
    input_data,
    categories,
    api_key,
    input_type="text",
    description="",
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
    model_source="auto",
    max_categories=12,
    categories_per_chunk=10,
    divisions=10,
    research_question=None,
    progress_callback=None,
    # New multi-model parameters
    models=None,
    consensus_threshold="majority",
):
    """
    Unified classification function for text, image, and PDF inputs.

    Supports single-model and multi-model (ensemble) classification. Input type
    is auto-detected from the data (text strings, image paths, or PDF paths).

    Args:
        input_data: The data to classify. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path or list of image file paths
            - For pdf: directory path or list of PDF file paths
        categories (list): List of category names for classification.
        api_key (str): API key for the model provider (single-model mode).
        input_type (str): DEPRECATED - input type is now auto-detected.
            Kept for backward compatibility.
        description (str): Description of the input data context.
        user_model (str): Model name to use. Default "gpt-4o".
        mode (str): PDF processing mode:
            - "image" (default): Render pages as images
            - "text": Extract text only
            - "both": Send both image and extracted text
        creativity (float): Temperature setting. None uses model default.
        safety (bool): If True, saves progress after each item.
        chain_of_verification (bool): Enable Chain of Verification for accuracy.
        chain_of_thought (bool): Enable step-by-step reasoning. Default True.
        step_back_prompt (bool): Enable step-back prompting.
        context_prompt (bool): Add expert context to prompts.
        thinking_budget (int): Token budget for thinking (Google models).
        example1-6 (str): Example categorizations for few-shot learning.
        filename (str): Output filename for CSV.
        save_directory (str): Directory to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
            "mistral", "perplexity", "huggingface", "xai".
        progress_callback: Optional callback for progress updates.
        models (list): For multi-model mode, list of (model, provider, api_key) tuples.
            If provided, overrides user_model/api_key/model_source.
        consensus_threshold: For multi-model mode, agreement threshold. Can be:
            - "majority": 50% agreement (default)
            - "two-thirds": 67% agreement
            - "unanimous": 100% agreement
            - float: Custom threshold between 0 and 1

    Returns:
        pd.DataFrame: Results with classification columns.

    Examples:
        >>> import catllm as cat
        >>>
        >>> # Single model classification
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     description="Customer feedback survey",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Multi-model ensemble
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative"],
        ...     models=[
        ...         ("gpt-4o", "openai", "sk-..."),
        ...         ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ...     ],
        ...     consensus_threshold="majority",  # or "two-thirds", "unanimous", or 0.75
        ... )
    """
    # Build models list
    if models is None:
        # Single model mode - build models list from individual params
        models = [(user_model, model_source, api_key)]

    # Map mode to pdf_mode
    pdf_mode = mode if mode in ("image", "text", "both") else "image"

    return classify_ensemble(
        survey_input=input_data,
        categories=categories,
        models=models,
        input_description=description,
        pdf_mode=pdf_mode,
        chain_of_thought=chain_of_thought,
        step_back_prompt=step_back_prompt,
        context_prompt=context_prompt,
        consensus_threshold=consensus_threshold,
        filename=filename,
        save_directory=save_directory,
        progress_callback=progress_callback,
    )
