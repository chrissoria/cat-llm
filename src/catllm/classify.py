"""
Classification functions for CatLLM.

This module provides unified classification for text, image, and PDF inputs,
supporting both single-model and multi-model (ensemble) classification.
"""

import warnings
from typing import Union, Callable

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
    api_key=None,
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
    # Multi-model parameters
    models=None,
    consensus_threshold: Union[str, float] = "majority",
    # Parameters previously only on classify_ensemble
    survey_question: str = "",
    use_json_schema: bool = True,
    max_workers: int = None,
    fail_strategy: str = "partial",
    max_retries: int = 5,
    batch_retries: int = 2,
    retry_delay: float = 1.0,
    row_delay: float = 0.0,
    pdf_dpi: int = 150,
    auto_download: bool = False,
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
        thinking_budget (int): Controls reasoning behavior per provider:
            Google: token budget for extended thinking (0=off, >0=on).
            OpenAI: maps to reasoning_effort (0="minimal", >0="high").
            Anthropic: enables extended thinking (0=off, >0=on, min 1024).
        example1-6 (str): Example categorizations for few-shot learning.
        filename (str): Output filename for CSV.
        save_directory (str): Directory to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
            "mistral", "perplexity", "huggingface", "xai".
        progress_callback: Optional callback for progress updates.
        models (list): For multi-model mode, list of (model, provider, api_key) tuples.
            If provided, overrides user_model/api_key/model_source.
        consensus_threshold (str or float): For multi-model mode, agreement threshold.
            - "majority": 50% agreement (default)
            - "two-thirds": 67% agreement
            - "unanimous": 100% agreement
            - float: Custom threshold between 0 and 1
        survey_question (str): The survey question (used when categories="auto").
        use_json_schema (bool): Use JSON schema for structured output. Default True.
        max_workers (int): Max parallel workers for API calls. None = auto.
        fail_strategy (str): How to handle failures - "partial" (default) or "strict".
        max_retries (int): Max retries per API call. Default 5.
        batch_retries (int): Max retries for batch-level failures. Default 2.
        retry_delay (float): Delay between retries in seconds. Default 1.0.
        row_delay (float): Delay in seconds between processing each row. Useful
            when multiple models share the same API provider/key to avoid rate
            limits. Default 0.0 (no delay).
        pdf_dpi (int): DPI for PDF page rendering. Default 150.
        auto_download (bool): Auto-download Ollama models. Default False.

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
        survey_question=survey_question,
        pdf_mode=pdf_mode,
        pdf_dpi=pdf_dpi,
        creativity=creativity,
        safety=safety,
        chain_of_thought=chain_of_thought,
        chain_of_verification=chain_of_verification,
        step_back_prompt=step_back_prompt,
        context_prompt=context_prompt,
        thinking_budget=thinking_budget,
        use_json_schema=use_json_schema,
        max_workers=max_workers,
        fail_strategy=fail_strategy,
        max_retries=max_retries,
        batch_retries=batch_retries,
        retry_delay=retry_delay,
        row_delay=row_delay,
        auto_download=auto_download,
        example1=example1,
        example2=example2,
        example3=example3,
        example4=example4,
        example5=example5,
        example6=example6,
        consensus_threshold=consensus_threshold,
        max_categories=max_categories,
        categories_per_chunk=categories_per_chunk,
        divisions=divisions,
        research_question=research_question,
        filename=filename,
        save_directory=save_directory,
        progress_callback=progress_callback,
    )
