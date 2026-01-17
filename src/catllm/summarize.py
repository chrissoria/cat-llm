"""
Summarization functions for CatLLM.

This module provides unified summarization for text and PDF inputs,
supporting both single-model and multi-model (ensemble) summarization.
"""

import warnings

__all__ = [
    # Main entry point
    "summarize",
    # Ensemble function
    "summarize_ensemble",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions_ensemble import (
    summarize_ensemble,
)


def summarize(
    input_data,
    api_key: str = None,
    description: str = "",
    instructions: str = "",
    max_length: int = None,
    focus: str = None,
    user_model: str = "gpt-4o",
    model_source: str = "auto",
    mode: str = "image",
    creativity: float = None,
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    filename: str = None,
    save_directory: str = None,
    progress_callback=None,
    models: list = None,
):
    """
    Summarize text or PDF data using LLMs.

    Supports single-model and multi-model (ensemble) summarization. In multi-model
    mode, summaries from all models are synthesized into a consensus summary.
    Input type is auto-detected from the data (text strings or PDF paths).

    Args:
        input_data: Data to summarize. Can be:
            - Text: list of strings, pandas Series, or single string
            - PDF: directory path, single PDF path, or list of PDF paths
        api_key (str): API key for the model provider (single-model mode)
        description (str): Description of what the content contains (provides context)
        instructions (str): Specific summarization instructions (e.g., "bullet points")
        max_length (int): Maximum summary length in words
        focus (str): What to focus on (e.g., "main arguments", "emotional content")
        user_model (str): Model to use (default "gpt-4o")
        model_source (str): Provider - "auto", "openai", "anthropic", "google", etc.
        mode (str): PDF processing mode (only used for PDF input):
            - "image" (default): Render pages as images
            - "text": Extract text only
            - "both": Send both image and extracted text
        creativity (float): Temperature setting (None uses provider default)
        chain_of_thought (bool): Enable step-by-step reasoning (default True)
        context_prompt (bool): Add expert context prefix
        step_back_prompt (bool): Enable step-back prompting
        filename (str): Output CSV filename
        save_directory (str): Directory to save results
        progress_callback: Optional callback for progress updates
        models (list): For multi-model mode, list of (model, provider, api_key) tuples

    Returns:
        pd.DataFrame: Results with summary column(s):
            - survey_input: Original text or page label (for PDFs)
            - summary: Generated summary (or consensus for multi-model)
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
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # PDF summarization (auto-detected)
        >>> results = cat.summarize(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     mode="image",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # PDF summarization with list of files
        >>> results = cat.summarize(
        ...     input_data=["doc1.pdf", "doc2.pdf"],
        ...     description="Financial reports",
        ...     mode="both",
        ...     focus="key metrics",
        ...     api_key="your-api-key"
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
    # Map mode to pdf_mode
    pdf_mode = mode if mode in ("image", "text", "both") else "image"

    return summarize_ensemble(
        survey_input=input_data,
        api_key=api_key,
        input_description=description,
        summary_instructions=instructions,
        max_length=max_length,
        focus=focus,
        user_model=user_model,
        model_source=model_source,
        pdf_mode=pdf_mode,
        creativity=creativity,
        chain_of_thought=chain_of_thought,
        context_prompt=context_prompt,
        step_back_prompt=step_back_prompt,
        filename=filename,
        save_directory=save_directory,
        progress_callback=progress_callback,
        models=models,
    )
