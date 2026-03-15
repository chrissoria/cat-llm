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
    # Robustness parameters
    safety: bool = False,
    max_retries: int = 5,
    batch_retries: int = 2,
    retry_delay: float = 1.0,
    row_delay: float = 0.0,
    fail_strategy: str = "partial",
    # Batch mode parameters
    batch_mode: bool = False,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
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
        safety (bool): If True, saves progress after each item. Requires filename.
        max_retries (int): Max retries per API call. Default 5.
        batch_retries (int): Max retries for batch-level failures. Default 2.
        retry_delay (float): Delay between retries in seconds. Default 1.0.
        row_delay (float): Delay in seconds between processing each row. Default 0.0.
        fail_strategy (str): How to handle failures - "partial" (default) or "strict".
        batch_mode (bool): If True, use async batch API (50% cost savings).
            Supported providers: openai, anthropic, google, mistral, xai.
            Not compatible with PDF input.
        batch_poll_interval (float): Seconds between batch status checks. Default 30.
        batch_timeout (float): Max seconds to wait for batch completion. Default 86400 (24h).

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
        >>> # With safety saves and row delay
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key="your-api-key",
        ...     safety=True,
        ...     filename="results.csv",
        ...     row_delay=1.0,
        ... )
        >>>
        >>> # Batch mode (50% cost savings)
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key="your-api-key",
        ...     batch_mode=True,
        ...     filename="batch_results.csv",
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

    # =========================================================================
    # Batch mode — bypass summarize_ensemble entirely
    # =========================================================================
    if batch_mode:
        from ._batch import UNSUPPORTED_BATCH_PROVIDERS, run_batch_summarize
        from .text_functions_ensemble import _detect_input_type, prepare_model_configs

        # Guard: text input only
        detected_type = _detect_input_type(input_data)
        if detected_type == "pdf":
            raise ValueError(
                "batch_mode=True only supports text input, but detected input type is 'pdf'. "
                "Set batch_mode=False for PDF summarization."
            )

        # Warn if progress_callback was provided (incompatible with batch)
        if progress_callback is not None:
            print(
                "[CatLLM] WARNING: progress_callback is not available in batch_mode "
                "(no per-item progress until the job completes). Ignoring callback."
            )

        # Build models list
        if models is None:
            batch_models = [(user_model, model_source, api_key)]
        else:
            batch_models = models

        model_configs = prepare_model_configs(batch_models)
        items = list(input_data) if not isinstance(input_data, list) else input_data

        prompt_params = {
            "input_description": description,
            "summary_instructions": instructions,
            "max_length": max_length,
            "focus": focus,
            "chain_of_thought": chain_of_thought,
            "context_prompt": context_prompt,
            "step_back_prompt": step_back_prompt,
            "stepback_insights": {},
            "creativity": creativity,
        }

        if len(batch_models) == 1:
            cfg = model_configs[0]
            if cfg["provider"] in UNSUPPORTED_BATCH_PROVIDERS:
                raise ValueError(
                    f"batch_mode=True is not supported for provider '{cfg['provider']}'. "
                    f"Supported providers: openai, anthropic, google, mistral, xai."
                )
            return run_batch_summarize(
                items=items,
                cfg=cfg,
                prompt_params=prompt_params,
                filename=filename,
                save_directory=save_directory,
                batch_poll_interval=batch_poll_interval,
                batch_timeout=batch_timeout,
                fail_strategy=fail_strategy,
            )

        # Ensemble batch path
        print(
            "[CatLLM] NOTE: batch_mode=True with multiple models is experimental. "
            "Each model submits a separate batch job concurrently."
        )
        from ._batch import run_batch_ensemble_summarize
        prompt_params_per_model = {
            cfg["model"]: {
                **prompt_params,
                "creativity": cfg["creativity"] if cfg["creativity"] is not None else creativity,
            }
            for cfg in model_configs
        }
        return run_batch_ensemble_summarize(
            items=items,
            model_configs=model_configs,
            prompt_params_per_model=prompt_params_per_model,
            fail_strategy=fail_strategy,
            filename=filename,
            save_directory=save_directory,
            batch_poll_interval=batch_poll_interval,
            batch_timeout=batch_timeout,
            max_retries=max_retries,
        )

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
        max_retries=max_retries,
        batch_retries=batch_retries,
        retry_delay=retry_delay,
        row_delay=row_delay,
        fail_strategy=fail_strategy,
        safety=safety,
        filename=filename,
        save_directory=save_directory,
        progress_callback=progress_callback,
        models=models,
    )
