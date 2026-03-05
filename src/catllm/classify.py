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

# Category analysis
from ._category_analysis import has_other_category, check_category_verbosity

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
    chain_of_thought=False,
    step_back_prompt=False,
    context_prompt=False,
    thinking_budget=0,
    # TODO: Auto-restructure examples — accept raw (response, labels) tuples or
    # dicts and automatically format them into the JSON string format the prompt
    # expects (e.g. "Response: '<text>' -> {"1": 0, "2": 1, ...}") so callers
    # don't need to know the internal prompt structure.
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
    # Batch mode parameters
    batch_mode: bool = False,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
    # Multi-model parameters
    models=None,
    consensus_threshold: Union[str, float] = "unanimous",
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
    add_other = "prompt",
    check_verbosity: bool = True,
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
        chain_of_thought (bool): Enable step-by-step reasoning. Default False.
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
        batch_mode (bool): If True, use async batch API (50% cost savings, higher rate limits).
            Supported providers: openai, anthropic, google, mistral, xai.
            Not supported: huggingface, perplexity, ollama.
            Ensemble mode: supported. Each model submits its own batch job concurrently.
            Providers without batch API (HuggingFace, Perplexity, Ollama) fall back to
            synchronous calls and are merged in with the batch results.
            Incompatible with: PDF/image input, progress_callback.
        batch_poll_interval (float): Seconds between batch job status checks. Default 30.
        batch_timeout (float): Max seconds to wait for batch completion. Default 86400 (24h).
        models (list): For multi-model mode, list of (model, provider, api_key) tuples.
            If provided, overrides user_model/api_key/model_source.
        consensus_threshold (str or float): For multi-model mode, agreement threshold.
            - "unanimous": 100% agreement (default — empirically produces
              the highest accuracy by aggressively eliminating false positives)
            - "majority": 50% agreement
            - "two-thirds": 67% agreement
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
        add_other (str or bool): Controls auto-addition of an "Other" catch-all
            category when none is detected. An "Other" category improves accuracy
            by preventing the model from forcing ambiguous responses into
            ill-fitting categories.
            - "prompt" (default): Ask the user to accept or reject the suggestion.
            - True: Silently add "Other" without prompting.
            - False: Never add "Other".
        check_verbosity (bool): Check whether each category has a description
            and examples (1 API call). Verbose categories with descriptions and
            examples significantly improve classification accuracy over bare
            labels. Default True. Set to False to skip.

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
        ...     consensus_threshold="unanimous",  # or "majority", "two-thirds", or 0.75
        ... )
    """
    # Build models list
    if models is None:
        # Single model mode - build models list from individual params
        models = [(user_model, model_source, api_key)]

    # Auto-append "Other" catch-all category if missing
    if add_other and categories and categories != "auto":
        if not has_other_category(categories):
            if add_other == "prompt":
                print(
                    "\n[CatLLM] It looks like your categories may not include a catch-all\n"
                    "  'Other' option. Adding one can improve accuracy by giving the\n"
                    "  model an outlet for ambiguous responses instead of forcing them\n"
                    "  into ill-fitting categories.\n"
                    "  (If you already have a catch-all under a different name, choose 'n'.)\n"
                )
                try:
                    answer = input("  Add 'Other' to your categories? (Y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("", "y", "yes"):
                    categories = list(categories) + ["Other"]
                    print(f"  -> Categories are now: {categories}\n")
                else:
                    print("  -> Keeping original categories.\n")
            else:
                # add_other=True — silently add
                categories = list(categories) + ["Other"]
                print(
                    f"[CatLLM] Auto-added 'Other' catch-all category. "
                    f"Categories are now: {categories}  "
                    f"(set add_other=False to disable)"
                )

    # Check category verbosity (1 API call)
    # TODO: Offer to auto-generate verbose versions of bare categories using
    # the LLM (description + examples) and let the user accept/edit them.
    if check_verbosity and categories and categories != "auto":
        # Extract API key and provider from first model entry
        first_entry = models[0]
        check_key = first_entry[2] if len(first_entry) >= 3 else None
        check_source = first_entry[1] if len(first_entry) >= 2 else "auto"

        if check_key:
            try:
                verbosity = check_category_verbosity(
                    categories,
                    api_key=check_key,
                    model_source=check_source,
                )
                lacking = [r for r in verbosity if not r["is_verbose"]]

                if lacking:
                    missing_desc = [r for r in lacking if not r["has_description"]]
                    missing_ex = [r for r in lacking if not r["has_examples"]]

                    print(
                        "\n[CatLLM] Category verbosity check (set check_verbosity=False to skip):"
                    )
                    for r in lacking:
                        issues = []
                        if not r["has_description"]:
                            issues.append("description")
                        if not r["has_examples"]:
                            issues.append("examples")
                        print(f'  - "{r["category"]}"  (missing: {", ".join(issues)})')

                    print(
                        "\n  Verbose categories with descriptions and examples significantly\n"
                        "  improve classification accuracy over bare labels.\n"
                        "\n"
                        "  Instead of:\n"
                        '    "Positive"\n'
                        "  Consider:\n"
                        '    "Positive: The response expresses satisfaction, approval, or\n'
                        "     happiness (e.g., 'I love this product', 'Great experience',\n"
                        "     'Very pleased with the result')\"\n"
                    )
            except Exception:
                pass  # Non-critical — don't block classification

    # =========================================================================
    # Evidence-based warnings for prompting strategies
    # Based on empirical findings from Soria et al. (2026) comparing prompting
    # strategies across 4 representative models and 4 survey tasks.
    # =========================================================================
    _strategy_warnings = []

    if chain_of_verification:
        _strategy_warnings.append(
            "[CatLLM] WARNING: chain_of_verification=True is enabled.\n"
            "  Empirical evidence shows CoVe DEGRADES accuracy by ~2 pp and\n"
            "  sensitivity by up to 12 pp for structured classification tasks.\n"
            "  The verification step causes models to retract correct classifications.\n"
            "  Cost: ~4x API calls per response.\n"
            "  This feature is provided for research purposes only — it is not\n"
            "  recommended for improving classification accuracy."
        )

    examples = [example1, example2, example3, example4, example5, example6]
    n_examples = sum(1 for ex in examples if ex is not None)
    if n_examples > 0:
        _strategy_warnings.append(
            f"[CatLLM] NOTE: {n_examples} few-shot example(s) provided.\n"
            "  Empirical evidence shows few-shot examples DEGRADE accuracy by\n"
            "  ~1.1-1.2 pp on average. Examples encourage over-classification\n"
            "  (sensitivity up, but precision drops ~2-3 pp), amplifying false\n"
            "  positives. This feature is provided for research purposes — for\n"
            "  best results, use verbose category definitions instead."
        )

    if thinking_budget and thinking_budget > 0:
        _strategy_warnings.append(
            f"[CatLLM] NOTE: thinking_budget={thinking_budget} is enabled.\n"
            "  Empirical evidence shows reasoning/thinking modes produce negligible\n"
            "  accuracy gains (<1 pp) for classification tasks, while significantly\n"
            "  increasing latency, token usage, and failure rates (up to 40% timeouts\n"
            "  observed for some models). Consider thinking_budget=0 unless you are\n"
            "  specifically researching reasoning effects."
        )

    if chain_of_thought:
        _strategy_warnings.append(
            "[CatLLM] NOTE: chain_of_thought=True is enabled.\n"
            "  Empirical evidence shows CoT has no measurable effect on structured\n"
            "  classification accuracy (~0 pp change). When categories are well-defined\n"
            "  with verbose descriptions, explicit reasoning steps add no value.\n"
            "  This won't hurt results, but it won't help either."
        )

    if step_back_prompt:
        _strategy_warnings.append(
            "[CatLLM] NOTE: step_back_prompt=True is enabled.\n"
            "  Empirical evidence shows step-back prompting produces small, inconsistent\n"
            "  gains (+0.6 pp average) and actually degrades top-tier model performance.\n"
            "  Cost: ~2x API calls per response."
        )

    if _strategy_warnings:
        print()
        print("\n\n".join(_strategy_warnings))
        print()

    # Map mode to pdf_mode
    pdf_mode = mode if mode in ("image", "text", "both") else "image"

    # =========================================================================
    # Batch mode — bypass classify_ensemble entirely
    # =========================================================================
    if batch_mode:
        from ._batch import UNSUPPORTED_BATCH_PROVIDERS, run_batch_classify
        from .text_functions_ensemble import prepare_json_schemas, prepare_model_configs

        # Guard: text input only (auto-detect)
        from .text_functions_ensemble import _detect_input_type
        detected_type = _detect_input_type(input_data)
        if detected_type in ("pdf", "image"):
            raise ValueError(
                f"batch_mode=True only supports text input, but detected input type is '{detected_type}'. "
                "Set batch_mode=False for PDF/image classification."
            )

        # Warn if progress_callback was provided (incompatible with batch)
        if progress_callback is not None:
            print(
                "[CatLLM] WARNING: progress_callback is not available in batch_mode "
                "(no per-item progress until the job completes). Ignoring callback."
            )

        # Build prompt components (mirrors what classify_ensemble does)
        categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
        survey_question_context = f"A respondent was asked: {survey_question}." if survey_question else ""
        examples = [example1, example2, example3, example4, example5, example6]
        examples_text = "\n".join(
            f"Example {i}: {ex}" for i, ex in enumerate(examples, 1) if ex is not None
        )

        model_configs = prepare_model_configs(models, auto_download=auto_download)
        json_schemas = prepare_json_schemas(model_configs, categories, use_json_schema)
        items = list(input_data) if not isinstance(input_data, list) else input_data

        if len(models) == 1:
            cfg = model_configs[0]
            if cfg["provider"] in UNSUPPORTED_BATCH_PROVIDERS:
                raise ValueError(
                    f"batch_mode=True is not supported for provider '{cfg['provider']}'. "
                    f"Supported providers: openai, anthropic, google, mistral, xai."
                )
            prompt_params = {
                "categories_str": categories_str,
                "survey_question_context": survey_question_context,
                "examples_text": examples_text,
                "chain_of_thought": chain_of_thought,
                "context_prompt": context_prompt,
                "step_back_prompt": step_back_prompt,
                "stepback_insights": {},
                "json_schema": json_schemas[cfg["model"]],
                "creativity": creativity,
                "thinking_budget": thinking_budget,
            }
            return run_batch_classify(
                items=items,
                cfg=cfg,
                categories=categories,
                prompt_params=prompt_params,
                filename=filename,
                save_directory=save_directory,
                batch_poll_interval=batch_poll_interval,
                batch_timeout=batch_timeout,
                fail_strategy=fail_strategy,
            )

        # Ensemble batch path: one job per model, run concurrently
        print(
            "[CatLLM] NOTE: batch_mode=True with multiple models is experimental. "
            "Each model submits a separate batch job concurrently. Providers without "
            "a batch API (HuggingFace, Perplexity, Ollama) fall back to synchronous calls."
        )
        from ._batch import run_batch_ensemble_classify
        prompt_params_per_model = {
            cfg["model"]: {
                "categories_str": categories_str,
                "survey_question_context": survey_question_context,
                "examples_text": examples_text,
                "chain_of_thought": chain_of_thought,
                "context_prompt": context_prompt,
                "step_back_prompt": step_back_prompt,
                "stepback_insights": {},
                "json_schema": json_schemas[cfg["model"]],
                "creativity": cfg["creativity"] if cfg["creativity"] is not None else creativity,
                "thinking_budget": thinking_budget,
            }
            for cfg in model_configs
        }
        return run_batch_ensemble_classify(
            items=items,
            model_configs=model_configs,
            categories=categories,
            prompt_params_per_model=prompt_params_per_model,
            consensus_threshold=consensus_threshold,
            fail_strategy=fail_strategy,
            filename=filename,
            save_directory=save_directory,
            batch_poll_interval=batch_poll_interval,
            batch_timeout=batch_timeout,
        )

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
