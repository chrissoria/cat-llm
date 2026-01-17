"""
Unified functions for classification and category extraction.
"""

__all__ = ["extract", "classify", "summarize"]


def extract(
    input_data,
    api_key,
    input_type="text",
    description="",
    max_categories=12,
    categories_per_chunk=10,
    divisions=5,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    mode="text",
    filename=None,
    model_source="auto",
    iterations=3,
    random_state=None,
    focus=None,
):
    """
    Unified category extraction function for text, image, and PDF inputs.

    This function dispatches to the appropriate specialized explore function
    based on the `input_type` parameter, providing a single entry point for
    discovering categories in your data.

    Args:
        input_data: The data to explore. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path, single file, or list of image paths
            - For pdf: directory path, single file, or list of PDF paths
        api_key (str): API key for the model provider.
        input_type (str): Type of input data. Options:
            - "text" (default): Text/survey responses
            - "image": Image files
            - "pdf": PDF documents
        description (str): Description of the input data. Used as:
            - survey_question for text
            - image_description for images
            - pdf_description for PDFs
        max_categories (int): Maximum number of final categories to return.
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-4o".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        mode (str): Processing mode:
            - For text: Not used
            - For image: "image" (default) or "both"
            - For pdf: "text" (default), "image", or "both"
        filename (str): Optional CSV filename to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
            "mistral", "huggingface", "xai".
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction (e.g.,
            "decisions to move", "emotional responses"). When provided, the model
            will prioritize extracting categories related to this focus.

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step

    Examples:
        >>> import catllm as cat
        >>>
        >>> # Extract categories from survey responses
        >>> results = cat.extract(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key"
        ... )
        >>> print(results['top_categories'])
        >>>
        >>> # Extract categories from images
        >>> results = cat.extract(
        ...     input_data="/path/to/images/",
        ...     description="Product photos",
        ...     input_type="image",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Extract categories from PDFs
        >>> results = cat.extract(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     input_type="pdf",
        ...     mode="text",
        ...     api_key="your-api-key"
        ... )
    """
    input_type = input_type.lower().rstrip('s')  # Normalize: "texts" -> "text", "images" -> "image", "pdfs" -> "pdf"

    if input_type == "text":
        from .text_functions import explore_common_categories
        return explore_common_categories(
            survey_input=input_data,
            api_key=api_key,
            survey_question=description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            focus=focus,
        )

    elif input_type == "image":
        from .image_functions import explore_image_categories
        return explore_image_categories(
            image_input=input_data,
            api_key=api_key,
            image_description=description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["image", "both"] else "image",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state
        )

    elif input_type == "pdf":
        from .pdf_functions import explore_pdf_categories
        return explore_pdf_categories(
            pdf_input=input_data,
            api_key=api_key,
            pdf_description=description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["text", "image", "both"] else "text",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state
        )

    else:
        raise ValueError(
            f"input_type '{input_type}' is not supported. "
            f"Please use one of: 'text', 'image', or 'pdf'.\n\n"
            f"Examples:\n"
            f"  - For survey responses or text data: input_type='text'\n"
            f"  - For image files (.jpg, .png, etc.): input_type='image'\n"
            f"  - For PDF documents: input_type='pdf'"
        )


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
    from .text_functions_ensemble import classify_ensemble

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
    from .text_functions_ensemble import summarize_ensemble

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
