"""
Unified classification function for text, image, and PDF inputs.
"""


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
    research_question=None
):
    """
    Unified classification function for text, image, and PDF inputs.

    This function dispatches to the appropriate specialized function based on
    the `input_type` parameter, providing a single entry point for all
    classification tasks.

    Args:
        input_data: The data to classify. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path or list of image file paths
            - For pdf: directory path or list of PDF file paths
        categories (list or "auto"): List of category names for classification,
            or "auto" to automatically extract categories (text only).
        api_key (str): API key for the model provider.
        input_type (str): Type of input data. Options:
            - "text" (default): Text classification
            - "image": Image classification
            - "pdf": PDF page classification
        description (str): Description of the input data. Used as:
            - survey_question for text
            - image_description for images
            - pdf_description for PDFs
        user_model (str): Model name to use. Default "gpt-4o".
        mode (str): PDF processing mode (only used when input_type="pdf"):
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
        max_categories (int): Max categories for auto mode (text only).
        categories_per_chunk (int): Categories per chunk for auto mode (text only).
        divisions (int): Number of divisions for auto mode (text only).
        research_question (str): Research question for auto mode (text only).

    Returns:
        pd.DataFrame: Results with classification columns.

    Examples:
        >>> import catllm as cat
        >>>
        >>> # Text classification (default)
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     description="Customer feedback survey",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Image classification
        >>> results = cat.classify(
        ...     input_data="/path/to/images/",
        ...     categories=["Has person", "Outdoor scene"],
        ...     description="Product photos",
        ...     input_type="image",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # PDF classification
        >>> results = cat.classify(
        ...     input_data="/path/to/pdfs/",
        ...     categories=["Contains table", "Has chart"],
        ...     description="Financial reports",
        ...     input_type="pdf",
        ...     mode="both",
        ...     api_key="your-api-key"
        ... )
    """
    input_type = input_type.lower().rstrip('s')  # Normalize: "texts" -> "text", "images" -> "image", "pdfs" -> "pdf"

    if input_type == "text":
        from .text_functions import multi_class
        return multi_class(
            survey_input=input_data,
            categories=categories,
            api_key=api_key,
            user_model=user_model,
            survey_question=description,
            example1=example1,
            example2=example2,
            example3=example3,
            example4=example4,
            example5=example5,
            example6=example6,
            creativity=creativity,
            safety=safety,
            chain_of_verification=chain_of_verification,
            chain_of_thought=chain_of_thought,
            step_back_prompt=step_back_prompt,
            context_prompt=context_prompt,
            thinking_budget=thinking_budget,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            research_question=research_question,
            filename=filename,
            save_directory=save_directory,
            model_source=model_source
        )

    elif input_type == "image":
        from .image_functions import image_multi_class
        return image_multi_class(
            image_description=description,
            image_input=input_data,
            categories=categories,
            api_key=api_key,
            user_model=user_model,
            creativity=creativity,
            safety=safety,
            chain_of_verification=chain_of_verification,
            chain_of_thought=chain_of_thought,
            step_back_prompt=step_back_prompt,
            context_prompt=context_prompt,
            thinking_budget=thinking_budget,
            example1=example1,
            example2=example2,
            example3=example3,
            example4=example4,
            example5=example5,
            example6=example6,
            filename=filename,
            save_directory=save_directory,
            model_source=model_source
        )

    elif input_type == "pdf":
        from .pdf_functions import pdf_multi_class
        return pdf_multi_class(
            pdf_description=description,
            pdf_input=input_data,
            categories=categories,
            api_key=api_key,
            user_model=user_model,
            mode=mode,
            creativity=creativity,
            safety=safety,
            chain_of_verification=chain_of_verification,
            chain_of_thought=chain_of_thought,
            step_back_prompt=step_back_prompt,
            context_prompt=context_prompt,
            thinking_budget=thinking_budget,
            example1=example1,
            example2=example2,
            example3=example3,
            example4=example4,
            example5=example5,
            example6=example6,
            filename=filename,
            save_directory=save_directory,
            model_source=model_source
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
