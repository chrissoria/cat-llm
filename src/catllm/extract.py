"""
Category extraction functions for CatLLM.

This module provides unified category extraction from text, image, and PDF inputs.
"""

import warnings

__all__ = [
    # Main entry point
    "extract",
    # Input-specific functions (for backward compatibility)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions import (
    explore_common_categories,
    explore_corpus,
)

from .image_functions import (
    explore_image_categories,
)

from .pdf_functions import (
    explore_pdf_categories,
)


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
    progress_callback=None,
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
        progress_callback (callable): Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).

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
            progress_callback=progress_callback,
        )

    elif input_type == "image":
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
            random_state=random_state,
            progress_callback=progress_callback,
        )

    elif input_type == "pdf":
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
            random_state=random_state,
            progress_callback=progress_callback,
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
