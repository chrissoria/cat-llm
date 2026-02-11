"""
Category exploration functions for CatLLM.

This module provides raw category extraction from text inputs,
returning unprocessed category lists for frequency/saturation analysis.
"""

import pandas as pd

__all__ = [
    "explore",
]

from .text_functions import explore_common_categories


def explore(
    input_data,
    api_key,
    description="",
    max_categories=12,
    categories_per_chunk=10,
    divisions=12,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    filename=None,
    model_source="auto",
    iterations=8,
    random_state=None,
    focus=None,
    progress_callback=None,
):
    """
    Explore categories in text data, returning the raw extracted list.

    Unlike extract(), which normalizes, deduplicates, and semantically merges
    categories, explore() returns every category string from every chunk across
    every iteration — with duplicates intact. This is useful for analyzing
    category stability and saturation across repeated extraction runs.

    Args:
        input_data: List of text responses or pandas Series.
        api_key (str): API key for the model provider.
        description (str): The survey question or description of the data.
        max_categories (int): Maximum categories per chunk (passed through).
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-4o".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        filename (str): Optional CSV filename to save raw category list.
        model_source (str): Provider - "auto", "openai", "anthropic", etc.
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction.
        progress_callback (callable): Optional callback for progress updates.

    Returns:
        list[str]: Every category string extracted from every chunk across
        every iteration. Length ≈ iterations × divisions × categories_per_chunk.

    Examples:
        >>> import catllm as cat
        >>>
        >>> raw_categories = cat.explore(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key",
        ...     iterations=3,
        ...     divisions=5,
        ... )
        >>> print(len(raw_categories))  # ~150
        >>> print(raw_categories[:5])
    """
    raw_items = explore_common_categories(
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
        filename=None,  # We handle saving ourselves
        model_source=model_source,
        iterations=iterations,
        random_state=random_state,
        focus=focus,
        progress_callback=progress_callback,
        return_raw=True,
    )

    if filename:
        df = pd.DataFrame(raw_items, columns=["Category"])
        df.to_csv(filename, index=False)
        print(f"Raw categories saved to {filename}")

    return raw_items
