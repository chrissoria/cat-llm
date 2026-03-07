"""
Embedding-based similarity scores for CatLLM.

Uses a local sentence-transformer model (BAAI/bge-small-en-v1.5, 33M params,
~130MB) to compute cosine similarity between each input text and each category.
Scores are independent per (text, category) pair — no softmax across categories,
since this is multi-label classification.

The embeddings feature is opt-in via embeddings=True on classify(). It adds
`_similarity` columns alongside the existing binary 0/1 classification columns.

Requires: pip install cat-llm[embeddings]
"""

import pandas as pd

_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _check_dependencies():
    """Check that sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "The embeddings feature requires sentence-transformers.\n"
            "Install with: pip install cat-llm[embeddings]\n"
            "  (requires: sentence-transformers, which pulls in torch and transformers)"
        )


def _is_model_cached() -> bool:
    """Check if the embedding model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(_EMBEDDING_MODEL_NAME, "config.json")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


def ensure_embeddings_available() -> bool:
    """
    Ensure the embedding model is available, prompting to download if needed.

    Returns:
        True if the model is ready to use, False if user declined download.
    """
    _check_dependencies()

    if _is_model_cached():
        return True

    print(
        "\n[CatLLM] The embedding model (~130MB) will be downloaded from\n"
        f"  HuggingFace Hub ({_EMBEDDING_MODEL_NAME}).\n"
        "  This is a one-time download — the model is cached locally after."
    )
    try:
        answer = input("  Continue? (Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("", "y", "yes"):
        return True
    else:
        print("  -> Embedding scores disabled for this run.\n")
        return False


def load_embedding_model():
    """
    Load and return the sentence-transformer embedding model.

    Returns:
        SentenceTransformer model instance.
    """
    _check_dependencies()

    from sentence_transformers import SentenceTransformer

    print(f"[CatLLM] Loading embedding model ({_EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    print("[CatLLM] Embedding model ready.")
    return model


def compute_embedding_scores(texts, categories, model, category_descriptions=None):
    """
    Compute cosine similarity scores between texts and categories.

    Each (text, category) score is independent — no softmax across categories.
    Raw cosine similarity is rescaled from [-1, 1] to [0, 1] via (sim + 1) / 2.

    Args:
        texts: List of input text strings.
        categories: List of category name strings.
        model: Loaded SentenceTransformer model.
        category_descriptions: Optional dict mapping category names to richer
            descriptions for embedding (e.g., {"Past_Support": "References to
            help received from family in the past"}).

    Returns:
        Dict mapping "category_N_similarity" -> list of float scores, where N
        is 1-indexed to match the existing classification column naming.
    """
    from sentence_transformers import util

    # Convert NaN/None to empty string
    clean_texts = [str(t) if pd.notna(t) else "" for t in texts]

    # Build category strings for embedding
    cat_strings = []
    for cat in categories:
        if category_descriptions and cat in category_descriptions:
            cat_strings.append(f"{cat}: {category_descriptions[cat]}")
        else:
            cat_strings.append(cat)

    # Encode all texts and categories
    text_embeddings = model.encode(clean_texts, normalize_embeddings=True,
                                   show_progress_bar=len(clean_texts) > 100)
    cat_embeddings = model.encode(cat_strings, normalize_embeddings=True)

    # Compute cosine similarity matrix: (num_texts, num_categories)
    sim_matrix = util.cos_sim(text_embeddings, cat_embeddings)

    # Rescale from [-1, 1] to [0, 1]
    scores = (sim_matrix + 1) / 2

    # Build output dict
    result = {}
    for i, _cat in enumerate(categories):
        col_name = f"category_{i + 1}_similarity"
        result[col_name] = [round(float(scores[row][i]), 4) for row in range(len(clean_texts))]

    return result


def apply_embedding_scores(df, categories, embedding_model, category_descriptions=None):
    """
    Insert embedding similarity columns into a result DataFrame.

    For each category N, a `category_N_similarity` column is inserted after the
    last existing column that belongs to that category number.

    Args:
        df: Result DataFrame from classify (single-model or ensemble).
        categories: List of category name strings.
        embedding_model: Loaded SentenceTransformer model.
        category_descriptions: Optional dict mapping category names to descriptions.

    Returns:
        DataFrame with `_similarity` columns inserted.
    """
    # Find the text column to use for embedding
    if "survey_input" in df.columns:
        texts = df["survey_input"].tolist()
    else:
        # Fallback: use first column
        texts = df.iloc[:, 0].tolist()

    scores = compute_embedding_scores(texts, categories, embedding_model,
                                      category_descriptions)

    # Insert each _similarity column after the last column for that category number
    result_df = df.copy()
    for i in range(len(categories)):
        prob_col = f"category_{i + 1}_similarity"
        prob_values = scores[prob_col]

        # Find the last column that starts with "category_{N}_" or equals "category_{N}"
        # Use exact match on the number to avoid category_1 matching category_10
        cat_prefix = f"category_{i + 1}_"
        cat_exact = f"category_{i + 1}"

        last_pos = -1
        for col_idx, col_name in enumerate(result_df.columns):
            if col_name == cat_exact or col_name.startswith(cat_prefix):
                last_pos = col_idx

        if last_pos >= 0:
            # Insert after the last matching column
            result_df.insert(last_pos + 1, prob_col, prob_values)
        else:
            # No matching column found — append at the end
            result_df[prob_col] = prob_values

    return result_df
