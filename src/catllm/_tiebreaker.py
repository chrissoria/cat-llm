"""
Embedding-based centroid tiebreaker for ensemble classification.

Resolves true ties in ensemble consensus by building per-category centroids
from unanimously-agreed rows, then comparing tied texts to those centroids.

This module is called after ensemble classification completes but before
building output DataFrames. It mutates all_results in place, updating
consensus values for tied rows and adding tiebreaker_resolved metadata.

How it works:
1. Identify "confident" rows — where every model unanimously agrees
2. Embed all texts, compute mean embedding per category from confident rows
3. For true ties only (positive_rate == threshold exactly), compare text
   embedding to positive vs negative centroid
4. Resolve: pick whichever centroid is closer; if only positive centroid
   exists, use absolute similarity threshold

Requires: pip install cat-llm[embeddings]
"""

import numpy as np
import pandas as pd


def _compute_centroid(embeddings_matrix):
    """
    Compute L2-normalized centroid (mean embedding) from a matrix of embeddings.

    Args:
        embeddings_matrix: numpy array of shape (N, D) with N embeddings.

    Returns:
        L2-normalized centroid vector of shape (D,).
    """
    mean_vec = np.mean(embeddings_matrix, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec


def _find_confident_and_tied_rows(all_results, category_key, threshold):
    """
    Bucket rows into confident-positive, confident-negative, and true-tied.

    A row is "confident" when all successful models unanimously agree (all 1s
    or all 0s). A row is "true-tied" when positive_rate == threshold exactly
    (e.g., 2-2 split with majority vote threshold=0.5).

    Args:
        all_results: List of result dicts from classify_ensemble.
        category_key: String key like "1", "2", etc.
        threshold: Numeric consensus threshold (0-1).

    Returns:
        (confident_positive_indices, confident_negative_indices, tied_indices)
        Each is a list of row indices into all_results.
    """
    confident_pos = []
    confident_neg = []
    tied = []

    for row_idx, result in enumerate(all_results):
        if result.get("skipped"):
            continue

        aggregated = result["aggregated"]
        if aggregated.get("error"):
            continue

        per_model = aggregated.get("per_model", {})
        if not per_model:
            continue

        # Collect votes for this category from all successful models
        votes = []
        for model_name, parsed in per_model.items():
            vote_str = parsed.get(category_key, "0")
            try:
                votes.append(int(vote_str))
            except (ValueError, TypeError):
                votes.append(0)

        if not votes:
            continue

        num_models = len(votes)
        positive_count = sum(votes)
        positive_rate = positive_count / num_models

        # Check for true tie: positive_rate == threshold exactly
        if abs(positive_rate - threshold) < 1e-9:
            tied.append(row_idx)
        elif positive_rate == 1.0:
            # All models agree positive
            confident_pos.append(row_idx)
        elif positive_rate == 0.0:
            # All models agree negative
            confident_neg.append(row_idx)

    return confident_pos, confident_neg, tied


def resolve_ties_with_centroids(
    all_results,
    categories,
    embedding_model,
    consensus_threshold,
    min_centroid_size=3,
):
    """
    Resolve true ties in ensemble consensus using embedding centroids.

    Builds per-category centroids from texts where ALL models unanimously agree,
    then uses cosine similarity to those centroids to break ties.

    Only mutates rows where positive_rate == threshold exactly (true ties).
    Rows with clear majorities are left alone.

    Args:
        all_results: List of result dicts from classify_ensemble (mutated in place).
        categories: List of category name strings.
        embedding_model: Loaded SentenceTransformer model.
        consensus_threshold: Numeric threshold (already resolved to float).
        min_centroid_size: Minimum confident rows needed to build a centroid.

    Returns:
        Dict with summary stats:
            - total_ties: number of true ties found across all categories
            - resolved: number of ties resolved by centroid
            - skipped_categories: categories skipped (insufficient confident data)
    """
    threshold = consensus_threshold

    # Quick scan: any ties at all?
    has_any_ties = False
    for cat_idx in range(len(categories)):
        cat_key = str(cat_idx + 1)
        _, _, tied = _find_confident_and_tied_rows(all_results, cat_key, threshold)
        if tied:
            has_any_ties = True
            break

    if not has_any_ties:
        print("[CatLLM] Embedding tiebreaker: no true ties found — skipping.")
        # Still mark all rows as resolved by vote for consistent output columns
        _mark_all_as_vote(all_results, categories)
        return {"total_ties": 0, "resolved": 0, "skipped_categories": []}

    # Encode all non-skipped texts in one batch
    texts = []
    row_to_embed_idx = {}  # row_idx -> index in texts list
    for row_idx, result in enumerate(all_results):
        if result.get("skipped"):
            continue
        raw_text = result.get("_original_item", result.get("response", ""))
        if pd.notna(raw_text) and str(raw_text).strip():
            row_to_embed_idx[row_idx] = len(texts)
            texts.append(str(raw_text))

    if not texts:
        return {"total_ties": 0, "resolved": 0, "skipped_categories": []}

    print(f"[CatLLM] Embedding tiebreaker: encoding {len(texts)} texts...")
    all_embeddings = embedding_model.encode(
        texts, normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )

    total_ties = 0
    total_resolved = 0
    skipped_categories = []

    for cat_idx in range(len(categories)):
        cat_key = str(cat_idx + 1)
        cat_name = categories[cat_idx]

        confident_pos, confident_neg, tied = _find_confident_and_tied_rows(
            all_results, cat_key, threshold,
        )

        if not tied:
            continue

        total_ties += len(tied)

        # Build positive centroid from confident-positive rows
        pos_embed_indices = [
            row_to_embed_idx[r] for r in confident_pos if r in row_to_embed_idx
        ]
        neg_embed_indices = [
            row_to_embed_idx[r] for r in confident_neg if r in row_to_embed_idx
        ]

        if len(pos_embed_indices) < min_centroid_size:
            skipped_categories.append(cat_name)
            # Mark these tied rows as resolved by vote (unchanged)
            for row_idx in tied:
                agg = all_results[row_idx]["aggregated"]
                if "tiebreaker_resolved" not in agg:
                    agg["tiebreaker_resolved"] = {}
                agg["tiebreaker_resolved"][cat_key] = "vote"
            continue

        pos_centroid = _compute_centroid(all_embeddings[pos_embed_indices])

        # Build negative centroid if enough data
        neg_centroid = None
        if len(neg_embed_indices) >= min_centroid_size:
            neg_centroid = _compute_centroid(all_embeddings[neg_embed_indices])

        # Resolve each tied row
        resolved_this_cat = 0
        for row_idx in tied:
            if row_idx not in row_to_embed_idx:
                continue

            embed_idx = row_to_embed_idx[row_idx]
            text_embedding = all_embeddings[embed_idx]

            # Cosine similarity (embeddings are already normalized, so dot product)
            sim_to_pos = float(np.dot(text_embedding, pos_centroid))

            if neg_centroid is not None:
                sim_to_neg = float(np.dot(text_embedding, neg_centroid))
                new_consensus = "1" if sim_to_pos >= sim_to_neg else "0"
            else:
                # Only positive centroid — use absolute threshold (0.5 on similarity)
                new_consensus = "1" if sim_to_pos >= 0.5 else "0"

            agg = all_results[row_idx]["aggregated"]
            agg["consensus"][cat_key] = new_consensus
            if "tiebreaker_resolved" not in agg:
                agg["tiebreaker_resolved"] = {}
            agg["tiebreaker_resolved"][cat_key] = "centroid"
            resolved_this_cat += 1

        total_resolved += resolved_this_cat

    # Mark all non-tied rows as resolved by vote
    _mark_all_as_vote(all_results, categories)

    if skipped_categories:
        print(
            f"[CatLLM] Embedding tiebreaker: skipped {len(skipped_categories)} "
            f"categor{'y' if len(skipped_categories) == 1 else 'ies'} "
            f"(fewer than {min_centroid_size} confident rows): "
            f"{', '.join(skipped_categories)}"
        )

    print(
        f"[CatLLM] Embedding tiebreaker: {total_ties} true ties found, "
        f"{total_resolved} resolved by centroid."
    )

    return {
        "total_ties": total_ties,
        "resolved": total_resolved,
        "skipped_categories": skipped_categories,
    }


def _mark_all_as_vote(all_results, categories):
    """Fill in 'vote' for any row/category not already marked by the tiebreaker."""
    for result in all_results:
        if result.get("skipped"):
            continue
        agg = result["aggregated"]
        if agg.get("error"):
            continue
        if "tiebreaker_resolved" not in agg:
            agg["tiebreaker_resolved"] = {}
        for cat_idx in range(len(categories)):
            cat_key = str(cat_idx + 1)
            if cat_key not in agg["tiebreaker_resolved"]:
                agg["tiebreaker_resolved"][cat_key] = "vote"
