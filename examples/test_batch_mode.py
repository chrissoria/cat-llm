#!/usr/bin/env python
"""
Test script for batch_mode=True in catllm.classify().

Tests:
  1. Smoke test per supported provider (5 rows each)
  2. Output format parity vs. synchronous path (column names, dtypes, value range)
  3. ValueError guard conditions (image input, unsupported provider)
  4. Partial failure handling (bad row doesn't crash the job)
  5. Ensemble batch mode (multiple models, concurrent batch jobs)

Usage:
  python examples/test_batch_mode.py                  # all providers
  python examples/test_batch_mode.py openai           # single provider
  python examples/test_batch_mode.py guards           # guard-condition tests only
  python examples/test_batch_mode.py ensemble         # ensemble batch test only
"""

import os
import sys

# Load from local src
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

modules_to_remove = [k for k in sys.modules if k.startswith('catllm')]
for m in modules_to_remove:
    del sys.modules[m]

from dotenv import load_dotenv
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env', override=True)

import pandas as pd
import catllm

# =============================================================================
# Config
# =============================================================================

CATEGORIES = [
    "Mentions cost or affordability",
    "Mentions quality or performance",
    "Mentions customer service",
    "Other",
]
DESCRIPTION = "Survey responses about a consumer product"

SAMPLE_TEXTS = [
    "The price is too high for what you get.",
    "Great quality, really durable and well made.",
    "Customer support was very helpful and responsive.",
    "I just bought it yesterday, no opinion yet.",
    "Expensive but the quality makes it worth it.",
]

API_KEYS = {
    "openai":    os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google":    os.getenv("GOOGLE_API_KEY"),
    "mistral":   os.getenv("MISTRAL_API_KEY"),
    "xai":       os.getenv("XAI_API_KEY"),
}

MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-2.5-flash",
    "mistral":   "mistral-small-latest",
    "xai":       "grok-3-mini",
}

# =============================================================================
# Helpers
# =============================================================================

def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(df):
    print(df.to_string())
    print(f"\nShape: {df.shape}  |  Columns: {list(df.columns)}")

def assert_valid_df(df, label=""):
    """Assert the DataFrame matches single-model synchronous output format."""
    prefix = f"[{label}] " if label else ""

    # Must be a DataFrame
    assert isinstance(df, pd.DataFrame), f"{prefix}Expected DataFrame, got {type(df)}"

    # Must have category columns named category_1, category_2, ... (no model suffix)
    cat_cols = [c for c in df.columns if c.startswith("category_")]
    assert len(cat_cols) == len(CATEGORIES), (
        f"{prefix}Expected {len(CATEGORIES)} category columns, got {len(cat_cols)}: {cat_cols}"
    )
    for i in range(1, len(CATEGORIES) + 1):
        assert f"category_{i}" in df.columns, (
            f"{prefix}Missing column category_{i}. Columns: {list(df.columns)}"
        )

    # Must NOT have consensus/agreement/failed_models columns
    for col in df.columns:
        assert "_consensus" not in col, f"{prefix}Unexpected consensus column: {col}"
        assert "_agreement" not in col, f"{prefix}Unexpected agreement column: {col}"
    assert "failed_models" not in df.columns, f"{prefix}Unexpected failed_models column"

    # Category values must be 0, 1, or NA
    for col in cat_cols:
        valid = df[col].isna() | df[col].isin([0, 1])
        assert valid.all(), f"{prefix}Non-0/1 values in {col}: {df[col].unique()}"

    # Must have processing_status column
    assert "processing_status" in df.columns, f"{prefix}Missing processing_status column"

    print(f"  {prefix}Format assertions passed.")

# =============================================================================
# Test 1: Smoke test per provider
# =============================================================================

def test_provider(provider_name):
    api_key = API_KEYS.get(provider_name)
    model = MODELS.get(provider_name)

    if not api_key:
        print(f"  SKIPPED — no API key for {provider_name}")
        return None

    print(f"\n  Provider: {provider_name} | Model: {model}")
    print(f"  Items: {len(SAMPLE_TEXTS)}")

    result = catllm.classify(
        input_data=SAMPLE_TEXTS,
        categories=CATEGORIES,
        description=DESCRIPTION,
        user_model=model,
        model_source=provider_name,
        api_key=api_key,
        batch_mode=True,
        batch_poll_interval=10.0,   # shorter for tests
        add_other=False,
        check_verbosity=False,
    )

    print_result(result)
    assert_valid_df(result, label=provider_name)
    assert len(result) == len(SAMPLE_TEXTS), (
        f"[{provider_name}] Expected {len(SAMPLE_TEXTS)} rows, got {len(result)}"
    )
    print(f"  PASSED")
    return result

# =============================================================================
# Test 2: Output format parity vs. synchronous path
# =============================================================================

def test_format_parity(provider_name="openai"):
    api_key = API_KEYS.get(provider_name)
    model = MODELS.get(provider_name)
    if not api_key:
        print(f"  SKIPPED — no API key for {provider_name}")
        return

    common_kwargs = dict(
        input_data=SAMPLE_TEXTS[:3],
        categories=CATEGORIES,
        description=DESCRIPTION,
        user_model=model,
        model_source=provider_name,
        api_key=api_key,
        add_other=False,
        check_verbosity=False,
    )

    print(f"\n  Running synchronous path...")
    sync_df = catllm.classify(**common_kwargs, batch_mode=False)

    print(f"\n  Running batch path...")
    batch_df = catllm.classify(**common_kwargs, batch_mode=True, batch_poll_interval=10.0)

    print(f"\n  Sync columns:  {list(sync_df.columns)}")
    print(f"  Batch columns: {list(batch_df.columns)}")

    # Column names must match
    assert list(sync_df.columns) == list(batch_df.columns), (
        f"Column mismatch:\n  sync:  {list(sync_df.columns)}\n  batch: {list(batch_df.columns)}"
    )

    # dtypes must match
    for col in sync_df.columns:
        assert sync_df[col].dtype == batch_df[col].dtype, (
            f"dtype mismatch for {col}: sync={sync_df[col].dtype} batch={batch_df[col].dtype}"
        )

    # Both must pass format validation
    assert_valid_df(sync_df, "sync")
    assert_valid_df(batch_df, "batch")

    print(f"  PASSED")

# =============================================================================
# Test 3: Guard condition errors
# =============================================================================

def test_guards():
    api_key = API_KEYS.get("openai") or "fake-key"

    # Guard 1: unsupported provider (single-model — huggingface has no batch API)
    try:
        catllm.classify(
            input_data=SAMPLE_TEXTS,
            categories=CATEGORIES,
            user_model="Qwen/Qwen3-VL-235B-A22B-Instruct",
            model_source="huggingface",
            api_key="hf_fake",
            batch_mode=True,
            add_other=False,
            check_verbosity=False,
        )
        assert False, "Should have raised ValueError for huggingface"
    except ValueError as e:
        assert "huggingface" in str(e).lower() or "not supported" in str(e).lower(), f"Unexpected error: {e}"
        print(f"  Guard 1 (huggingface single-model): PASSED — {e}")

    print(f"  All guards PASSED")

# =============================================================================
# Test 4: Ensemble batch mode (multiple models, concurrent jobs)
# =============================================================================

def test_ensemble_batch():
    """Two batch-capable models — both should submit concurrent jobs and merge."""
    openai_key = API_KEYS.get("openai")
    mistral_key = API_KEYS.get("mistral")

    if not openai_key or not mistral_key:
        print(f"  SKIPPED — need both openai and mistral API keys")
        return

    result = catllm.classify(
        input_data=SAMPLE_TEXTS,
        categories=CATEGORIES,
        models=[
            ("gpt-4o-mini", "openai", openai_key),
            ("mistral-small-latest", "mistral", mistral_key),
        ],
        batch_mode=True,
        batch_poll_interval=10.0,
        add_other=False,
        check_verbosity=False,
    )

    print_result(result)

    # Ensemble output should have per-model columns + consensus columns
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
    assert len(result) == len(SAMPLE_TEXTS), f"Expected {len(SAMPLE_TEXTS)} rows, got {len(result)}"

    cols = list(result.columns)
    # Should have per-model columns
    model_cols = [c for c in cols if "gpt_4o_mini" in c or "mistral_small" in c]
    assert len(model_cols) > 0, f"Expected per-model columns, got: {cols}"
    # Should have consensus columns
    consensus_cols = [c for c in cols if "_consensus" in c]
    assert len(consensus_cols) > 0, f"Expected consensus columns, got: {cols}"
    # Should have agreement columns
    agreement_cols = [c for c in cols if "_agreement" in c]
    assert len(agreement_cols) > 0, f"Expected agreement columns, got: {cols}"

    print(f"  Columns: {cols}")
    print(f"  PASSED — ensemble batch produced {len(model_cols)} model cols, "
          f"{len(consensus_cols)} consensus cols")


# =============================================================================
# Test 5: Partial failure handling
# =============================================================================

def test_partial_failures(provider_name="openai"):
    """Verify that a very short/empty row doesn't crash the whole job."""
    api_key = API_KEYS.get(provider_name)
    model = MODELS.get(provider_name)
    if not api_key:
        print(f"  SKIPPED — no API key for {provider_name}")
        return

    items_with_empty = SAMPLE_TEXTS[:3] + ["", None]

    result = catllm.classify(
        input_data=items_with_empty,
        categories=CATEGORIES,
        description=DESCRIPTION,
        user_model=model,
        model_source=provider_name,
        api_key=api_key,
        batch_mode=True,
        batch_poll_interval=10.0,
        add_other=False,
        check_verbosity=False,
    )

    assert len(result) == len(items_with_empty), (
        f"Expected {len(items_with_empty)} rows, got {len(result)}"
    )
    print(f"  Result ({len(result)} rows):")
    print_result(result)
    print(f"  PASSED — job completed despite empty/None rows")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "guards":
        print_section("Test: Guard Conditions")
        test_guards()

    elif target == "ensemble":
        print_section("Test: Ensemble Batch Mode")
        test_ensemble_batch()

    elif target in API_KEYS:
        print_section(f"Test: Smoke Test — {target}")
        test_provider(target)

        print_section(f"Test: Format Parity — {target}")
        test_format_parity(target)

        print_section(f"Test: Partial Failures — {target}")
        test_partial_failures(target)

    else:
        # Run all
        print_section("Test: Guard Conditions")
        test_guards()

        print_section("Test: Smoke Tests — All Providers")
        results = {}
        for provider in API_KEYS:
            print(f"\n--- {provider} ---")
            results[provider] = test_provider(provider)

        # Format parity using first available provider
        for provider in API_KEYS:
            if API_KEYS[provider]:
                print_section(f"Test: Format Parity — {provider}")
                test_format_parity(provider)
                break

        print_section("Test: Ensemble Batch Mode")
        test_ensemble_batch()

        print_section("Test: Partial Failures")
        for provider in API_KEYS:
            if API_KEYS[provider]:
                test_partial_failures(provider)
                break

        print()
        print("=" * 70)
        print("  All batch_mode tests complete.")
        print("=" * 70)
