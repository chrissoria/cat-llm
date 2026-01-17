# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: MIT

from .__about__ import (
    __version__,
    __author__,
    __description__,
    __title__,
    __url__,
    __license__,
)

# =============================================================================
# Public API - Organized by function type
# =============================================================================

# Main entry points
from .extract import extract
from .classify import classify
from .summarize import summarize

# =============================================================================
# Provider utilities (for advanced users)
# =============================================================================
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
    set_ollama_endpoint,
    check_ollama_running,
    list_ollama_models,
    check_ollama_model,
    pull_ollama_model,
    PROVIDER_CONFIG,
)

# =============================================================================
# Backward compatibility - Deprecated functions
# These are kept for backward compatibility but users should migrate to the
# new unified API (extract, classify, summarize)
# =============================================================================

# Extraction functions (use extract() instead)
from .extract import (
    explore_common_categories,
    explore_corpus,
    explore_image_categories,
    explore_pdf_categories,
)

# Classification functions (use classify() instead)
from .classify import (
    classify_ensemble,
    multi_class,
    image_multi_class,
    pdf_multi_class,
)

# Summarization functions (use summarize() instead)
from .summarize import summarize_ensemble

# =============================================================================
# Domain-specific functions
# =============================================================================
from .CERAD_functions import *

# =============================================================================
# Additional utilities from existing modules (backward compatibility)
# =============================================================================
from .text_functions import (
    build_json_schema,
    extract_json,
    validate_classification_json,
)

from .image_functions import (
    image_score_drawing,
    image_features,
)

# Define public API
__all__ = [
    # Main entry points
    "extract",
    "classify",
    "summarize",
    # Provider utilities
    "UnifiedLLMClient",
    "detect_provider",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "pull_ollama_model",
    "PROVIDER_CONFIG",
    # Domain-specific
    "cerad_drawn_score",
    # Deprecated (backward compatibility)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
    "classify_ensemble",
    "summarize_ensemble",
    "multi_class",
    "image_multi_class",
    "pdf_multi_class",
    "image_score_drawing",
    "image_features",
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
]