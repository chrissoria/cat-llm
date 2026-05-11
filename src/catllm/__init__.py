# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
cat-llm — the meta-package for the CatLLM ecosystem.

Installing cat-llm pulls in all domain packages. Use ``import catllm`` and
tab-complete to find every available function.

Domain-neutral (from cat-stack):
    classify, extract, explore, summarize

Survey (from cat-survey):
    classify_survey, extract_survey, explore_survey, summarize_survey

Social media (from cat-vader):
    classify_social, extract_social, explore_social

Academic (from cat-ademic):
    classify_academic, extract_academic, explore_academic, summarize_academic

Political text (from cat-pol):
    classify_policy, extract_policy, explore_policy, summarize_policy

Web content (from cat-web):
    classify_web, extract_web, explore_web, summarize_web

Cognitive assessment (from cat-cog):
    cerad_drawn_score
"""

from .__about__ import (
    __version__,
    __author__,
    __description__,
    __title__,
    __url__,
    __license__,
)

# =============================================================================
# Domain-neutral base (cat-stack)
# =============================================================================
from catstack import classify, extract, explore, summarize

# Provider utilities (re-exported for backward compatibility)
from catstack import (
    UnifiedLLMClient,
    detect_provider,
    set_ollama_endpoint,
    check_ollama_running,
    list_ollama_models,
    check_ollama_model,
    pull_ollama_model,
    PROVIDER_CONFIG,
    BatchJobExpiredError,
    BatchJobFailedError,
    has_other_category,
    check_category_verbosity,
    build_json_schema,
    extract_json,
    validate_classification_json,
    image_score_drawing,
    image_features,
)

# Backward-compatible deprecated functions (re-exported from cat-stack)
from catstack import (
    explore_common_categories,
    explore_corpus,
    explore_image_categories,
    explore_pdf_categories,
    classify_ensemble,
    multi_class,
    image_multi_class,
    pdf_multi_class,
    summarize_ensemble,
)

# =============================================================================
# Survey (cat-survey)
# =============================================================================
from catsurvey import classify as classify_survey
from catsurvey import extract as extract_survey
from catsurvey import explore as explore_survey
from catsurvey import summarize as summarize_survey

# =============================================================================
# Social media (cat-vader)
# =============================================================================
from catvader import classify as classify_social
from catvader import extract as extract_social
from catvader import explore as explore_social

# =============================================================================
# Academic (cat-ademic)
# =============================================================================
from catademic import classify as classify_academic
from catademic import extract as extract_academic
from catademic import explore as explore_academic
from catademic import summarize as summarize_academic

# =============================================================================
# Political text (cat-pol)
# =============================================================================
from catpol import classify as classify_policy
from catpol import extract as extract_policy
from catpol import explore as explore_policy
from catpol import summarize as summarize_policy
from catpol import prompt_tune as prompt_tune_policy
from catpol import list_sources as list_policy_sources
from catpol import fetch_source as fetch_policy_source

# =============================================================================
# Web content (cat-web)
# =============================================================================
from catweb import classify as classify_web
from catweb import extract as extract_web
from catweb import explore as explore_web
from catweb import summarize as summarize_web

# =============================================================================
# Cognitive assessment (cat-cog)
# =============================================================================
from catcog import cerad_drawn_score

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Domain-neutral (cat-stack)
    "classify",
    "extract",
    "explore",
    "summarize",
    # Survey (cat-survey)
    "classify_survey",
    "extract_survey",
    "explore_survey",
    "summarize_survey",
    # Social media (cat-vader)
    "classify_social",
    "extract_social",
    "explore_social",
    # Academic (cat-ademic)
    "classify_academic",
    "extract_academic",
    "explore_academic",
    "summarize_academic",
    # Political text (cat-pol)
    "classify_policy",
    "extract_policy",
    "explore_policy",
    "summarize_policy",
    "prompt_tune_policy",
    "list_policy_sources",
    "fetch_policy_source",
    # Web content (cat-web)
    "classify_web",
    "extract_web",
    "explore_web",
    "summarize_web",
    # Cognitive assessment (cat-cog)
    "cerad_drawn_score",
    # Provider utilities
    "UnifiedLLMClient",
    "detect_provider",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "pull_ollama_model",
    "PROVIDER_CONFIG",
    "BatchJobExpiredError",
    "BatchJobFailedError",
    "has_other_category",
    "check_category_verbosity",
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
    "image_score_drawing",
    "image_features",
    # Deprecated (backward compatibility)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
    "classify_ensemble",
    "multi_class",
    "image_multi_class",
    "pdf_multi_class",
    "summarize_ensemble",
]
