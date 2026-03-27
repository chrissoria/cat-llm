"""
Domain registry: maps domain IDs to labels, available functions, and catllm callables.
"""

import catllm

# Each domain defines:
#   label: Display name
#   functions: dict of function_id -> {label, callable}
#   available: whether the domain package is importable

DOMAINS = {}


def _register(domain_id, label, functions):
    """Register a domain if its functions are importable."""
    fn_map = {}
    for fn_id, fn_info in functions.items():
        fn_map[fn_id] = fn_info
    DOMAINS[domain_id] = {"label": label, "fn_map": fn_map}


# General (cat-stack) -- always available
_register("general", "General", {
    "classify":  {"label": "Classify",  "callable": catllm.classify},
    "extract":   {"label": "Extract Categories", "callable": catllm.extract},
    "explore":   {"label": "Explore Categories", "callable": catllm.explore},
    "summarize": {"label": "Summarize", "callable": catllm.summarize},
})

# Survey (cat-survey)
try:
    _register("survey", "Survey", {
        "classify":  {"label": "Classify",  "callable": catllm.classify_survey},
        "extract":   {"label": "Extract Categories", "callable": catllm.extract_survey},
        "explore":   {"label": "Explore Categories", "callable": catllm.explore_survey},
        "summarize": {"label": "Summarize", "callable": catllm.summarize_survey},
    })
except AttributeError:
    pass

# Social Media (cat-vader) -- no summarize
try:
    _register("social_media", "Social Media", {
        "classify": {"label": "Classify", "callable": catllm.classify_social},
        "extract":  {"label": "Extract Categories", "callable": catllm.extract_social},
        "explore":  {"label": "Explore Categories", "callable": catllm.explore_social},
    })
except AttributeError:
    pass

# Academic (cat-ademic)
try:
    _register("academic", "Academic", {
        "classify":  {"label": "Classify",  "callable": catllm.classify_academic},
        "extract":   {"label": "Extract Categories", "callable": catllm.extract_academic},
        "explore":   {"label": "Explore Categories", "callable": catllm.explore_academic},
        "summarize": {"label": "Summarize", "callable": catllm.summarize_academic},
    })
except AttributeError:
    pass

# Policy (cat-pol)
try:
    _register("policy", "Policy", {
        "classify":  {"label": "Classify",  "callable": catllm.classify_policy},
        "extract":   {"label": "Extract Categories", "callable": catllm.extract_policy},
        "explore":   {"label": "Explore Categories", "callable": catllm.explore_policy},
        "summarize": {"label": "Summarize", "callable": catllm.summarize_policy},
    })
except AttributeError:
    pass

# Web (cat-web)
try:
    _register("web", "Web", {
        "classify":  {"label": "Classify",  "callable": catllm.classify_web},
        "extract":   {"label": "Extract Categories", "callable": catllm.extract_web},
        "explore":   {"label": "Explore Categories", "callable": catllm.explore_web},
        "summarize": {"label": "Summarize", "callable": catllm.summarize_web},
    })
except AttributeError:
    pass

# Cognitive (cat-cog)
try:
    _register("cognitive", "Cognitive", {
        "cerad_score": {"label": "CERAD Score", "callable": catllm.cerad_drawn_score},
    })
except AttributeError:
    pass


def get_domain_ids():
    """Return ordered list of available domain IDs."""
    order = ["general", "survey", "social_media", "academic", "policy", "web", "cognitive"]
    return [d for d in order if d in DOMAINS]


def get_domain_label(domain_id):
    """Get display label for a domain."""
    return DOMAINS.get(domain_id, {}).get("label", domain_id)


def get_functions(domain_id):
    """Return dict of function_id -> {label, callable} for a domain."""
    return DOMAINS.get(domain_id, {}).get("fn_map", {})


def get_callable(domain_id, function_id):
    """Get the catllm callable for a (domain, function) pair."""
    fn_map = get_functions(domain_id)
    fn_info = fn_map.get(function_id)
    if fn_info:
        return fn_info["callable"]
    return None
