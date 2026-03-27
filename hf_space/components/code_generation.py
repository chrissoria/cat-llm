"""
Generate reproducible Python code snippets for each task type and domain.
"""

# Map domain id -> (import statement, function prefix)
DOMAIN_IMPORTS = {
    "general": ("import catllm", "catllm"),
    "survey": ("import catllm", "catllm"),
    "social_media": ("import catllm", "catllm"),
    "academic": ("import catllm", "catllm"),
    "policy": ("import catllm", "catllm"),
    "web": ("import catllm", "catllm"),
    "cognitive": ("import catllm", "catllm"),
}

# Map (domain, function) -> function name suffix
DOMAIN_FUNCTION_NAMES = {
    ("general", "classify"): "classify",
    ("general", "extract"): "extract",
    ("general", "explore"): "explore",
    ("general", "summarize"): "summarize",
    ("survey", "classify"): "classify_survey",
    ("survey", "extract"): "extract_survey",
    ("survey", "explore"): "explore_survey",
    ("survey", "summarize"): "summarize_survey",
    ("social_media", "classify"): "classify_social",
    ("social_media", "extract"): "extract_social",
    ("social_media", "explore"): "explore_social",
    ("academic", "classify"): "classify_academic",
    ("academic", "extract"): "extract_academic",
    ("academic", "explore"): "explore_academic",
    ("academic", "summarize"): "summarize_academic",
    ("policy", "classify"): "classify_policy",
    ("policy", "extract"): "extract_policy",
    ("policy", "explore"): "explore_policy",
    ("policy", "summarize"): "summarize_policy",
    ("web", "classify"): "classify_web",
    ("web", "extract"): "extract_web",
    ("web", "explore"): "explore_web",
    ("web", "summarize"): "summarize_web",
    ("cognitive", "cerad_score"): "cerad_drawn_score",
}


def _get_fn_call(domain, function):
    """Get the full function call string, e.g. 'catllm.classify_survey'."""
    imp, prefix = DOMAIN_IMPORTS.get(domain, ("import catllm", "catllm"))
    fn_name = DOMAIN_FUNCTION_NAMES.get((domain, function), function)
    return imp, f"{prefix}.{fn_name}"


def generate_classify_code(
    domain, input_type, description, categories, model, model_source,
    mode=None, classify_mode="Single Model", models_list=None,
    consensus_threshold=0.5, model_temperatures=None, ensemble_runs=None,
    domain_kwargs=None,
):
    """Generate classification code."""
    imp, fn_call = _get_fn_call(domain, "classify")
    categories_str = ",\n    ".join([f'"{cat}"' for cat in categories])

    if input_type == "text":
        input_placeholder = 'df["your_column"].tolist()'
        load_data = 'import pandas as pd\n\ndf = pd.read_csv("your_data.csv")\n'
    elif input_type == "pdf":
        input_placeholder = '"path/to/your/pdfs/"'
        load_data = ""
    else:
        input_placeholder = '"path/to/your/images/"'
        load_data = ""

    # Domain-specific kwargs
    extra_params = ""
    if domain_kwargs:
        for k, v in domain_kwargs.items():
            if v is not None and v != "":
                if isinstance(v, str):
                    extra_params += f',\n    {k}="{v}"'
                else:
                    extra_params += f",\n    {k}={v}"

    if classify_mode == "Single Model":
        mode_param = f',\n    mode="{mode}"' if mode and input_type == "pdf" else ""
        return f"""{imp}
{load_data}
categories = [
    {categories_str}
]

result = {fn_call}(
    input_data={input_placeholder},
    categories=categories,
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}"{mode_param}{extra_params}
)

result.to_csv("classified_results.csv", index=False)
"""
    else:
        # Multi-model
        if ensemble_runs:
            model_lines = [f'("{m}", "auto", "YOUR_API_KEY", {{"creativity": {t}}})' for m, t in ensemble_runs]
        elif models_list:
            model_lines = []
            for m in models_list:
                temp = (model_temperatures or {}).get(m)
                if temp is not None:
                    model_lines.append(f'("{m}", "auto", "YOUR_API_KEY", {{"creativity": {temp}}})')
                else:
                    model_lines.append(f'("{m}", "auto", "YOUR_API_KEY")')
        else:
            model_lines = ['"gpt-4o", "auto", "YOUR_API_KEY"']

        models_str = ",\n        ".join(model_lines)
        mode_param = f',\n    mode="{mode}"' if mode and input_type == "pdf" else ""
        threshold_str = "majority" if consensus_threshold == 0.5 else "two-thirds" if consensus_threshold == 0.67 else "unanimous"
        consensus_param = f',\n    consensus_threshold="{threshold_str}"' if classify_mode == "Ensemble" else ""

        return f"""{imp}
{load_data}
categories = [
    {categories_str}
]

models = [
        {models_str}
]

result = {fn_call}(
    input_data={input_placeholder},
    categories=categories,
    models=models,
    description="{description}"{mode_param}{consensus_param}{extra_params}
)

result.to_csv("classified_results.csv", index=False)
"""


def generate_extract_code(domain, input_type, description, model, model_source, max_categories, mode=None, domain_kwargs=None):
    """Generate extraction code."""
    imp, fn_call = _get_fn_call(domain, "extract")

    if input_type == "text":
        input_placeholder = 'df["your_column"].tolist()'
        load_data = 'import pandas as pd\n\ndf = pd.read_csv("your_data.csv")\n'
    elif input_type == "pdf":
        input_placeholder = '"path/to/your/pdfs/"'
        load_data = ""
    else:
        input_placeholder = '"path/to/your/images/"'
        load_data = ""

    mode_param = f',\n    mode="{mode}"' if mode else ""
    extra_params = ""
    if domain_kwargs:
        for k, v in domain_kwargs.items():
            if v is not None and v != "":
                extra_params += f',\n    {k}="{v}"' if isinstance(v, str) else f",\n    {k}={v}"

    return f"""{imp}
{load_data}
result = {fn_call}(
    input_data={input_placeholder},
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}",
    max_categories={max_categories}{mode_param}{extra_params}
)

print(result["top_categories"])
"""


def generate_summarize_code(domain, input_type, description, model, model_source, focus=None, max_length=None, instructions=None, mode=None, domain_kwargs=None):
    """Generate summarization code."""
    imp, fn_call = _get_fn_call(domain, "summarize")

    focus_param = f',\n    focus="{focus}"' if focus else ""
    length_param = f",\n    max_length={max_length}" if max_length else ""
    instructions_param = f',\n    instructions="{instructions}"' if instructions else ""

    extra_params = ""
    if domain_kwargs:
        for k, v in domain_kwargs.items():
            if v is not None and v != "":
                extra_params += f',\n    {k}="{v}"' if isinstance(v, str) else f",\n    {k}={v}"

    if input_type == "text":
        input_placeholder = 'df["your_column"].tolist()'
        load_data = 'import pandas as pd\n\ndf = pd.read_csv("your_data.csv")\n'
    else:
        input_placeholder = '"path/to/your/pdfs/"'
        load_data = ""

    mode_param = f',\n    mode="{mode}"' if mode else ""

    return f"""{imp}
{load_data}
result = {fn_call}(
    input_data={input_placeholder},
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}"{mode_param}{focus_param}{length_param}{instructions_param}{extra_params}
)

result.to_csv("summarized_results.csv", index=False)
"""
