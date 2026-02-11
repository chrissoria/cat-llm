"""
Text classification functions for CatLLM.

This module provides multi-class text classification using a unified HTTP-based approach
that works with multiple LLM providers (OpenAI, Anthropic, Google, Mistral, xAI,
Perplexity, HuggingFace, and Ollama) without requiring provider-specific SDKs.
"""

import json
import warnings

# Exported names (excludes deprecated multi_class)
__all__ = [
    "UnifiedLLMClient",
    "detect_provider",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "check_system_resources",
    "get_ollama_model_size_estimate",
    "pull_ollama_model",
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
    "ollama_two_step_classify",
    "explore_corpus",
    "explore_common_categories",
    # Internal utilities used by other modules
    "_detect_model_source",
    "_get_stepback_insight",
    "_detect_huggingface_endpoint",
]
import pandas as pd
import regex
from tqdm import tqdm

from .calls.stepback import (
    get_stepback_insight_openai,
    get_stepback_insight_anthropic,
    get_stepback_insight_google,
    get_stepback_insight_mistral
)
from .calls.CoVe import (
    chain_of_verification_openai,
    chain_of_verification_google,
    chain_of_verification_anthropic,
    chain_of_verification_mistral
)
from .calls.top_n import (
    get_openai_top_n,
    get_anthropic_top_n,
    get_google_top_n,
    get_mistral_top_n
)

from ._providers import (
    UnifiedLLMClient,
    PROVIDER_CONFIG,
    detect_provider,
    _detect_model_source,
    _detect_huggingface_endpoint,
    set_ollama_endpoint,
    check_ollama_running,
    list_ollama_models,
    check_ollama_model,
    check_system_resources,
    get_ollama_model_size_estimate,
    pull_ollama_model,
    OLLAMA_MODEL_SIZES,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _get_stepback_insight(model_source, stepback, api_key, user_model, creativity):
    """Get step-back insight using the appropriate provider."""
    stepback_functions = {
        "openai": get_stepback_insight_openai,
        "perplexity": get_stepback_insight_openai,
        "huggingface": get_stepback_insight_openai,
        "huggingface-together": get_stepback_insight_openai,
        "xai": get_stepback_insight_openai,
        "anthropic": get_stepback_insight_anthropic,
        "google": get_stepback_insight_google,
        "mistral": get_stepback_insight_mistral,
    }

    func = stepback_functions.get(model_source)
    if func is None:
        return None, False

    return func(
        stepback=stepback,
        api_key=api_key,
        user_model=user_model,
        model_source=model_source,
        creativity=creativity
    )



# =============================================================================
# JSON Schema Functions
# =============================================================================

def build_json_schema(categories: list, include_additional_properties: bool = True) -> dict:
    """Build a JSON schema for the classification output.

    Args:
        categories: List of category names
        include_additional_properties: If True, includes additionalProperties: false
                                       (required by OpenAI strict mode, not supported by Google)
    """
    properties = {}
    for i, cat in enumerate(categories, 1):
        properties[str(i)] = {
            "type": "string",
            "enum": ["0", "1"],
            "description": cat,
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }

    if include_additional_properties:
        schema["additionalProperties"] = False

    return schema


def extract_json(reply: str) -> str:
    """Extract JSON from model reply."""
    if reply is None:
        return '{"1":"e"}'

    extracted = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
    if extracted:
        # Clean up the JSON string
        return extracted[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '')
    else:
        return '{"1":"e"}'


def validate_classification_json(json_str: str, num_categories: int) -> tuple[bool, dict | None]:
    """
    Validate that a JSON string contains valid classification output.

    Args:
        json_str: The JSON string to validate
        num_categories: Expected number of categories

    Returns:
        tuple: (is_valid, parsed_dict or None)
    """
    try:
        parsed = json.loads(json_str)

        if not isinstance(parsed, dict):
            return False, None

        # Check that all expected keys are present and values are "0" or "1"
        for i in range(1, num_categories + 1):
            key = str(i)
            if key not in parsed:
                return False, None
            val = str(parsed[key]).strip()
            if val not in ("0", "1"):
                return False, None

        # Normalize values to strings
        normalized = {str(i): str(parsed[str(i)]).strip() for i in range(1, num_categories + 1)}
        return True, normalized

    except (json.JSONDecodeError, KeyError, TypeError):
        return False, None


def ollama_two_step_classify(
    client,
    response_text: str,
    categories: list,
    categories_str: str,
    survey_question: str = "",
    creativity: float = None,
    max_retries: int = 5,
) -> tuple[str, str | None]:
    """
    Two-step classification for Ollama models.

    Step 1: Classify the response (natural language output OK)
    Step 2: Convert classification to strict JSON format

    This approach is more reliable for local models that struggle with
    simultaneous reasoning and JSON formatting.

    Args:
        client: UnifiedLLMClient instance
        response_text: The survey response to classify
        categories: List of category names
        categories_str: Pre-formatted category string
        survey_question: Optional context
        creativity: Temperature setting
        max_retries: Number of retry attempts for JSON validation

    Returns:
        tuple: (json_string, error_message or None)
    """
    num_categories = len(categories)
    survey_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # ==========================================================================
    # Step 1: Classification (natural language - focus on accuracy)
    # ==========================================================================
    step1_messages = [
        {
            "role": "system",
            "content": "You are an expert at categorizing survey responses. Focus on accurate classification."
        },
        {
            "role": "user",
            "content": f"""{survey_context}

Analyze this survey response and determine which categories apply:

Response: "{response_text}"

Categories:
{categories_str}

For each category, explain briefly whether it applies (YES) or not (NO) to this response.
Format your answer as:
1. [Category name]: YES/NO - [brief reason]
2. [Category name]: YES/NO - [brief reason]
...and so on for all categories."""
        }
    ]

    step1_reply, step1_error = client.complete(
        messages=step1_messages,
        json_schema=None,  # No JSON requirement for step 1
        creativity=creativity,
    )

    if step1_error:
        return '{"1":"e"}', f"Step 1 failed: {step1_error}"

    # ==========================================================================
    # Step 2: JSON Formatting with validation and retry
    # ==========================================================================
    example_json = json.dumps({str(i): "0" for i in range(1, num_categories + 1)})

    for attempt in range(max_retries):
        step2_messages = [
            {
                "role": "system",
                "content": "You convert classification results to JSON. Output ONLY valid JSON, nothing else."
            },
            {
                "role": "user",
                "content": f"""Convert this classification to JSON format.

Classification results:
{step1_reply}

Rules:
- Output ONLY a JSON object, no other text
- Use category numbers as keys (1, 2, 3, etc.)
- Use "1" if the category was marked YES, "0" if NO
- Include ALL {num_categories} categories

Example format:
{example_json}

Your JSON output:"""
            }
        ]

        step2_reply, step2_error = client.complete(
            messages=step2_messages,
            json_schema=None,  # Ollama doesn't support strict schema anyway
            creativity=0.1,  # Low temperature for formatting task
        )

        if step2_error:
            if attempt < max_retries - 1:
                continue
            return '{"1":"e"}', f"Step 2 failed: {step2_error}"

        # Extract and validate JSON
        extracted = extract_json(step2_reply)
        is_valid, normalized = validate_classification_json(extracted, num_categories)

        if is_valid:
            return json.dumps(normalized), None

        # If invalid, try again with more explicit instructions
        if attempt < max_retries - 1:
            step1_reply = f"""Previous attempt produced invalid JSON.

Original classification:
{step1_reply}

Please be more careful to output EXACTLY {num_categories} categories numbered 1 through {num_categories}."""

    # All retries exhausted - try to salvage what we can
    extracted = extract_json(step2_reply) if step2_reply else '{"1":"e"}'
    return extracted, f"JSON validation failed after {max_retries} attempts"


# =============================================================================
# Category Exploration Functions
# =============================================================================

def explore_corpus(
    survey_question,
    survey_input,
    api_key: str = None,
    research_question=None,
    specificity="broad",
    categories_per_chunk=10,
    divisions=5,
    model: str = "gpt-4o",
    provider: str = "auto",
    creativity=None,
    filename="corpus_exploration.csv",
    focus: str = None,
):
    """
    Extract categories from survey corpus using LLM.

    Uses raw HTTP requests via UnifiedLLMClient - supports all providers.

    Args:
        survey_question: The survey question being analyzed
        survey_input: Series or list of survey responses
        api_key: API key for the LLM provider
        research_question: Optional research context
        specificity: "broad" or "specific" categories
        categories_per_chunk: Number of categories to extract per chunk
        divisions: Number of chunks to process
        model: Model name (e.g., "gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash")
        provider: Provider name or "auto" to detect from model name
        creativity: Temperature setting
        filename: Output CSV filename (None to skip saving)
        focus: Optional focus instruction for category extraction (e.g., "decisions to move",
               "emotional responses", "financial considerations"). When provided, the model
               will prioritize extracting categories related to this focus.

    Returns:
        DataFrame with extracted categories and counts
    """
    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    print(f"Exploring categories for question: '{survey_question}'")
    print(f"Using provider: {provider}, model: {model}")
    if focus:
        print(f"Focus: {focus}")
    print(f"          {categories_per_chunk * divisions} unique categories to be extracted.")
    print()

    # Input normalization
    if not isinstance(survey_input, pd.Series):
        survey_input = pd.Series(survey_input)
    survey_input = survey_input.dropna()

    n = len(survey_input)
    if n == 0:
        raise ValueError("survey_input is empty after dropping NA.")

    # Auto-adjust divisions for small datasets
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    chunk_size = int(round(max(1, n / divisions), 0))

    if chunk_size < (categories_per_chunk / 2):
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    # Initialize unified client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build system message
    if research_question:
        system_content = (
            f"You are a helpful assistant that extracts categories from survey responses. "
            f"The specific task is to identify {specificity} categories of responses to a survey question. "
            f"The research question is: {research_question}"
        )
    else:
        system_content = "You are a helpful assistant that extracts categories from survey responses."

    # Sample chunks
    random_chunks = []
    for i in range(divisions):
        chunk = survey_input.sample(n=chunk_size).tolist()
        random_chunks.append(chunk)

    responses = []
    responses_list = []

    for i in tqdm(range(divisions), desc="Processing chunks"):
        survey_participant_chunks = '; '.join(str(x) for x in random_chunks[i])
        focus_text = f" Focus specifically on {focus}." if focus else ""
        prompt = (
            f'Identify {categories_per_chunk} {specificity} categories of responses to the question "{survey_question}" '
            f"in the following list of responses.{focus_text} Responses are each separated by a semicolon. "
            f"Responses are contained within triple backticks here: ```{survey_participant_chunks}``` "
            f"Number your categories from 1 through {categories_per_chunk} and be concise with the category labels and provide no description of the categories."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

        reply, error = client.complete(
            messages=messages,
            creativity=creativity,
            force_json=False,  # Text response, not JSON
        )

        if error:
            if "context_length_exceeded" in str(error) or "maximum context length" in str(error):
                raise ValueError(
                    f"Token limit exceeded for model {model}. "
                    f"Try increasing the 'divisions' parameter to create smaller chunks."
                )
            else:
                print(f"API error on chunk {i+1}: {error}")
                reply = ""

        responses.append(reply)

        # Extract just the text as a list
        items = []
        for line in (reply or "").split('\n'):
            if '. ' in line:
                try:
                    items.append(line.split('. ', 1)[1])
                except IndexError:
                    pass

        responses_list.append(items)

    flat_list = [item.lower() for sublist in responses_list for item in sublist]

    if not flat_list:
        raise ValueError("No categories were extracted from the model responses.")

    df = pd.DataFrame(flat_list, columns=['Category'])
    counts = pd.Series(flat_list).value_counts()
    df['counts'] = df['Category'].map(counts)
    df = df.sort_values(by='counts', ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset='Category', keep='first').reset_index(drop=True)

    if filename is not None:
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    return df


def explore_common_categories(
    survey_input,
    api_key: str = None,
    survey_question: str = "",
    max_categories: int = 12,
    categories_per_chunk: int = 10,
    divisions: int = 5,
    model: str = "gpt-4o",
    provider: str = "auto",
    creativity: float = None,
    specificity: str = "broad",
    research_question: str = None,
    filename: str = None,
    iterations: int = 5,
    random_state: int = None,
    focus: str = None,
    progress_callback: callable = None,
    return_raw: bool = False,
    # Legacy parameter names for backward compatibility
    user_model: str = None,
    model_source: str = None,
):
    """
    Extract and rank common categories from survey corpus.

    Uses raw HTTP requests via UnifiedLLMClient - supports all providers.

    Args:
        survey_input: Series or list of survey responses
        api_key: API key for the LLM provider
        survey_question: The survey question being analyzed
        max_categories: Maximum number of top categories to return
        categories_per_chunk: Number of categories to extract per chunk
        divisions: Number of chunks to process per iteration
        model: Model name (e.g., "gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash")
        provider: Provider name or "auto" to detect from model name
        creativity: Temperature setting
        specificity: "broad" or "specific" categories
        research_question: Optional research context
        filename: Output CSV filename (None to skip saving)
        iterations: Number of passes over the data
        random_state: Random seed for reproducibility
        focus: Optional focus instruction for category extraction (e.g., "decisions to move",
               "emotional responses", "financial considerations"). When provided, the model
               will prioritize extracting categories related to this focus.
        progress_callback: Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).

    Returns:
        dict with 'counts_df', 'top_categories', and 'raw_top_text'
    """
    import re
    import numpy as np

    # Handle legacy parameter names
    if user_model is not None:
        model = user_model
    if model_source is not None:
        provider = model_source

    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    # Input normalization
    if not isinstance(survey_input, pd.Series):
        survey_input = pd.Series(survey_input)
    survey_input = survey_input.dropna().astype("string")
    n = len(survey_input)
    if n == 0:
        raise ValueError("survey_input is empty after dropping NA.")

    # Auto-adjust divisions for small datasets
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    # Chunk sizing
    chunk_size = int(round(max(1, n / divisions), 0))
    if chunk_size < (categories_per_chunk / 2):
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    print(f"Exploring categories for question: '{survey_question}'")
    print(f"Using provider: {provider}, model: {model}")
    if focus:
        print(f"Focus: {focus}")
    print(f"          {categories_per_chunk * divisions * iterations} total category extractions across {iterations} iterations.")
    print(f"          Top {max_categories} categories will be identified.\n")

    # RNG for reproducible re-sampling across passes
    rng = np.random.default_rng(random_state)

    # Initialize unified client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build system message
    if research_question:
        system_content = (
            f"You are a helpful assistant that extracts categories from survey responses. "
            f"The specific task is to identify {specificity} categories of responses to a survey question. "
            f"The research question is: {research_question}"
        )
    else:
        system_content = "You are a helpful assistant that extracts categories from survey responses."

    def make_prompt(responses_blob: str) -> str:
        focus_text = f" Focus specifically on {focus}." if focus else ""
        return (
            f'Identify {categories_per_chunk} {specificity} categories of responses to the question "{survey_question}" '
            f"in the following list of responses.{focus_text} Responses are separated by semicolons. "
            f"Responses are within triple backticks: ```{responses_blob}``` "
            f"Number your categories from 1 through {categories_per_chunk} and provide concise labels only (no descriptions)."
        )

    # Parse numbered list
    line_pat = re.compile(r"^\s*\d+\s*[\.\)\-]\s*(.+)$")

    all_items = []

    # Calculate total steps for progress tracking: (iterations * divisions) + 1 for final merge
    total_steps = (iterations * divisions) + 1
    current_step = 0

    for pass_idx in range(iterations):
        random_chunks = []
        for _ in range(divisions):
            seed = int(rng.integers(0, 2**32 - 1))
            chunk = survey_input.sample(n=chunk_size, random_state=seed).tolist()
            random_chunks.append(chunk)

        for i in tqdm(range(divisions), desc=f"Processing chunks (pass {pass_idx+1}/{iterations})"):
            survey_participant_chunks = "; ".join(str(x) for x in random_chunks[i])
            prompt = make_prompt(survey_participant_chunks)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            reply, error = client.complete(
                messages=messages,
                creativity=creativity,
                force_json=False,  # Text response, not JSON
            )

            if error:
                raise RuntimeError(
                    f"Model call failed on pass {pass_idx+1}, chunk {i+1}: {error}"
                )

            items = []
            for raw_line in (reply or "").splitlines():
                m = line_pat.match(raw_line.strip())
                if m:
                    items.append(m.group(1).strip())
            if not items:
                for raw_line in (reply or "").splitlines():
                    s = raw_line.strip()
                    if s:
                        items.append(s)

            all_items.extend(items)

            # Progress callback
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"Pass {pass_idx+1}/{iterations}, chunk {i+1}/{divisions}")

    # Early return for raw output (used by explore())
    if return_raw:
        return all_items

    # Normalize and count
    def normalize_category(cat):
        terms = sorted([t.strip().lower() for t in str(cat).split("/")])
        return "/".join(terms)

    flat_list = [str(x).strip() for x in all_items if str(x).strip()]
    if not flat_list:
        raise ValueError("No categories were extracted from the model responses.")

    df = pd.DataFrame(flat_list, columns=["Category"])
    df["normalized"] = df["Category"].map(normalize_category)

    result = (
        df.groupby("normalized")
          .agg(Category=("Category", lambda x: x.value_counts().index[0]),
               counts=("Category", "size"))
          .sort_values("counts", ascending=False)
          .reset_index(drop=True)
    )

    # Second-pass semantic merge prompt
    seed_list = result["Category"].head(max_categories * 3).tolist()

    second_prompt = f"""
You are a data analyst reviewing categorized survey data.

Task: From the provided categories, identify and return the top {max_categories} CONCEPTUALLY UNIQUE categories.

Critical Instructions:
1) Exact duplicates are already removed.
2) Merge SEMANTIC duplicates (same concept, different wording). Examples:
   - "closer to work" = "commute/proximity to work"
   - "breakup/household conflict" = "relationship problems"
3) When merging:
   - Combine frequencies mentally
   - Keep the most frequent OR clearest label
   - Each concept appears ONLY ONCE
4) Keep category names {specificity}.
5) Return ONLY a numbered list of {max_categories} categories. No extra text.

Pre-processed Categories (sorted by frequency, top sample):
{seed_list}

Output:
1. category
2. category
...
{max_categories}. category
""".strip()

    # Second pass call
    reply2, error2 = client.complete(
        messages=[{"role": "user", "content": second_prompt}],
        creativity=creativity,
        force_json=False,  # Text response
    )

    # Final progress callback for the merge step
    if progress_callback:
        progress_callback(total_steps, total_steps, "Merging categories")

    if error2:
        print(f"Warning: Second pass failed: {error2}")
        top_categories_text = ""
    else:
        top_categories_text = reply2 or ""

    final = []
    for line in top_categories_text.splitlines():
        m = line_pat.match(line.strip())
        if m:
            final.append(m.group(1).strip())
    if not final:
        final = [l.strip("-* ").strip() for l in top_categories_text.splitlines() if l.strip()]

    # Fallback to counts_df if second pass failed
    if not final:
        final = result["Category"].head(max_categories).tolist()

    print("\nTop categories:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(final[:max_categories])))

    if filename:
        result.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

    return {
        "counts_df": result,
        "top_categories": final[:max_categories],
        "raw_top_text": top_categories_text
    }


# =============================================================================
# Main Classification Function
# =============================================================================

def multi_class(
    survey_input,
    categories,
    api_key: str = None,
    model: str = "gpt-4o",
    provider: str = "auto",
    survey_question: str = "",
    example1: str = None,
    example2: str = None,
    example3: str = None,
    example4: str = None,
    example5: str = None,
    example6: str = None,
    creativity: float = None,
    safety: bool = False,
    chain_of_verification: bool = False,
    chain_of_thought: bool = True,
    step_back_prompt: bool = False,
    context_prompt: bool = False,
    thinking_budget: int = 0,
    max_categories: int = 12,
    categories_per_chunk: int = 10,
    divisions: int = 10,
    research_question: str = None,
    use_json_schema: bool = True,
    filename: str = None,
    save_directory: str = None,
    auto_download: bool = False,
):
    """
    Multi-class text classification using a unified HTTP-based approach.

    This function uses raw HTTP requests for all providers, eliminating SDK dependencies.
    Supports multiple prompting strategies including chain-of-thought, chain-of-verification,
    step-back prompting, and context prompting.

    Args:
        survey_input: List or Series of text responses to classify
        categories: List of category names, or "auto" to auto-detect categories
        api_key: API key for the LLM provider (not required for Ollama)
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.5-flash",
               or any Ollama model like "llama3.2", "mistral", "phi3")
        provider: Provider name or "auto" to detect from model name.
                  For local models, use provider="ollama"
        survey_question: Optional context about what question was asked
        example1-6: Optional few-shot examples for classification
        creativity: Temperature setting (None for provider default)
        safety: If True, saves results incrementally during processing
        chain_of_verification: If True, uses 4-step CoVe prompting for verification
        chain_of_thought: If True, uses step-by-step reasoning in prompt
        step_back_prompt: If True, first asks about underlying factors before classifying
        context_prompt: If True, adds expert context prefix to prompts
        thinking_budget: Token budget for Google's extended thinking (0 to disable)
        max_categories: Maximum categories when using auto-detection
        categories_per_chunk: Categories per chunk for auto-detection
        divisions: Number of divisions for auto-detection
        research_question: Research context for auto-detection
        use_json_schema: Whether to use strict JSON schema (vs just json_object mode)
        filename: Optional CSV filename to save results
        save_directory: Optional directory for safety saves
        auto_download: If True, automatically download missing Ollama models

    Returns:
        DataFrame with classification results

    Example with Ollama (local):
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            model="llama3.2",
            provider="ollama",
        )

    Example with cloud provider:
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            api_key="your-api-key",
            model="gpt-4o",
        )

    Example with chain-of-verification:
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            api_key="your-api-key",
            model="gpt-4o",
            chain_of_verification=True,
            survey_question="Why did you move?",
        )

    .. deprecated::
        Use :func:`catllm.classify` instead. This function will be removed in a future version.
    """
    warnings.warn(
        "multi_class() is deprecated and will be removed in a future version. "
        "Use catllm.classify() instead, which supports single and multi-model classification.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key requirement
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    # Handle categories="auto" - auto-detect categories from the data
    if categories == "auto":
        if survey_question == "":
            raise TypeError("survey_question is required when using categories='auto'. Please provide the survey question you are analyzing.")

        categories = explore_common_categories(
            survey_question=survey_question,
            survey_input=survey_input,
            research_question=research_question,
            api_key=api_key,
            model_source=provider,
            user_model=model,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions
        )

    # Build examples text for few-shot prompting
    examples = [example1, example2, example3, example4, example5, example6]
    examples_text = "\n".join(
        f"Example {i}: {ex}" for i, ex in enumerate(examples, 1) if ex is not None
    )

    # Survey question context
    survey_question_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # Step-back insight initialization
    stepback_insight = None
    step_back_added = False
    if step_back_prompt:
        if survey_question == "":
            raise TypeError("survey_question is required when using step_back_prompt. Please provide the survey question you are analyzing.")

        stepback_question = f'What are the underlying factors or dimensions that explain how people typically answer "{survey_question}"?'
        stepback_insight, step_back_added = _get_stepback_insight(
            provider, stepback_question, api_key, model, creativity
        )

    # Ollama-specific checks
    if provider == "ollama":
        if not check_ollama_running():
            raise ConnectionError(
                "\n" + "="*60 + "\n"
                "  OLLAMA NOT RUNNING\n"
                "="*60 + "\n\n"
                "Ollama must be running to use local models.\n\n"
                "To start Ollama:\n"
                "  macOS:   Open the Ollama app, or run 'ollama serve'\n"
                "  Linux:   Run 'ollama serve' in terminal\n"
                "  Windows: Open the Ollama app\n\n"
                "Don't have Ollama installed?\n"
                "  Download from: https://ollama.ai/download\n\n"
                "After starting Ollama, run your code again.\n"
                + "="*60
            )

        # Check system resources before proceeding
        resources = check_system_resources(model)

        # Check if model needs to be downloaded
        model_installed = check_ollama_model(model)

        if not model_installed:
            if not pull_ollama_model(model, auto_confirm=auto_download):
                raise RuntimeError(
                    f"Model '{model}' not available. "
                    f"To download manually: ollama pull {model}"
                )
        else:
            # Model is installed - still check if it can run
            if resources["warnings"] or not resources["can_run"]:
                print(f"\n{'='*60}")
                print(f"  Model '{model}' - System Resource Check")
                print(f"{'='*60}")
                size_estimate = get_ollama_model_size_estimate(model)
                print(f"  Model size:      {size_estimate}")
                if resources["details"].get("estimated_ram"):
                    print(f"  RAM required:    ~{resources['details']['estimated_ram']}")
                if resources["details"].get("total_ram"):
                    print(f"  System RAM:      {resources['details']['total_ram']}")

                if resources["warnings"]:
                    print(f"\n  {'!'*50}")
                    for warning in resources["warnings"]:
                        print(f"  Warning: {warning}")
                    print(f"  {'!'*50}")

                if not resources["can_run"]:
                    print(f"\n  Warning: Model may not run well on this system.")
                    print(f"  Consider a smaller variant (e.g., '{model}:1b' or '{model}:3b').")
                    print(f"{'='*60}")

                    if not auto_download:
                        try:
                            response = input(f"\n  Continue anyway? [y/N]: ").strip().lower()
                            if response not in ['y', 'yes']:
                                raise RuntimeError(
                                    f"Model '{model}' may be too large for this system. "
                                    f"Try a smaller variant like '{model}:3b' or '{model}:1b'."
                                )
                        except (EOFError, KeyboardInterrupt):
                            raise RuntimeError("Operation cancelled by user.")

                print()

    print(f"Using provider: {provider}, model: {model}")

    # Initialize client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build category string and schema
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    # Build JSON schema - Google doesn't support additionalProperties
    if use_json_schema:
        include_additional = (provider != "google")
        json_schema = build_json_schema(categories, include_additional_properties=include_additional)
    else:
        json_schema = None

    # Print categories
    print(f"\nCategories to classify ({len(categories)} total):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print()

    # Build prompt template
    def build_prompt(response_text: str) -> tuple:
        """Build the classification prompt for a single response.

        Returns:
            tuple: (messages list, user_prompt string for CoVe)
        """
        if chain_of_thought:
            user_prompt = f"""{survey_question_context}

Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

{examples_text}

Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""
        else:
            user_prompt = f"""{survey_question_context}
Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}
{examples_text}
Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""

        # Add context prompt prefix if enabled
        if context_prompt:
            context = """You are an expert researcher in survey data categorization.
Apply multi-label classification and base decisions on explicit and implicit meanings.
When uncertain, prioritize precision over recall.

"""
            user_prompt = context + user_prompt

        # Build messages list
        messages = []

        # Add step-back insight if available
        if step_back_prompt and step_back_added and stepback_insight:
            messages.append({"role": "user", "content": stepback_question})
            messages.append({"role": "assistant", "content": stepback_insight})

        messages.append({"role": "user", "content": user_prompt})

        return messages, user_prompt

    # Build chain of verification prompts
    def build_cove_prompts(prompt: str, response_text: str) -> tuple:
        """Build chain of verification prompts."""
        step2_prompt = f"""You provided this initial categorization:
<<INITIAL_REPLY>>

Original task: {prompt}

Generate a focused list of 3-5 verification questions to fact-check your categorization. Each question should:
- Be concise and specific (one sentence)
- Address a distinct aspect of the categorization
- Be answerable independently

Focus on verifying:
- Whether each category assignment is accurate
- Whether the categories match the criteria in the original task
- Whether there are any logical inconsistencies

Provide only the verification questions as a numbered list."""

        step3_prompt = f"""Answer the following verification question based on the survey response provided.

Survey response: {response_text}

Verification question: <<QUESTION>>

Provide a brief, direct answer (1-2 sentences maximum).

Answer:"""

        step4_prompt = f"""Original task: {prompt}
Initial categorization:
<<INITIAL_REPLY>>
Verification questions and answers:
<<VERIFICATION_QA>>
If no categories are present, assign "0" to all categories.
Provide the final corrected categorization in the same JSON format:"""

        return step2_prompt, step3_prompt, step4_prompt

    def remove_numbering(line: str) -> str:
        """Remove numbering/bullets from a line for CoVe question parsing."""
        line = line.strip()
        if line.startswith('- '):
            return line[2:].strip()
        if line.startswith('• '):
            return line[2:].strip()
        if line and line[0].isdigit():
            i = 0
            while i < len(line) and line[i].isdigit():
                i += 1
            if i < len(line) and line[i] in '.)':
                return line[i+1:].strip()
        return line

    def run_chain_of_verification(initial_reply: str, step2_prompt: str, step3_prompt: str, step4_prompt: str) -> str:
        """Run chain of verification using the unified client."""
        # Step 2: Generate verification questions (text response, not JSON)
        step2_filled = step2_prompt.replace("<<INITIAL_REPLY>>", initial_reply)
        questions_reply, err = client.complete(
            messages=[{"role": "user", "content": step2_filled}],
            creativity=creativity,
            force_json=False,  # Text response
        )
        if err:
            return initial_reply  # Fall back to initial reply on error

        # Parse questions
        questions = [remove_numbering(line) for line in questions_reply.strip().split('\n') if line.strip()]

        # Step 3: Answer each verification question (text responses)
        qa_pairs = []
        for question in questions[:5]:  # Limit to 5 questions
            step3_filled = step3_prompt.replace("<<QUESTION>>", question)
            answer_reply, err = client.complete(
                messages=[{"role": "user", "content": step3_filled}],
                creativity=creativity,
                force_json=False,  # Text response
            )
            if not err:
                qa_pairs.append(f"Q: {question}\nA: {answer_reply.strip()}")

        verification_qa = "\n\n".join(qa_pairs)

        # Step 4: Final corrected categorization (JSON response)
        step4_filled = step4_prompt.replace("<<INITIAL_REPLY>>", initial_reply).replace("<<VERIFICATION_QA>>", verification_qa)
        final_reply, err = client.complete(
            messages=[{"role": "user", "content": step4_filled}],
            json_schema=json_schema,
            creativity=creativity,
        )

        if err:
            return initial_reply
        return final_reply

    # Process each response
    results = []
    extracted_jsons = []

    # Use two-step approach for Ollama (more reliable JSON output)
    use_two_step = (provider == "ollama")

    if use_two_step:
        print("Using two-step classification for Ollama (classify -> format JSON)")

    for idx, response in enumerate(tqdm(survey_input, desc="Classifying responses")):
        if pd.isna(response):
            results.append(("Skipped NaN", "Skipped NaN input"))
            extracted_jsons.append('{"1":"e"}')
            continue

        if use_two_step:
            json_result, error = ollama_two_step_classify(
                client=client,
                response_text=response,
                categories=categories,
                categories_str=categories_str,
                survey_question=survey_question,
                creativity=creativity,
                max_retries=5,
            )

            if error:
                results.append((json_result, error))
            else:
                results.append((json_result, None))
            extracted_jsons.append(json_result)

        else:
            messages, user_prompt = build_prompt(response)
            reply, error = client.complete(
                messages=messages,
                json_schema=json_schema,
                creativity=creativity,
                thinking_budget=thinking_budget if provider == "google" else None,
            )

            if error:
                results.append((None, error))
                extracted_jsons.append('{"1":"e"}')
            else:
                # Apply chain of verification if enabled
                if chain_of_verification and reply:
                    step2, step3, step4 = build_cove_prompts(user_prompt, response)
                    reply = run_chain_of_verification(reply, step2, step3, step4)

                results.append((reply, None))
                extracted_jsons.append(extract_json(reply))

        # Safety incremental save
        if safety:
            if filename is None:
                raise TypeError("filename is required when using safety=True. Please provide a filename to save to.")

            # Build partial DataFrame and save
            normalized_partial = []
            for json_str in extracted_jsons:
                try:
                    parsed = json.loads(json_str)
                    normalized_partial.append(pd.json_normalize(parsed))
                except json.JSONDecodeError:
                    normalized_partial.append(pd.DataFrame({"1": ["e"]}))

            if normalized_partial:
                normalized_df = pd.concat(normalized_partial, ignore_index=True)
                partial_df = pd.DataFrame({
                    'survey_input': pd.Series(survey_input[:len(results)]).reset_index(drop=True),
                    'model_response': [r[0] for r in results],
                    'error': [r[1] for r in results],
                    'json': pd.Series(extracted_jsons).reset_index(drop=True),
                })
                partial_df = pd.concat([partial_df, normalized_df], axis=1)
                partial_df = partial_df.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

                save_path = filename
                if save_directory:
                    import os
                    os.makedirs(save_directory, exist_ok=True)
                    save_path = os.path.join(save_directory, filename)
                partial_df.to_csv(save_path, index=False)

    # Build output DataFrame
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))

    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    # Create main DataFrame
    df = pd.DataFrame({
        'survey_input': pd.Series(survey_input).reset_index(drop=True),
        'model_response': [r[0] for r in results],
        'error': [r[1] for r in results],
        'json': pd.Series(extracted_jsons).reset_index(drop=True),
    })

    df = pd.concat([df, normalized_data], axis=1)

    # Rename category columns
    df = df.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

    # Process category columns
    cat_cols = [col for col in df.columns if col.startswith('category_')]

    # Identify invalid rows
    has_invalid = df[cat_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce').isna() & col.notna()
    ).any(axis=1)

    df['processing_status'] = (~has_invalid).map({True: 'success', False: 'error'})
    df.loc[has_invalid, cat_cols] = pd.NA

    # Convert to numeric
    for col in cat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN with 0 for valid rows
    df.loc[~has_invalid, cat_cols] = df.loc[~has_invalid, cat_cols].fillna(0)

    # Convert to Int64
    df[cat_cols] = df[cat_cols].astype('Int64')

    # Create categories_id
    df['categories_id'] = df[cat_cols].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1
    )

    if filename:
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

    return df


# Note: For the legacy implementation with chain_of_verification, step_back_prompt,
# and other advanced features, see text_functions_old.py
