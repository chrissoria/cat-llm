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


def _detect_model_source(user_model, model_source):
    """Auto-detect model source from model name if not explicitly provided."""
    model_source = model_source.lower()

    if model_source is None or model_source == "auto":
        user_model_lower = user_model.lower()

        if "gpt" in user_model_lower:
            return "openai"
        elif "claude" in user_model_lower:
            return "anthropic"
        elif "gemini" in user_model_lower or "gemma" in user_model_lower:
            return "google"
        elif "llama" in user_model_lower or "meta" in user_model_lower:
            return "huggingface"
        elif "mistral" in user_model_lower or "mixtral" in user_model_lower:
            return "mistral"
        elif "sonar" in user_model_lower or "pplx" in user_model_lower:
            return "perplexity"
        elif "deepseek" in user_model_lower or "qwen" in user_model_lower:
            return "huggingface"
        elif "grok" in user_model_lower:
            return "xai"
        else:
            raise ValueError(
                f"Could not auto-detect model source from '{user_model}'. "
                "Please specify model_source explicitly: OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral"
            )
    return model_source


def _get_stepback_insight(model_source, stepback, api_key, user_model, creativity):
    """Get step-back insight using the appropriate provider."""
    stepback_functions = {
        "openai": get_stepback_insight_openai,
        "perplexity": get_stepback_insight_openai,
        "huggingface": get_stepback_insight_openai,
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


#extract categories from corpus
def explore_corpus(
    survey_question, 
    survey_input,
    api_key,
    research_question=None,
    specificity="broad",
    categories_per_chunk=10,
    divisions=5,
    user_model="gpt-5",
    creativity=None,
    filename="corpus_exploration.csv",
    model_source="OpenAI"
):
    import os
    import pandas as pd
    import random
    from openai import OpenAI
    from openai import OpenAI, BadRequestError
    from tqdm import tqdm

    print(f"Exploring class for question: '{survey_question}'.\n          {categories_per_chunk * divisions} unique categories to be extracted.")
    print()

    model_source = model_source.lower() # eliminating case sensitivity

    n = len(survey_input)

    # Auto-adjust divisions for small datasets
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))  # At least 3 responses per chunk
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    chunk_size = round(max(1, n / divisions),0)
    chunk_size = int(chunk_size)

    if chunk_size < (categories_per_chunk/2):
        # Auto-reduce categories_per_chunk instead of erroring
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    random_chunks = []
    for i in range(divisions):
        chunk = survey_input.sample(n=chunk_size).tolist()
        random_chunks.append(chunk)
    
    responses = []
    responses_list = []
    
    for i in tqdm(range(divisions), desc="Processing chunks"):
        survey_participant_chunks = '; '.join(random_chunks[i])
        prompt = f"""Identify {categories_per_chunk} {specificity} categories of responses to the question "{survey_question}" in the following list of responses. \
Responses are each separated by a semicolon. \
Responses are contained within triple backticks here: ```{survey_participant_chunks}``` \
Number your categories from 1 through {categories_per_chunk} and be concise with the category labels and provide no description of the categories."""
        
        if model_source == "openai":
            client = OpenAI(api_key=api_key)
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[
                        {'role': 'system', 'content': f"""You are a helpful assistant that extracts categories from survey responses. \
                                                    The specific task is to identify {specificity} categories of responses to a survey question. \
                         The research question is: {research_question}""" if research_question else "You are a helpful assistant."},
                        {'role': 'user', 'content': prompt}
                    ],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response_obj.choices[0].message.content
                responses.append(reply)
            except BadRequestError as e:
                if "context_length_exceeded" in str(e) or "maximum context length" in str(e):
                    error_msg = (f"Token limit exceeded for model {user_model}. "
                        f"Try increasing the 'iterations' parameter to create smaller chunks.")
                    raise ValueError(error_msg)
                else:
                    print(f"OpenAI API error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            raise ValueError(f"Unsupported model_source: {model_source}")
        
        # Extract just the text as a list
        items = []
        for line in responses[i].split('\n'):
            if '. ' in line:
                try:
                    items.append(line.split('. ', 1)[1])
                except IndexError:
                    pass

        responses_list.append(items)

    flat_list = [item.lower() for sublist in responses_list for item in sublist]

    #convert flat_list to a df
    df = pd.DataFrame(flat_list, columns=['Category'])
    counts = pd.Series(flat_list).value_counts()  # Use original list before conversion
    df['counts'] = df['Category'].map(counts)
    df = df.sort_values(by='counts', ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset='Category', keep='first').reset_index(drop=True)

    if filename is not None:
        df.to_csv(filename, index=False)
    
    return df

#extract top categories from corpus
def explore_common_categories(
    survey_input,
    api_key,
    survey_question="",
    max_categories=12,
    categories_per_chunk=10,
    divisions=5,
    user_model="gpt-5",
    creativity=None,
    specificity="broad",
    research_question=None,
    filename=None,          # need to implement
    model_source="auto",
    iterations=5,
    random_state=None
):
    import re
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    model_source = _detect_model_source(user_model, model_source)

    # --- input normalization ---
    if not isinstance(survey_input, pd.Series):
        survey_input = pd.Series(survey_input)
    survey_input = survey_input.dropna().astype("string")
    n = len(survey_input)
    if n == 0:
        raise ValueError("survey_input is empty after dropping NA.")

    # --- auto-adjust divisions for small datasets ---
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))  # At least 3 responses per chunk
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    # --- chunk sizing ---
    chunk_size = int(round(max(1, n / divisions), 0))
    if chunk_size < (categories_per_chunk / 2):
        # Auto-reduce categories_per_chunk instead of erroring
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    print(
        f"Exploring class for question: '{survey_question}'.\n"
        f"          {categories_per_chunk * divisions} unique categories to be extracted and {max_categories} to be identified as the most common.\n"
    )

    # --- RNG for reproducible re-sampling across passes ---
    rng = np.random.default_rng(random_state)

    # main calls - initialize clients based on model source
    if model_source in ["openai", "perplexity", "huggingface", "xai"]:
        from openai import OpenAI
        # conditional base_url setting based on model source
        base_url = (
            "https://api.perplexity.ai" if model_source == "perplexity" 
            else "https://router.huggingface.co/v1" if model_source == "huggingface"
            else "https://api.x.ai/v1" if model_source == "xai"
            else None  # default for OpenAI
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif model_source == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    elif model_source == "google":
        import requests
        # Google uses REST API, no client initialization needed
        client = None
    elif model_source == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
    else:
        raise ValueError(f"Unsupported model_source: {model_source}")

    def make_prompt(responses_blob: str) -> str:
        return (
            f'Identify {categories_per_chunk} {specificity} categories of responses to the question \"{survey_question}\" '
            f"in the following list of responses. Responses are separated by semicolons. "
            f"Responses are within triple backticks: ```{responses_blob}``` "
            f"Number your categories from 1 through {categories_per_chunk} and provide concise labels only (no descriptions)."
        )

    def call_model(prompt: str) -> str:
        if model_source in ["openai", "perplexity", "huggingface", "xai"]:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}"
                        if research_question else
                        "You are a helpful assistant."
                    )
                },
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(
                model=user_model,
                messages=messages,
                **({"temperature": creativity} if creativity is not None else {})
            )
            return resp.choices[0].message.content

        elif model_source == "anthropic":
            sys_text = (
                f"You are a helpful assistant that extracts categories from survey responses. "
                f"The specific task is to identify {specificity} categories of responses to a survey question. "
                f"The research question is: {research_question}"
                if research_question else
                "You are a helpful assistant."
            )
            resp = client.messages.create(
                model=user_model,
                max_tokens=4096,
                system=sys_text,
                messages=[{"role": "user", "content": prompt}],
                **({"temperature": creativity} if creativity is not None else {})
            )
            return resp.content[0].text
        
        elif model_source == "google":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            # Build system-like content in the prompt
            if research_question:
                system_context = (f"You are a helpful assistant that extracts categories from survey responses. "
                                f"The specific task is to identify {specificity} categories of responses to a survey question. "
                                f"The research question is: {research_question}\n\n")
            else:
                system_context = "You are a helpful assistant.\n\n"
            
            full_prompt = system_context + prompt
            
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    **({"temperature": creativity} if creativity is not None else {})
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response generated"
        
        elif model_source == "mistral":
            # Build system prompt
            if research_question:
                system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                                f"The specific task is to identify {specificity} categories of responses to a survey question. "
                                f"The research question is: {research_question}")
            else:
                system_content = "You are a helpful assistant."
            
            resp = client.chat.complete(
                model=user_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                **({"temperature": creativity} if creativity is not None else {})
            )
            return resp.choices[0].message.content

    # --- parse numbered list like "1. Foo" / "1) Foo" / "1 - Foo" ---
    line_pat = re.compile(r"^\s*\d+\s*[\.\)\-]\s*(.+)$")

    all_items = []

    for pass_idx in range(iterations):
        # fresh chunks each pass, with reproducible per-chunk seeds
        random_chunks = []
        for _ in range(divisions):
            seed = int(rng.integers(0, 2**32 - 1))
            # pandas sample with random_state for reproducibility
            chunk = survey_input.sample(n=chunk_size, random_state=seed).tolist()
            random_chunks.append(chunk)

        for i in tqdm(range(divisions), desc=f"Processing chunks (pass {pass_idx+1}/{iterations})"):
            survey_participant_chunks = "; ".join(random_chunks[i])
            prompt = make_prompt(survey_participant_chunks)
            try:
                reply = call_model(prompt)
            except Exception as e:
                # common failure mode is context/token limit
                raise RuntimeError(
                    f"Model call failed on pass {pass_idx+1}, chunk {i+1}: {e}"
                ) from e

            # extract numbered items
            items = []
            for raw_line in (reply or "").splitlines():
                m = line_pat.match(raw_line.strip())
                if m:
                    items.append(m.group(1).strip())
            # fallback: if model returned bare lines without numbers
            if not items:
                for raw_line in (reply or "").splitlines():
                    s = raw_line.strip()
                    if s:
                        items.append(s)

            all_items.extend(items)

    # --- normalize and count ---
    def normalize_category(cat):
        # normalize case + slash-delimited variants (e.g., "friends/family" == "family / friends")
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

    # --- second-pass semantic merge prompt (max_categories selection) ---
    seed_list = result["Category"].head(max_categories * 3).tolist()

    second_prompt = f"""
You are a data analyst reviewing categorized survey data.

Task: From the provided categories, identify and return the top {max_categories} CONCEPTUALLY UNIQUE categories.

Critical Instructions:
1) Exact duplicates are already removed.
2) Merge SEMANTIC duplicates (same concept, different wording). Examples:
   - "closer to work" ≈ "commute/proximity to work"
   - "breakup/household conflict" ≈ "relationship problems"
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

    if model_source in ["openai", "perplexity", "huggingface", "xai"]:
        resp2 = client.chat.completions.create(
            model=user_model,
            messages=[{"role": "user", "content": second_prompt}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        top_categories_text = resp2.choices[0].message.content
    else:
        # Anthropic
        resp2 = client.messages.create(
            model=user_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": second_prompt}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        top_categories_text = resp2.content[0].text

    # parse the final numbered list
    final = []
    for line in top_categories_text.splitlines():
        m = line_pat.match(line.strip())
        if m:
            final.append(m.group(1).strip())
    if not final:
        # permissive fallback if the model returns plain bullets
        final = [l.strip("-*• ").strip() for l in top_categories_text.splitlines() if l.strip()]

    print("\nTop categories:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(final[:max_categories])))

    return {
        "counts_df": result,            # normalized frequency table
        "top_categories": final[:max_categories],
        "raw_top_text": top_categories_text
    }


#multi-class text classification
# GOAL: enable self-consistency
def multi_class(
    survey_input,
    categories,
    api_key,
    user_model="gpt-5",
    survey_question = "",
    example1 = None,
    example2 = None,
    example3 = None,
    example4 = None,
    example5 = None,
    example6 = None,
    creativity = None,
    safety = False,
    chain_of_verification = False,
    chain_of_thought = True,
    step_back_prompt = False,
    context_prompt = False,
    thinking_budget = 0,
    max_categories = 12,
    categories_per_chunk = 10,
    divisions = 10,
    research_question = None,
    filename = None,
    save_directory = None,
    model_source = "auto"
):
    import os
    import json
    import pandas as pd
    import regex
    import time
    from tqdm import tqdm

    #used in chain of verification 
    def remove_numbering(line):
        line = line.strip()
    
        # Handle bullet points
        if line.startswith('- '):
            return line[2:].strip()
        if line.startswith('• '):
            return line[2:].strip()
    
        # Handle numbered lists "1.", "10.", etc.
        if line and line[0].isdigit():
            # Find where the number ends
            i = 0
            while i < len(line) and line[i].isdigit():
                i += 1
        
            # Check if followed by '.' or ')'
            if i < len(line) and line[i] in '.':
                return line[i+1:].strip()
            elif i < len(line) and line[i] in ')':
                return line[i+1:].strip()
    
        return line

    model_source = _detect_model_source(user_model, model_source)

    if categories == "auto":
        if survey_question == "": # step back requires the survey question to function well
            raise TypeError("survey_question is required when using step_back_prompt. Please provide the survey question you are analyzing.")

        categories = explore_common_categories(
            survey_question=survey_question,
            survey_input=survey_input,
            research_question=research_question,
            api_key=api_key,
            model_source=model_source,
            user_model=user_model,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions
        )
        
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    categories_per_chunk = len(categories)
    category_dict = {str(i+1): "0" for i in range(categories_per_chunk)}
    example_JSON = json.dumps(category_dict, indent=4)

    # for ensuring JSON format for anthropic api
    properties = {
            str(i+1): {
                "type": "string",
                "description": cat,
                "enum": ["0", "1"]
                } 
            for i, cat in enumerate(categories)
        }
    
    print(f"\nThe categories you entered to be coded by {model_source} {user_model}:")

    if categories != "auto":
    # ensure number of categories is what user wants

        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")
    
    link1 = []
    extracted_jsons = []

    #handling example inputs
    examples = [example1, example2, example3, example4, example5, example6]
    examples_text = "\n".join(
    f"Example {i}: {ex}" for i, ex in enumerate(examples, 1) if ex is not None
)
    # allowing users to contextualize the survey question
    if survey_question != None:
        survey_question_context = f"A respondent was asked: {survey_question}."
    else:
        survey_question_context = ""

    # step back insight initialization
    if step_back_prompt:
        if survey_question == "":
            raise TypeError("survey_question is required when using step_back_prompt. Please provide the survey question you are analyzing.")

        stepback = f"""What are the underlying factors or dimensions that explain how people typically answer "{survey_question}"?"""
        stepback_insight, step_back_added = _get_stepback_insight(
            model_source, stepback, api_key, user_model, creativity
        )
    else:
        stepback_insight = None
        step_back_added = False

    def _build_prompt(response_text):
        """Build the classification prompt for a single response."""
        if chain_of_thought:
            prompt = f"""{survey_question_context}

            Categorize this survey response "{response_text}" into the following categories that apply:
            {categories_str}

            Let's think step by step:
            1. First, identify the main themes mentioned in the response
            2. Then, match each theme to the relevant categories
            3. Finally, assign 1 to matching categories and 0 to non-matching categories

            {examples_text}

            Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values."""
        else:
            prompt = f"""{survey_question_context} \
            Categorize this survey response "{response_text}" into the following categories that apply: \
            {categories_str}
            {examples_text}
            Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values."""

        if context_prompt:
            context = """You are an expert researcher in survey data categorization.
            Apply multi-label classification and base decisions on explicit and implicit meanings.
            When uncertain, prioritize precision over recall."""
            prompt = context + prompt

        return prompt

    def _build_cove_prompts(prompt, response_text):
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

    def _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt):
        """Handle OpenAI-compatible API calls (OpenAI, Perplexity, HuggingFace, xAI)."""
        from openai import OpenAI, BadRequestError

        base_url = (
            "https://api.perplexity.ai" if model_source == "perplexity"
            else "https://router.huggingface.co/v1" if model_source == "huggingface"
            else "https://api.x.ai/v1" if model_source == "xai"
            else None
        )
        client = OpenAI(api_key=api_key, base_url=base_url)

        max_retries = 8
        delay = 2

        for attempt in range(max_retries):
            try:
                messages = [
                    *([{'role': 'user', 'content': stepback}] if step_back_prompt and step_back_added else []),
                    *([{'role': 'assistant', 'content': stepback_insight}] if step_back_added else []),
                    {'role': 'user', 'content': prompt}
                ]

                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **({"temperature": creativity} if creativity is not None else {})
                )

                reply = response_obj.choices[0].message.content

                if chain_of_verification:
                    reply = chain_of_verification_openai(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering
                    )

                return reply, None

            except BadRequestError as e:
                if "json_validate_failed" in str(e) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ JSON validation failed. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e

            except Exception as e:
                if ("500" in str(e) or "504" in str(e)) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

        return """{"1":"e"}""", "Max retries exceeded"

    def _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt):
        """Handle Anthropic API calls."""
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        tools = [{
            "name": "return_categories",
            "description": "Return categorization results as 0 (not present) or 1 (present) for each category",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys())
            }
        }]

        try:
            response_obj = client.messages.create(
                model=user_model,
                max_tokens=4096,
                tools=tools,
                tool_choice={"type": "tool", "name": "return_categories"},
                messages=[{'role': 'user', 'content': prompt}],
                **({"temperature": creativity} if creativity is not None else {})
            )
            json_reply = response_obj.content[0].input
            reply = json.dumps(json_reply)

            if chain_of_verification:
                reply = chain_of_verification_anthropic(
                    initial_reply=reply,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    client=client,
                    user_model=user_model,
                    creativity=creativity,
                    remove_numbering=remove_numbering
                )

            return reply, None

        except anthropic.NotFoundError as e:
            raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_google(prompt, step2_prompt, step3_prompt, step4_prompt):
        """Handle Google API calls."""
        import requests

        def make_google_request(url, headers, payload, max_retries=8):
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code
                    retryable_errors = [429, 500, 502, 503, 504]

                    if status_code in retryable_errors and attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt) if status_code == 429 else 2 * (2 ** attempt)
                        error_type = "Rate limited" if status_code == 429 else f"Server error {status_code}"
                        print(f"⚠️ {error_type}. Attempt {attempt + 1}/{max_retries}")
                        print(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({"temperature": creativity} if creativity is not None else {}),
                **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget is not None else {})
            }
        }

        try:
            result = make_google_request(url, headers, payload)

            if "candidates" in result and result["candidates"]:
                reply = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                reply = "No response generated"

            if chain_of_verification:
                reply = chain_of_verification_google(
                    initial_reply=reply,
                    prompt=prompt,
                    step2_prompt=step2_prompt,
                    step3_prompt=step3_prompt,
                    step4_prompt=step4_prompt,
                    url=url,
                    headers=headers,
                    creativity=creativity,
                    remove_numbering=remove_numbering,
                    make_google_request=make_google_request
                )

            return reply, None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"❌ Model '{user_model}' not found. Please check the model name and try again.") from e
            elif e.response.status_code in [401, 403]:
                raise ValueError(f"❌ Authentication failed. Please check your Google API key.") from e
            else:
                print(f"HTTP error occurred: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return """{"1":"e"}""", f"Error processing input: {e}"

    def _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt):
        """Handle Mistral API calls."""
        from mistralai import Mistral
        from mistralai.models import SDKError, MistralError

        client = Mistral(api_key=api_key)
        max_retries = 8
        delay = 2

        for attempt in range(max_retries):
            try:
                messages = [
                    *([{'role': 'user', 'content': stepback}] if step_back_prompt and step_back_added else []),
                    *([{'role': 'assistant', 'content': stepback_insight}] if step_back_added else []),
                    {'role': 'user', 'content': prompt}
                ]

                response = client.chat.complete(
                    model=user_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = response.choices[0].message.content

                if chain_of_verification:
                    reply = chain_of_verification_mistral(
                        initial_reply=reply,
                        step2_prompt=step2_prompt,
                        step3_prompt=step3_prompt,
                        step4_prompt=step4_prompt,
                        client=client,
                        user_model=user_model,
                        creativity=creativity,
                        remove_numbering=remove_numbering
                    )

                return reply, None

            except SDKError as e:
                error_str = str(e).lower()

                if "invalid_model" in error_str or "invalid model" in error_str:
                    raise ValueError(f"❌ Model '{user_model}' not found.") from e
                elif "401" in str(e) or "unauthorized" in error_str:
                    raise ValueError(f"❌ Authentication failed. Please check your Mistral API key.") from e

                retryable_errors = ["500", "502", "503", "504"]
                if any(code in str(e) for code in retryable_errors) and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ Server error detected. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

            except MistralError as e:
                if hasattr(e, 'status_code') and e.status_code in [500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️ Server error {e.status_code}. Attempt {attempt + 1}/{max_retries}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")
                    return """{"1":"e"}""", f"Error processing input: {e}"

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return """{"1":"e"}""", f"Error processing input: {e}"

        return """{"1":"e"}""", "Max retries exceeded"

    def _process_single_response(response_text):
        """Process a single survey response and return (reply, error_msg)."""
        prompt = _build_prompt(response_text)

        if chain_of_verification:
            step2_prompt, step3_prompt, step4_prompt = _build_cove_prompts(prompt, response_text)
        else:
            step2_prompt = step3_prompt = step4_prompt = None

        if model_source in ["openai", "perplexity", "huggingface", "xai"]:
            return _call_openai_compatible(prompt, step2_prompt, step3_prompt, step4_prompt)
        elif model_source == "anthropic":
            return _call_anthropic(prompt, step2_prompt, step3_prompt, step4_prompt)
        elif model_source == "google":
            return _call_google(prompt, step2_prompt, step3_prompt, step4_prompt)
        elif model_source == "mistral":
            return _call_mistral(prompt, step2_prompt, step3_prompt, step4_prompt)
        else:
            raise ValueError("Unknown source! Choose from OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral")

    def _extract_json(reply):
        """Extract JSON from model reply."""
        if reply is None:
            return """{"1":"e"}"""

        extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
        if extracted_json:
            return extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
        else:
            print("""{"1":"e"}""")
            return """{"1":"e"}"""

    # Main processing loop
    for idx, response in enumerate(tqdm(survey_input, desc="Categorizing responses")):
        if pd.isna(response):
            link1.append("Skipped NaN input")
            extracted_jsons.append(example_JSON)
            continue

        reply, error_msg = _process_single_response(response)

        if error_msg:
            link1.append(error_msg)
        else:
            link1.append(reply)

        extracted_jsons.append(_extract_json(reply))

        # --- Safety Save ---
        if safety:
            if filename == None: # step back requires the survey question to function well
                raise TypeError("filename is required when using safety. Please provide the filename you want to save to csv.")
    
            # Normalize processed jsons so far
            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)
    
            # Save progress so far
            temp_df = pd.DataFrame({
                'survey_input': (
                    survey_input[:idx+1].reset_index(drop=True) 
                    if isinstance(survey_input, (pd.DataFrame, pd.Series)) 
                    else pd.Series(survey_input[:idx+1])
                ),
                'model_response': pd.Series(link1[:idx+1]).reset_index(drop=True),
                'json': pd.Series(extracted_jsons[:idx+1]).reset_index(drop=True)
            })
            temp_df = pd.concat([temp_df, normalized_data], axis=1)
    
            # save to CSV
            temp_df.to_csv(filename, index=False)

    # --- Final DataFrame ---
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed_obj = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed_obj))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
    normalized_data = pd.concat(normalized_data_list, ignore_index=True)
    categorized_data = pd.DataFrame({
        'survey_input': (
            survey_input.reset_index(drop=True) if isinstance(survey_input, (pd.DataFrame, pd.Series)) 
            else pd.Series(survey_input)
        ),
        'model_response': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    categorized_data = categorized_data.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

    # renaming columns for easier readability
    cat_cols = [col for col in categorized_data.columns if col.startswith('category_')]

    # Step 1: Identify rows with invalid strings (like "e")
    has_invalid_strings = categorized_data[cat_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce').isna() & col.notna()
    ).any(axis=1)

    # Step 2: set processing status before modifying the data, if categories extracted then success
    categorized_data['processing_status'] = (~has_invalid_strings).map({True: 'success', False: 'error'})

    # Step 3: set invalid rows to NA
    categorized_data.loc[has_invalid_strings, cat_cols] = pd.NA

    # Step 4: converting to numeric
    for col in cat_cols:
        categorized_data[col] = pd.to_numeric(categorized_data[col], errors='coerce')
    
    # Step 4.5: Fill NaN with 0 ONLY for valid rows (valid but sparse JSONs)
    categorized_data.loc[~has_invalid_strings, cat_cols] = (
        categorized_data.loc[~has_invalid_strings, cat_cols].fillna(0)
    )

    # Step 5: Convert to Int64
    categorized_data[cat_cols] = categorized_data[cat_cols].astype('Int64')


    # Step 6: Create categories_id (comma-separated binary values for each category)
    categorized_data['categories_id'] = categorized_data[cat_cols].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1
    )

    if filename:
        categorized_data.to_csv(filename, index=False)
    
    return categorized_data