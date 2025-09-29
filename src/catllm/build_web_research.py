#build dataset classification
def build_web_research_dataset(
    search_question, 
    search_input,
    api_key,
    answer_format = "concise",
    additional_instructions = "",
    categories = ['Answer','URL'],
    user_model="claude-sonnet-4-20250514",
    creativity=None,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="Anthropic",
    time_delay=15
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import time
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    print(categories_str)
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    # ensure number of categories is what user wants
    #print("\nThe information to be extracted:")
    #for i, cat in enumerate(categories, 1):
        #print(f"{i}. {cat}")
    
    link1 = []
    extracted_jsons = []

    for idx, item in enumerate(tqdm(search_input, desc="Building dataset")):
        if idx > 0:  # Skip delay for first item only
            time.sleep(time_delay)
        reply = None  

        if pd.isna(item): 
            link1.append("Skipped NaN input")
            default_json = example_JSON 
            extracted_jsons.append(default_json)
            #print(f"Skipped NaN input.")
        else:
            prompt = f"""<role>You are a research assistant specializing in finding current, factual information.</role>

            <task>Find information about {item}'s {search_question}</task>

            <rules>
            - Search for the most current and authoritative information available
            - Provide your answer as {answer_format}
            - Prioritize official sources when possible
            - If information is not found, state "Information not found"
            - Include exactly one source URL where you found the information
            - Do not include any explanatory text or commentary beyond the JSON
                {additional_instructions}
            </rules>

            <format>
            Return your response as valid JSON with this exact structure:
            {{
            "answer": "Your factual answer or 'Information not found'",
            "url": "Source URL or 'No source available'"
        }}
        </format>"""
            #print(prompt)
            if model_source == "Anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                try:
                    message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {}),
                    tools=[{
                    "type": "web_search_20250305", 
                    "name": "web_search"
                    }]
                )
                    reply = " ".join(
                        block.text
                        for block in message.content
                        if getattr(block, "type", "") == "text"
                    ).strip()
                    link1.append(reply)
                    
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            elif model_source == "Google":
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                try:
                    headers = {
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "tools": [{"google_search": {}}],
                        **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
                    }
        
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
        
                    # extract reply from Google's response structure
                    if "candidates" in result and result["candidates"]:
                        reply = result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        reply = "No response generated"
            
                    link1.append(reply)
        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            else:
                raise ValueError("Unknown source! Currently this function only supports 'Anthropic' or 'Google' as model_source.")
            # in situation that no JSON is found
            if reply is not None:
                extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
                if extracted_json:
                    raw_json = extracted_json[0].strip()  # Only strip leading/trailing whitespace
                    try:
                        # Parse to validate JSON structure
                        parsed_obj = json.loads(raw_json)
                        # Re-serialize for consistent formatting (optional)
                        cleaned_json = json.dumps(parsed_obj)
                        extracted_jsons.append(cleaned_json)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        # Fallback to raw extraction if parsing fails
                        extracted_jsons.append(raw_json)
                else:
                    # Use consistent schema for errors
                    error_message = json.dumps({"answer": "e", "url": "e"})
                    extracted_jsons.append(error_message)
                    print(error_message)
            else:
                # Handle None reply case
                error_message = json.dumps({"answer": "e", "url": "e"})
                extracted_jsons.append(error_message)
                #print(error_message)

        # --- Safety Save ---
        if safety:
            # Save progress so far
            temp_df = pd.DataFrame({
                'survey_response': search_input[:idx+1],
                'model_response': link1,
                'json': extracted_jsons
            })
            # Normalize processed jsons so far
            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"1": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)
            temp_df = pd.concat([temp_df, normalized_data], axis=1)
            # Save to CSV
            if save_directory is None:
                save_directory = os.getcwd()
            temp_df.to_csv(os.path.join(save_directory, filename), index=False)

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
        'survey_response': (
            search_input.reset_index(drop=True) if isinstance(search_input, (pd.DataFrame, pd.Series)) 
            else pd.Series(search_input)
        ),
        'link1': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    
    return categorized_data