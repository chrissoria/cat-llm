#extract categories from corpus
def explore_corpus(
    survey_question, 
    survey_input,
    api_key,
    research_question=None,
    specificity="broad",
    cat_num=10,
    divisions=5,
    user_model="gpt-4o-2024-11-20",
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

    print(f"Exploring class for question: '{survey_question}'.\n          {cat_num * divisions} unique categories to be extracted.")
    print()

    model_source = model_source.lower() # eliminating case sensitivity 

    chunk_size = round(max(1, len(survey_input) / divisions),0)
    chunk_size = int(chunk_size)

    if chunk_size < (cat_num/2):
        raise ValueError(f"Cannot extract {cat_num} {specificity} categories from chunks of only {chunk_size} responses. \n" 
                    f"Choose one solution: \n"
                    f"(1) Reduce 'divisions' parameter (currently {divisions}) to create larger chunks, or \n"
                    f"(2) Reduce 'cat_num' parameter (currently {cat_num}) to extract fewer categories per chunk.")

    random_chunks = []
    for i in range(divisions):
        chunk = survey_input.sample(n=chunk_size).tolist()
        random_chunks.append(chunk)
    
    responses = []
    responses_list = []
    
    for i in tqdm(range(divisions), desc="Processing chunks"):
        survey_participant_chunks = '; '.join(random_chunks[i])
        prompt = f"""Identify {cat_num} {specificity} categories of responses to the question "{survey_question}" in the following list of responses. \
Responses are each separated by a semicolon. \
Responses are contained within triple backticks here: ```{survey_participant_chunks}``` \
Number your categories from 1 through {cat_num} and be concise with the category labels and provide no description of the categories."""
        
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
    survey_question, 
    survey_input,
    api_key,
    top_n=10,
    cat_num=10,
    divisions=5,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    filename=None,
    model_source="OpenAI"
):
    import os
    import pandas as pd
    import random
    from openai import OpenAI
    from openai import OpenAI, BadRequestError
    from tqdm import tqdm

    print(f"Exploring class for question: '{survey_question}'.\n          {cat_num * divisions} unique categories to be extracted and {top_n} to be identified as the most common.")
    print()

    model_source = model_source.lower() # eliminating case sensitivity 

    chunk_size = round(max(1, len(survey_input) / divisions),0)
    chunk_size = int(chunk_size)

    if chunk_size < (cat_num/2):
        raise ValueError(f"Cannot extract {cat_num} categories from chunks of only {chunk_size} responses. \n" 
                    f"Choose one solution: \n"
                    f"(1) Reduce 'divisions' parameter (currently {divisions}) to create larger chunks, or \n"
                    f"(2) Reduce 'cat_num' parameter (currently {cat_num}) to extract fewer categories per chunk.")

    random_chunks = []
    for i in range(divisions):
        chunk = survey_input.sample(n=chunk_size).tolist()
        random_chunks.append(chunk)
    
    responses = []
    responses_list = []
    
    for i in tqdm(range(divisions), desc="Processing chunks"):
        survey_participant_chunks = '; '.join(random_chunks[i])
        prompt = f"""Identify {cat_num} {specificity} categories of responses to the question "{survey_question}" in the following list of responses. \
Responses are each separated by a semicolon. \
Responses are contained within triple backticks here: ```{survey_participant_chunks}``` \
Number your categories from 1 through {cat_num} and be concise with the category labels and provide no description of the categories."""
        
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

    second_prompt = f"""From this list of categories, extract the top {top_n} most common categories. \
The categories are contained within triple backticks here: ```{df['Category'].tolist()}``` \
Return the top {top_n} categories as a numbered list sorted from the most to least common and keep the categories {specificity}, with no additional text or explanation."""
        
    if model_source == "openai":
        client = OpenAI(api_key=api_key)
        response_obj = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': second_prompt}],
            temperature=creativity
        )
    top_categories = response_obj.choices[0].message.content
    print(top_categories)

    top_categories_final = []
    for line in top_categories.split('\n'):
        if '. ' in line:
            try:
                top_categories_final.append(line.split('. ', 1)[1])
            except IndexError:
                pass

    return top_categories_final

#multi-class text classification
def multi_class(
    survey_question, 
    survey_input,
    categories,
    api_key,
    user_model="gpt-4o",
    creativity=None,
    safety=False,
    to_csv=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm

    model_source = model_source.lower() # eliminating case sensitivity 
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    # ensure number of categories is what user wants
    print("\nThe categories you entered:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    
    link1 = []
    extracted_jsons = []

    for idx, response in enumerate(tqdm(survey_input, desc="Categorizing responses")):
        reply = None  

        if pd.isna(response): 
            link1.append("Skipped NaN input")
            default_json = example_JSON 
            extracted_jsons.append(default_json)
            #print(f"Skipped NaN input.")
        else:
            prompt = f"""A respondent was asked: {survey_question}. \
Categorize this survey response "{response}" into the following categories that apply: \
{categories_str} \
Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values."""
            #print(prompt)
            if model_source == ("openai"):
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                try:
                    response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                    reply = response_obj.choices[0].message.content
                    link1.append(reply)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            elif model_source == "perplexity":
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
                try:
                    response_obj = client.chat.completions.create(
                        model=user_model,
                        messages=[{'role': 'user', 'content': prompt}],
                        **({"temperature": creativity} if creativity is not None else {})
                    )
                    reply = response_obj.choices[0].message.content
                    link1.append(reply)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            elif model_source == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                try:
                    message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    **({"temperature": creativity} if creativity is not None else {}),
                    messages=[{"role": "user", "content": prompt}]
                )
                    reply = message.content[0].text  # Anthropic returns content as list
                    link1.append(reply)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
                    
            elif model_source == "google":
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                try:
                    headers = {
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                        }
                    
                    payload = {
                        "contents": [{
                            "parts": [{"text": prompt}]
                            }],
                            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
                            }
                    
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    result = response.json()

                    if "candidates" in result and result["candidates"]:
                        reply = result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        reply = "No response generated"

                    link1.append(reply)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            elif model_source == "mistral":
                from mistralai import Mistral
                client = Mistral(api_key=api_key)
                try:
                    response = client.chat.complete(
                    model=user_model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                    reply = response.choices[0].message.content
                    link1.append(reply)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
            else:
                raise ValueError("Unknown source! Choose from OpenAI, Anthropic, Perplexity, or Mistral")
            # in situation that no JSON is found
            if reply is not None:
                extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
                if extracted_json:
                    cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                    extracted_jsons.append(cleaned_json)
                    #print(cleaned_json)
                else:
                    error_message = """{"1":"e"}"""
                    extracted_jsons.append(error_message)
                    print(error_message)
            else:
                error_message = """{"1":"e"}"""
                extracted_jsons.append(error_message)
                #print(error_message)

        # --- Safety Save ---
        if safety:
            # Save progress so far
            temp_df = pd.DataFrame({
                'survey_response': survey_input[:idx+1],
                'link1': link1,
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
            # save to CSV
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
        'survey_input': (
            survey_input.reset_index(drop=True) if isinstance(survey_input, (pd.DataFrame, pd.Series)) 
            else pd.Series(survey_input)
        ),
        'model_response': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)

    if to_csv:
        if save_directory is None:
            save_directory = os.getcwd()
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)
    
    return categorized_data