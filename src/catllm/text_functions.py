#extract categories from corpus
def explore_corpus(
    survey_question, 
    survey_input,
    api_key,
    research_question=None,
    specificity="broad",
    cat_num=10,
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
    user_model="gpt-5",
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
# what this function does:
# does context prompting, giving the model a background on the task at hand and the user's survey question
# system prompting, overall context and purpose for the language model
# role prompting, assings a spacific identity to the model
# also enables few shot prompting, allowing the user to input a few examples
# provides POSITIVE INSTRUCTIONS reather than limitations/restrictions
# GOAL: enable step-back prompting
# GOAL 2: enable self-consistency
def multi_class(
    survey_input,
    categories,
    api_key,
    user_model="gpt-5",
    user_prompt = None,
    survey_question = "",
    example1 = None,
    example2 = None,
    example3 = None,
    example4 = None,
    example5 = None,
    example6 = None,
    creativity=None,
    safety=False,
    to_csv=False,
    chain_of_verification=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="auto"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm

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

    model_source = model_source.lower() # eliminating case sensitivity 

    # auto-detect model source if not provided
    if model_source is None or model_source == "auto":
        user_model_lower = user_model.lower()
    
        if "gpt" in user_model_lower:
            model_source = "openai"
        elif "claude" in user_model_lower:
            model_source = "anthropic"
        elif "gemini" in user_model_lower or "gemma" in user_model_lower:
            model_source = "google"
        elif "llama" in user_model_lower or "meta" in user_model_lower:
            model_source = "huggingface"
        elif "mistral" in user_model_lower or "mixtral" in user_model_lower:
            model_source = "mistral"
        elif "sonar" in user_model_lower or "pplx" in user_model_lower:
            model_source = "perplexity"
        elif "deepseek"  in user_model_lower or "qwen" in user_model_lower:
            model_source = "huggingface"
        else:
            raise ValueError(f"❌ Could not auto-detect model source from '{user_model}'. Please specify model_source explicitly: OpenAI, Anthropic, Perplexity, Google, Huggingface, or Mistral")
    else:
        model_source = model_source.lower()
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    # ensure number of categories is what user wants
    print(f"\nThe categories you entered to be coded by {model_source} {user_model}:")
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

    for idx, response in enumerate(tqdm(survey_input, desc="Categorizing responses")):
        reply = None  

        if pd.isna(response): 
            link1.append("Skipped NaN input")
            default_json = example_JSON 
            extracted_jsons.append(default_json)
            #print(f"Skipped NaN input.")
        else:

            prompt = f"""{survey_question_context} \
            Categorize this survey response "{response}" into the following categories that apply: \
            {categories_str}
            {examples_text}
            Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values."""

            if chain_of_verification:
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

                Survey response: {response}

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


            if model_source in ["openai", "perplexity", "huggingface"]:
                from openai import OpenAI
                from openai import OpenAI, BadRequestError, AuthenticationError
                # conditional base_url setting based on model source
                base_url = (
                    "https://api.perplexity.ai" if model_source == "perplexity" 
                    else "https://router.huggingface.co/v1" if model_source == "huggingface"
                    else None  # default
                )
    
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                try:
                    response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                    )
        
                    reply = response_obj.choices[0].message.content
                    
                    if chain_of_verification:
                        try:
                            initial_reply = reply
                            #STEP 2: Generate verification questions
                            step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

                            verification_response = client.chat.completions.create(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step2_filled}],
                                **({"temperature": creativity} if creativity is not None else {})
                                )
                            
                            verification_questions = verification_response.choices[0].message.content
                            #STEP 3: Answer verification questions
                            questions_list = [
                                remove_numbering(q) 
                                for q in verification_questions.split('\n') 
                                if q.strip()
                                ]
                            verification_qa = []

                            #prompting each question individually
                            for question in questions_list:

                                step3_filled = step3_prompt.replace('<<QUESTION>>', question)

                                answer_response = client.chat.completions.create(
                                    model=user_model,
                                    messages=[{'role': 'user', 'content': step3_filled}],
                                    **({"temperature": creativity} if creativity is not None else {})
                                    )

                                answer = answer_response.choices[0].message.content
                                verification_qa.append(f"Q: {question}\nA: {answer}")

                            #STEP 4: Final corrected categorization
                            verification_qa_text = "\n\n".join(verification_qa)
                            
                            step4_filled = (step4_prompt
                            .replace('<<INITIAL_REPLY>>', initial_reply)
                            .replace('<<VERIFICATION_QA>>', verification_qa_text))

                            print(f"Final prompt:\n{step4_filled}\n")

                            final_response = client.chat.completions.create(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step4_filled}],
                                **({"temperature": creativity} if creativity is not None else {})
                            )

                            reply = final_response.choices[0].message.content
    
                            print("Chain of verification completed. Final response generated.\n")
                            link1.append(reply)

                        except Exception as e:
                            print(f"ERROR in Chain of Verification: {str(e)}")
                            print("Falling back to initial response.\n")
                            link1.append(reply)
                    else:
                        #if chain of verification is not enabled, just append initial reply
                        link1.append(reply)
                    
                except BadRequestError as e:
                    # Model doesn't exist - halt immediately
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            elif model_source == "anthropic":

                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                
                try:
                    response_obj = client.messages.create(
                    model=user_model,
                    max_tokens=4096,
                    messages=[{'role': 'user', 'content': prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                    )
        
                    reply = response_obj.content[0].text
                    
                    if chain_of_verification:
                        try:
                            initial_reply = reply
                            #STEP 2: Generate verification questions
                            step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

                            verification_response = client.messages.create(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step2_filled}],
                                max_tokens=4096,
                                **({"temperature": creativity} if creativity is not None else {})
                                )
                            
                            verification_questions = verification_response.content[0].text
                            #STEP 3: Answer verification questions
                            questions_list = [
                                remove_numbering(q) 
                                for q in verification_questions.split('\n') 
                                if q.strip()
                                ]
                            print(f"Verification questions:\n{questions_list}\n")
                            verification_qa = []

                            #prompting each question individually
                            for question in questions_list:

                                step3_filled = step3_prompt.replace('<<QUESTION>>', question)

                                answer_response = client.messages.create(
                                    model=user_model,
                                    messages=[{'role': 'user', 'content': step3_filled}],
                                    max_tokens=4096,
                                    **({"temperature": creativity} if creativity is not None else {})
                                    )

                                answer = answer_response.content[0].text
                                verification_qa.append(f"Q: {question}\nA: {answer}")

                            #STEP 4: Final corrected categorization
                            verification_qa_text = "\n\n".join(verification_qa)
                            
                            step4_filled = (step4_prompt
                            .replace('<<INITIAL_REPLY>>', initial_reply)
                            .replace('<<VERIFICATION_QA>>', verification_qa_text))

                            print(f"Final prompt:\n{step4_filled}\n")

                            final_response = client.messages.create(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step4_filled}],
                                max_tokens=4096,
                                **({"temperature": creativity} if creativity is not None else {})
                            )

                            reply = final_response.content[0].text
    
                            print("Chain of verification completed. Final response generated.\n")
                            link1.append(reply)

                        except Exception as e:
                            print(f"ERROR in Chain of Verification: {str(e)}")
                            print("Falling back to initial response.\n")
                            link1.append(reply)
                    else:
                        #if chain of verification is not enabled, just append initial reply
                        link1.append(reply)
                    
                except anthropic.NotFoundError as e:
                    # Model doesn't exist - halt immediately
                    raise ValueError(f"❌ Model '{user_model}' on {model_source} not found. Please check the model name and try again.") from e
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
                    
            elif model_source == "google":
                import requests

                def make_google_request(url, headers, payload, max_retries=3):
                    """Make Google API request with exponential backoff on 429 errors"""
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(url, headers=headers, json=payload)
                            response.raise_for_status()
                            return response.json()
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 429 and attempt < max_retries - 1:
                                wait_time = 10 * (2 ** attempt)
                                print(f"⚠️ Rate limited. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                raise

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
                    
                    result = make_google_request(url, headers, payload)

                    if "candidates" in result and result["candidates"]:
                        reply = result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        reply = "No response generated"

                    if chain_of_verification:
                        try:
                            import time
                            initial_reply = reply
                            # STEP 2: Generate verification questions
                            step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
                            
                            payload_step2 = {
                                "contents": [{
                                    "parts": [{"text": step2_filled}]
                                    }],
                                    **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
                                    }
                
                            result_step2 = make_google_request(url, headers, payload_step2)
                
                            verification_questions = result_step2["candidates"][0]["content"]["parts"][0]["text"]
                
                            # STEP 3: Answer verification questions
                            questions_list = [
                                remove_numbering(q) 
                                for q in verification_questions.split('\n') 
                                if q.strip()
                            ]
                            verification_qa = []
                            
                            for question in questions_list:
                                time.sleep(2) # temporary rate limit handling
                                step3_filled = step3_prompt.replace('<<QUESTION>>', question)
                                payload_step3 = {
                                    "contents": [{
                                        "parts": [{"text": step3_filled}]
                                        }],
                                        **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
                                }
                    
                                result_step3 = make_google_request(url, headers, payload_step3)
                    
                                answer = result_step3["candidates"][0]["content"]["parts"][0]["text"]
                                verification_qa.append(f"Q: {question}\nA: {answer}")
                
                            # STEP 4: Final corrected categorization
                            verification_qa_text = "\n\n".join(verification_qa)
                
                            step4_filled = (step4_prompt
                            .replace('<<PROMPT>>', prompt)
                            .replace('<<INITIAL_REPLY>>', initial_reply)
                            .replace('<<VERIFICATION_QA>>', verification_qa_text))
                
                            payload_step4 = {
                                "contents": [{
                                    "parts": [{"text": step4_filled}]
                                    }],
                            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
                            }
                
                            result_step4 = make_google_request(url, headers, payload_step4)
                
                            reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]
                            print("Chain of verification completed. Final response generated.\n")

                            link1.append(reply)
                
                        except Exception as e:
                            print(f"ERROR in Chain of Verification: {str(e)}")
                            print("Falling back to initial response.\n")
        
                    else:
                        # if chain of verification is not enabled, just append initial reply
                        link1.append(reply)
                
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        raise ValueError(f"❌ Model '{user_model}' not found. Please check the model name and try again.") from e
                    elif e.response.status_code == 401 or e.response.status_code == 403:
                        raise ValueError(f"❌ Authentication failed. Please check your Google API key.") from e
                    else:
                        print(f"HTTP error occurred: {e}")
                        link1.append(f"Error processing input: {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            elif model_source == "mistral":
                from mistralai import Mistral
                from mistralai.models import SDKError

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
                    
                    if chain_of_verification:
                        try:
                            initial_reply = reply
                            #STEP 2: Generate verification questions
                            step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

                            verification_response = client.chat.complete(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step2_filled}],
                                **({"temperature": creativity} if creativity is not None else {})
                                )
                            
                            verification_questions = verification_response.choices[0].message.content
                            #STEP 3: Answer verification questions
                            questions_list = [
                                remove_numbering(q) 
                                for q in verification_questions.split('\n') 
                                if q.strip()
                                ]
                            verification_qa = []

                            #prompting each question individually
                            for question in questions_list:

                                step3_filled = step3_prompt.replace('<<QUESTION>>', question)

                                answer_response = client.chat.complete(
                                    model=user_model,
                                    messages=[{'role': 'user', 'content': step3_filled}],
                                    **({"temperature": creativity} if creativity is not None else {})
                                    )

                                answer = answer_response.choices[0].message.content
                                verification_qa.append(f"Q: {question}\nA: {answer}")

                            #STEP 4: Final corrected categorization
                            verification_qa_text = "\n\n".join(verification_qa)
                            
                            step4_filled = (step4_prompt
                            .replace('<<INITIAL_REPLY>>', initial_reply)
                            .replace('<<VERIFICATION_QA>>', verification_qa_text))

                            final_response = client.chat.complete(
                                model=user_model,
                                messages=[{'role': 'user', 'content': step4_filled}],
                                **({"temperature": creativity} if creativity is not None else {})
                            )

                            reply = final_response.choices[0].message.content
    
                            link1.append(reply)
                        except Exception as e:
                            print(f"ERROR in Chain of Verification: {str(e)}")
                            print("Falling back to initial response.\n")
                    else:
                        #if chain of verification is not enabled, just append initial reply
                        link1.append(reply)

                except SDKError as e:
                    error_str = str(e).lower()
                    if "invalid_model" in error_str or "invalid model" in error_str:
                        raise ValueError(f"❌ Model '{user_model}' not found.") from e
                    elif "401" in str(e) or "unauthorized" in str(e).lower():
                        raise ValueError(f"❌ Authentication failed. Please check your Mistral API key.") from e
                    else:
                        print(f"An error occurred: {e}")
                        link1.append(f"Error processing input: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    link1.append(f"Error processing input: {e}")

            else:
                raise ValueError("Unknown source! Choose from OpenAI, Anthropic, Perplexity, Google, Huggingface, or Mistral")
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