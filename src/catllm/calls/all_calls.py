# openai stepback prompt

def get_stepback_insight_openai(
    stepback,
    api_key,
    user_model,
    model_source="openai",
    creativity=None
):
    from openai import OpenAI
    # Conditional base_url setting based on model source
    base_url = (
        "https://api.perplexity.ai" if model_source == "perplexity" 
        else "https://router.huggingface.co/v1" if model_source == "huggingface"
        else "https://api.x.ai/v1" if model_source == "xai"
        else None
    )
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        stepback_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.choices[0].message.content
        
        return stepback_insight, True
        
    except Exception as e:
        print(f"An error occurred during step-back prompting: {e}")
        return None, False


# claude stepback prompt

def get_stepback_insight_anthropic(
    stepback,
    api_key,
    user_model,
    model_source="anthropic",
    creativity=None
):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        stepback_response = client.messages.create(
            model=user_model,
            max_tokens=4096,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.content[0].text
        
        return stepback_insight, True
        
    except Exception as e:
        print(f"An error occurred during step-back prompting: {e}")
        return None, False
    
# google stepback prompt

def get_stepback_insight_google(
    stepback,
    api_key,
    user_model,
    model_source="google",
    creativity=None
):
    
    import requests
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": stepback}],

            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad status codes
        
        result = response.json()
        stepback_insight = result['candidates'][0]['content']['parts'][0]['text']
        
        return stepback_insight, True
        
    except Exception as e:
        print(f"An error occurred during step-back prompting: {e}")
        return None, False

# mistral stepback prompt

def get_stepback_insight_mistral(
    stepback,
    api_key,
    user_model,
    model_source="mistral",
    creativity=None
):
    
    from mistralai import Mistral
    
    client = Mistral(api_key=api_key)
    
    try:
        stepback_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.choices[0].message.content
        
        return stepback_insight, True
        
    except Exception as e:
        print(f"An error occurred during step-back prompting: {e}")
        return None, False
    
# openai chain of verification calls

def chain_of_verification_openai(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        
        verification_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verification_questions = verification_response.choices[0].message.content
        
        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q) 
            for q in verification_questions.split('\n') 
            if q.strip()
        ]
        verification_qa = []
        
        # Prompting each question individually
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            
            answer_response = client.chat.completions.create(
                model=user_model,
                messages=[{'role': 'user', 'content': step3_filled}],
                **({"temperature": creativity} if creativity is not None else {})
            )
            
            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")
        
        # STEP 4: Final corrected categorization
        verification_qa_text = "\n\n".join(verification_qa)
        
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        
        print(f"Final prompt:\n{step4_filled}\n")
        
        final_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            response_format={"type": "json_object"},
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verified_reply = final_response.choices[0].message.content
        print("Chain of verification completed. Final response generated.\n")
        
        return verified_reply
        
    except Exception as e:
        print(f"ERROR in Chain of Verification: {str(e)}")
        print("Falling back to initial response.\n")
        return initial_reply
    
# anthropic chain of verification calls

def chain_of_verification_anthropic(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process for Anthropic Claude.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        
        verification_response = client.messages.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            max_tokens=4096,
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verification_questions = verification_response.content[0].text
        
        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q) 
            for q in verification_questions.split('\n') 
            if q.strip()
        ]
        print(f"Verification questions:\n{questions_list}\n")
        verification_qa = []
        
        # Prompting each question individually
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
        
        # STEP 4: Final corrected categorization
        verification_qa_text = "\n\n".join(verification_qa)
        
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        
        print(f"Final prompt:\n{step4_filled}\n")

        tools = [{
            "name": "return_categories",
            "description": "Return categorization results as 0 (not present) or 1 (present) for each category",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys())  # All categories required
            }
        }]
        
        final_response = client.messages.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            max_tokens=4096,
            tools=tools,
            tool_choice={"type": "tool", "name": "return_categories"},
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        result_dict = final_response.content[0].input

        verified_reply = json.dumps(result_dict)
        print("Chain of verification completed. Final response generated.\n")
        
        return verified_reply
        
    except Exception as e:
        print(f"ERROR in Chain of Verification: {str(e)}")
        print("Falling back to initial response.\n")
        return initial_reply

# google chain of verification calls
def chain_of_verification_google(
    initial_reply,
    prompt,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    url,
    headers,
    creativity,
    remove_numbering,
    make_google_request
):
    import time
    """
    Execute Chain of Verification (CoVe) process for Google Gemini.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        
        payload_step2 = {
            "contents": [{
                "parts": [{"text": step2_filled}]
            }],
            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {}),
            **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget is not None else {})
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
            time.sleep(2)  # temporary rate limit handling
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            
            payload_step3 = {
                "contents": [{
                    "parts": [{"text": step3_filled}]
                }],
                **({"generationConfig": {"temperature": creativity}} if creativity is not None else {}),
                **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget is not None else {})
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
            "generationConfig": {
                                "responseMimeType": "application/json",
                                **({"temperature": creativity} if creativity is not None else {}),
                                **({"thinkingConfig": {"thinkingBudget": thinking_budget}} if thinking_budget is not None else {})
            }
        }
        
        result_step4 = make_google_request(url, headers, payload_step4)
        verified_reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]
        
        print("Chain of verification completed. Final response generated.\n")
        return verified_reply
        
    except Exception as e:
        print(f"ERROR in Chain of Verification: {str(e)}")
        print("Falling back to initial response.\n")
        return initial_reply
    
# mistral chain of verification calls

def chain_of_verification_mistral(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process for Mistral AI.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        
        verification_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verification_questions = verification_response.choices[0].message.content
        
        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q) 
            for q in verification_questions.split('\n') 
            if q.strip()
        ]
        verification_qa = []
        
        # Prompting each question individually
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            
            answer_response = client.chat.complete(
                model=user_model,
                messages=[{'role': 'user', 'content': step3_filled}],
                **({"temperature": creativity} if creativity is not None else {})
            )
            
            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")
        
        # STEP 4: Final corrected categorization
        verification_qa_text = "\n\n".join(verification_qa)
        
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        
        final_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            response_format={"type": "json_object"},
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verified_reply = final_response.choices[0].message.content
        print("Chain of verification completed. Final response generated.\n")
        
        return verified_reply
        
    except Exception as e:
        print(f"ERROR in Chain of Verification: {str(e)}")
        print("Falling back to initial response.\n")
        return initial_reply
        
# openai explore corpus call
def get_openai_top_n(
    prompt,
    user_model,
    specificity,
    model_source,
    api_key,
    research_question,
    creativity
):
    """
    Get response from OpenAI API with system message.
    """
    from openai import OpenAI

    base_url = (
        "https://api.perplexity.ai" if model_source == "perplexity" 
        else "https://router.huggingface.co/v1" if model_source == "huggingface"
        else "https://api.x.ai/v1" if model_source == "xai"
        else None
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

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
    
    return response_obj.choices[0].message.content

def get_anthropic_top_n(
    prompt,
    user_model,
    model_source,
    specificity,
    api_key,
    research_question,
    creativity
):
    """
    Get response from Anthropic API with system prompt.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Build system prompt
    if research_question:
        system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}")
    else:
        system_content = "You are a helpful assistant."
    
    response_obj = client.messages.create(
        model=user_model,
        max_tokens=4096,
        system=system_content,
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        **({"temperature": creativity} if creativity is not None else {})
    )
    
    return response_obj.content[0].text