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
        
        final_response = client.messages.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            max_tokens=4096,
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verified_reply = final_response.content[0].text
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
            time.sleep(2)  # temporary rate limit handling
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
            **({"temperature": creativity} if creativity is not None else {})
        )
        
        verified_reply = final_response.choices[0].message.content
        print("Chain of verification completed. Final response generated.\n")
        
        return verified_reply
        
    except Exception as e:
        print(f"ERROR in Chain of Verification: {str(e)}")
        print("Falling back to initial response.\n")
        return initial_reply
