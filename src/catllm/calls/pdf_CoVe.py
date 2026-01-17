# PDF-aware Chain of Verification (CoVe) functions for various LLM providers
# These functions include the PDF page in verification steps for accurate document-based categorization


def pdf_chain_of_verification_openai(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering,
    pdf_content
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with OpenAI.
    The PDF page (as image) is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Args:
        pdf_content: The PDF page content in OpenAI format (as image_url dict after conversion)
    """
    try:
        # STEP 2: Generate verification questions (text only - questions about the categorization)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **({"temperature": creativity} if creativity is not None else {})
        )

        verification_questions = verification_response.choices[0].message.content

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content
            ]

            answer_response = client.chat.completions.create(
                model=user_model,
                messages=[{'role': 'user', 'content': message_content}],
                **({"temperature": creativity} if creativity is not None else {})
            )

            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content
        ]

        final_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': final_message_content}],
            response_format={"type": "json_object"},
            **({"temperature": creativity} if creativity is not None else {})
        )

        verified_reply = final_response.choices[0].message.content

        return verified_reply

    except Exception as e:
        return initial_reply


def pdf_chain_of_verification_anthropic(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    pdf_content,
    api_key=None
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Anthropic Claude.
    The PDF page is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Uses direct HTTP requests instead of Anthropic SDK.

    Args:
        pdf_content: The PDF page content in Anthropic format (dict with type: "document")
        api_key: Anthropic API key for authentication
    """
    import requests

    if api_key is None:
        return initial_reply

    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    def make_anthropic_request(messages, max_tokens=4096):
        """Helper to make Anthropic API requests."""
        payload = {
            "model": user_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if creativity is not None:
            payload["temperature"] = creativity

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")
        return ""

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_questions = make_anthropic_request(
            [{'role': 'user', 'content': step2_filled}]
        )

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content
            ]

            answer = make_anthropic_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content
        ]

        verified_reply = make_anthropic_request(
            [{'role': 'user', 'content': final_message_content}]
        )

        return verified_reply

    except Exception as e:
        return initial_reply


def pdf_chain_of_verification_google(
    initial_reply,
    prompt,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    url,
    headers,
    creativity,
    remove_numbering,
    make_google_request,
    pdf_data,
    mime_type
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Google Gemini.
    The PDF page is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Args:
        pdf_data: Base64 encoded PDF page data
        mime_type: MIME type of the content (e.g., "application/pdf")
    """
    import time

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        payload_step2 = {
            "contents": [{
                "parts": [{"text": step2_filled}]
            }],
            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
        }

        result_step2 = make_google_request(url, headers, payload_step2)
        verification_questions = result_step2["candidates"][0]["content"]["parts"][0]["text"]

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            time.sleep(2)  # Rate limit handling
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            payload_step3 = {
                "contents": [{
                    "parts": [
                        {"text": step3_filled},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": pdf_data
                            }
                        }
                    ]
                }],
                **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
            }

            result_step3 = make_google_request(url, headers, payload_step3)
            answer = result_step3["candidates"][0]["content"]["parts"][0]["text"]
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<PROMPT>>', prompt)
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        payload_step4 = {
            "contents": [{
                "parts": [
                    {"text": step4_filled},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": pdf_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({"temperature": creativity} if creativity is not None else {})
            }
        }

        result_step4 = make_google_request(url, headers, payload_step4)
        verified_reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]

        return verified_reply

    except Exception as e:
        return initial_reply


def pdf_chain_of_verification_mistral(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering,
    pdf_content
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Mistral AI.
    The PDF page (as image) is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Args:
        pdf_content: The PDF page content in Mistral format (dict with image_url after conversion)
    """
    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **({"temperature": creativity} if creativity is not None else {})
        )

        verification_questions = verification_response.choices[0].message.content

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content
            ]

            answer_response = client.chat.complete(
                model=user_model,
                messages=[{'role': 'user', 'content': message_content}],
                **({"temperature": creativity} if creativity is not None else {})
            )

            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content
        ]

        final_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': final_message_content}],
            response_format={"type": "json_object"},
            **({"temperature": creativity} if creativity is not None else {})
        )

        verified_reply = final_response.choices[0].message.content

        return verified_reply

    except Exception as e:
        return initial_reply
