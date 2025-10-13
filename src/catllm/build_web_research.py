#build dataset classification
def build_web_research_dataset(
    search_question, 
    search_input,
    api_key,
    answer_format = "concise",
    additional_instructions = "",
    categories = ['Answer'],
    user_model="claude-sonnet-4-20250514",
    creativity=None,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="Anthropic",
    start_date=None,
    end_date=None,
    search_depth="", #enables Tavily searches
    tavily_api=None,
    output_urls = True,
    max_retries = 6, #API rate limit error handler retries
    time_delay=5
):
    import os
    import re
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import time
    from datetime import datetime

    #ensures proper date format
    def _validate_date(date_str):
        """Validates YYYY-MM-DD format"""
        if date_str is None:
            return True  # None is acceptable (means no date constraint)
        
        if not isinstance(date_str, str):
            return False
        
        # Check pattern: YYYY_MM_DD
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
        
        # Validate actual date
        try:
            year, month, day = date_str.split('-')
            datetime(int(year), int(month), int(day))
            return True
        except (ValueError, OverflowError):
            return False
    
    # Validate dates at the start of the function
    if not _validate_date(start_date):
        raise ValueError(f"start_date must be in YYYY-MM-DD format, got: {start_date}")
    
    if not _validate_date(end_date):
        raise ValueError(f"end_date must be in YYYY-MM-DD format, got: {end_date}")

    model_source = model_source.lower() # eliminating case sensitivity 

    if model_source == "perplexity" and start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    if model_source == "perplexity" and end_date is not None:  
        end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")

    # in case user switches to google but doesn't switch model
    if model_source == "google" and user_model == "claude-sonnet-4-20250514":
        user_model = "gemini-2.5-flash"
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)
    
    link1 = []
    extracted_jsons = []
    extracted_urls = []

    for idx, item in enumerate(tqdm(search_input, desc="Building dataset")):
        if idx > 0:  # Skip delay for first item only
            time.sleep(time_delay)
        reply = None  

        if pd.isna(item): 
            link1.append("Skipped NaN input")
            extracted_urls.append([])
            default_json = example_JSON 
            extracted_jsons.append(default_json)
        else:
            prompt = f"""<role>You are a research assistant specializing in finding current, factual information.</role>

            <task>Find information about {item}'s {search_question}</task>

            <rules>
            - Search for the most current and authoritative information available
            - Provide your answer as {answer_format}
            - Prioritize official sources when possible
            - If information is not found, state "Information not found"
            - Do not include any explanatory text or commentary beyond the JSON
                {additional_instructions}
            </rules>

            <format>
            Return your response as valid JSON with this exact structure:
            {{
            "answer": "Your factual answer or 'Information not found'",
            "second_best_answer": "Your second best factual answer or 'Information not found'",
            "confidence": "confidence in response 0-5 or 'Information not found'"
        }}

        </format>"""

            if start_date is not None and end_date is not None:
                append_text = f"\n- Focus on webpages with a page age between {start_date} and {end_date}."
                prompt = prompt.replace("<rules>", "<rules>" + append_text)
            elif start_date is not None:
                append_text = f"\n- Focus on webpages published after {start_date}."
                prompt = prompt.replace("<rules>", "<rules>" + append_text)
            elif end_date is not None:
                append_text = f"\n- Focus on webpages published before {end_date}."
                prompt = prompt.replace("<rules>", "<rules>" + append_text)

            if search_depth == "advanced" and model_source != "perplexity":
                try:
                    from tavily import TavilyClient
                    tavily_client = TavilyClient(tavily_api)
                    tavily_response = tavily_client.search(
                        query=f"{item}'s {search_question}",
                        include_answer=True,
                        max_results=15,
                        search_depth="advanced",
                        **({"start_date": start_date} if start_date is not None else {}),
                        **({"end_date": end_date} if end_date is not None else {})
                    )

                    urls = [
                        result['url'] 
                        for result in tavily_response.get('results', []) 
                        if 'url' in result
                    ]
                    seen = set()
                    urls = [u for u in urls if not (u in seen or seen.add(u))]
                    extracted_urls.append(urls)
        
                except Exception as e:
                    error_msg = str(e).lower()
                    if "unauthorized" in error_msg or "403" in error_msg or "401" in error_msg or "api_key" in error_msg:
                        raise ValueError("ERROR: Invalid or missing tavily_api required for advanced search. Get one at https://app.tavily.com/home. To install: pip install tavily-python") from e
                    else:
                        print(f"Tavily search error: {e}")
                        link1.append(f"Error with Tavily search: {e}")
                        extracted_urls.append([])
                        continue 

                #print(tavily_response)
            
                advanced_prompt = f"""Based on the following search results about {item}'s {search_question}, provide your answer in this EXACT JSON format and {answer_format}:
                If you can't find the information, respond with 'Information not found'.
                {{"answer": "your answer here or 'Information not found'",
                "second_best_answer": "your second best answer here or 'Information not found'",
                "confidence": "confidence in response 0-5 or 'Information not found'"}}

                Search results:
                {tavily_response}

                Additional context from sources:
                {chr(10).join([f"- {r.get('title', '')}: {r.get('content', '')}" for r in tavily_response.get('results', [])[:3]])}

                Return ONLY the JSON object, no other text."""
            
            if model_source == "anthropic" and search_depth != "advanced":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                #print(prompt)
                attempt = 0
                while attempt < max_retries:
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
                        
                        urls = [
                            item["url"]
                            for block in message.content
                            if getattr(block, "type", "") == "web_search_tool_result"
                            for item in (getattr(block, "content", []) or [])
                            if isinstance(item, dict) and item.get("type") == "web_search_result" and "url" in item
                        ]

                        seen = set()
                        urls = [u for u in urls if not (u in seen or seen.add(u))]
                        extracted_urls.append(urls)

                        break
                    except anthropic.RateLimitError as e:
                        wait_time = 2 ** attempt  # Exponential backoff, keeps doubling after each attempt
                        print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time) #in case user wants to try and buffer the amount of errors by adding a wait time before attemps
                        attempt += 1
                    except Exception as e:
                        print(f"A Non-rate-limit error occurred: {e}")
                        link1.append(f"Error processing input: {e}")
                        extracted_urls.append([])
                        break #stop retrying 
                else:
                    link1.append("Max retries exceeded for rate limit errors.")
                    extracted_urls.append([])

            elif model_source == "anthropic" and search_depth == "advanced":
                import anthropic
                claude_client = anthropic.Anthropic(api_key=api_key)

                attempt = 0
                while attempt < max_retries:
                    try:
                        message = claude_client.messages.create(
                            model=user_model,
                            max_tokens=1024,
                            messages=[{"role": "user", "content": advanced_prompt}],
                            **({"temperature": creativity} if creativity is not None else {})
                            )
            
                        reply = " ".join(
                            block.text
                            for block in message.content
                            if getattr(block, "type", "") == "text"
                            ).strip()
            
                        try:
                            import json
                            json_response = json.loads(reply)
                            final_answer = json_response.get('answer', reply)
                            link1.append(final_answer)
                        except json.JSONDecodeError:

                            print(f"JSON parse error, using raw reply: {reply}")
                            link1.append(reply)
            
                        break  # Success
            
                    except anthropic.RateLimitError as e:
                        wait_time = 2 ** attempt
                        print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        attempt += 1

                    except Exception as e:
                        print(f"A Non-rate-limit error occurred: {e}")
                        link1.append(f"Error processing input: {e}")
                        break
                else:
                    # Max retries exceeded
                    link1.append("Max retries exceeded for rate limit errors.")

            elif model_source == "google" and search_depth != "advanced":
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

                    urls = []
                    for cand in result.get("candidates", []):
                        rendered_html = (
                            cand.get("groundingMetadata", {})
                            .get("searchEntryPoint", {})
                            .get("renderedContent", "")
                        )
                        if rendered_html:
                        # regex: capture href="..."; limited to class="chip"
                            found = re.findall(
                                r'<a[^>]*class=["\']chip["\'][^>]*href=["\']([^"\']+)["\']',
                                rendered_html,
                                flags=re.IGNORECASE
                            )
                            urls.extend(found)

                    seen = set()
                    urls = [u for u in urls if not (u in seen or seen.add(u))]
                    extracted_urls.append(urls)
        
                    # extract reply from Google's response structure
                    if "candidates" in result and result["candidates"]:
                        reply = result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        reply = "No response generated"
            
                    link1.append(reply)
        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
                    extracted_urls.append([])

            elif model_source == "google" and search_depth == "advanced":
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
                try:
                    headers = {
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                        }
        
                    payload = {
                        "contents": [{"parts": [{"text": advanced_prompt}]}],
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

            elif model_source == "perplexity":

                from perplexity import Perplexity
                client = Perplexity(api_key=api_key)
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model=user_model,
                        max_tokens=1024,
                        **({"temperature": creativity} if creativity is not None else {}),
                        web_search_options={"search_context_size": "high" if search_depth == "advanced" else "medium"},
                        **({"search_after_date_filter": start_date} if start_date else {}),
                        **({"search_before_date_filter": end_date} if end_date else {}),
                        response_format={ #requiring a JSON
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "second_best_answer": {"type": "string"},
                                    "confidence": {"type": "integer"}
                            },
                            "required": ["answer", "second_best_answer"]
                        }
                    }
                }
            )

                    reply = response.choices[0].message.content
                    #print(response)
                    link1.append(reply)

                    urls = list(response.citations) if hasattr(response, 'citations') else []
                    
                    seen = set()
                    urls = [u for u in urls if not (u in seen or seen.add(u))]
                    extracted_urls.append(urls)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    link1.append(f"Error processing input: {e}")
                    extracted_urls.append([])
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
                    error_message = json.dumps({"answer": "e"})
                    extracted_jsons.append(error_message)
                    print(error_message)
            else:
                # Handle None reply case
                error_message = json.dumps({"answer": "e"})
                extracted_jsons.append(error_message)
                #print(error_message)

        # --- Safety Save ---
        if safety:
            # Save progress so far
            temp_df = pd.DataFrame({
                'raw_response': search_input[:idx+1]
                #'model_response': link1,
                #'json': extracted_jsons
            })
            # Normalize processed jsons so far
            normalized_data_list = []
            for json_str in extracted_jsons:
                try:
                    parsed_obj = json.loads(json_str)
                    normalized_data_list.append(pd.json_normalize(parsed_obj))
                except json.JSONDecodeError:
                    normalized_data_list.append(pd.DataFrame({"answer": ["e"]}))
            normalized_data = pd.concat(normalized_data_list, ignore_index=True)
            temp_urls = pd.DataFrame(extracted_urls).add_prefix("url_")
            temp_df = pd.concat([temp_df, normalized_data], axis=1)
            temp_df = pd.concat([temp_df, temp_urls], axis=1)
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
            normalized_data_list.append(pd.DataFrame({"answer": ["e"]}))
    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    # converting urls to dataframe and adding prefix
    df_urls = pd.DataFrame(extracted_urls).add_prefix("url_")

    categorized_data = pd.DataFrame({
        'search_input': (
            search_input.reset_index(drop=True) if isinstance(search_input, (pd.DataFrame, pd.Series)) 
            else pd.Series(search_input)
        ),
        'raw_response': pd.Series(link1).reset_index(drop=True),
        #'json': pd.Series(extracted_jsons).reset_index(drop=True),
        #"all_urls": pd.Series(extracted_urls).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    categorized_data = pd.concat([categorized_data, df_urls], axis=1)

    # drop second best answer column if it exists
    # we only ask for the second best answer to "force" the model to think more carefully about its best answer, but we don't actually need to keep it
    categorized_data = categorized_data.drop(columns=["second_best_answer"], errors='ignore')

    # dropping this column for advanced searches (this column is mostly useful for basic searches to see what the model saw)
    if search_depth == "advanced":
        categorized_data = categorized_data.drop(columns=["raw_response"], errors='ignore')

    #for users who don't want the urls included in the final dataframe
    if output_urls is False:
        categorized_data = categorized_data.drop(columns=[col for col in categorized_data.columns if col.startswith("url_")])   

    if save_directory is not None:
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)
    
    return categorized_data