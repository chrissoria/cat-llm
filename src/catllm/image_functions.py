# image multi-class (binary) function
def image_multi_class(
    image_description, 
    image_input,
    categories,
    api_key,
    user_model="gpt-4o",
    creativity=None,
    to_csv=False,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import glob
    import base64
    from pathlib import Path

    if save_directory is not None and not os.path.isdir(save_directory):
    # Directory doesn't exist - raise an exception to halt execution
        raise FileNotFoundError(f"Directory {save_directory} doesn't exist")

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

    model_source = model_source.lower() # eliminating case sensitivity 

    if not isinstance(image_input, list):
        # If image_input is a filepath (string)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))
    
        print(f"Found {len(image_files)} images.")
    else:
        # If image_files is already a list
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    cat_num = len(categories)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)

    # ensure number of categories is what user wants
    print("Categories to classify:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    
    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Categorising images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration
        
    # Only open the file if path is valid
        if os.path.isdir(img_path):
            encoded = "Not a Valid Image, contains file path"
        else:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                    encoded = f"Error: {str(e)}"
    # Handle extension safely
        if encoded.startswith("Error:") or encoded == "Not a Valid Image, contains file path":
            encoded_image = encoded
            valid_image = False
            extracted_jsons.append("""{"no_valid_path": 1}""")
        else:
            ext = Path(img_path).suffix.lstrip(".").lower()
            encoded_image = f"data:image/{ext};base64,{encoded}"
            valid_image = True
    
    # Handle extension safely
        ext = Path(img_path).suffix.lstrip(".").lower()
        if model_source == "openai" or model_source == "mistral":
            encoded_image = f"data:image/{ext};base64,{encoded}"
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are an image-tagging assistant.\n"
                        f"Task ► Examine the attached image and decide, **for each category below**, "
                        f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                        f"Image is expected to show: {image_description}\n\n"
                        f"Categories:\n{categories_str}\n\n"
                        f"Output format ► Respond with **only** a JSON object whose keys are the "
                        f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                        f"No additional keys, comments, or text.\n\n"
                        f"Example (three categories):\n"
                        f"{example_JSON}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": encoded_image, "detail": "high"},
                },
            ]
            
        elif model_source == "anthropic":
            encoded_image = f"data:image/{ext};base64,{encoded}"
            prompt = [
                {"type": "text",
                    "text": (
                        f"You are an image-tagging assistant.\n"
                        f"Task ► Examine the attached image and decide, **for each category below**, "
                        f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                        f"Image is expected to show: {image_description}\n\n"
                        f"Categories:\n{categories_str}\n\n"
                        f"Output format ► Respond with **only** a JSON object whose keys are the "
                        f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                        f"No additional keys, comments, or text.\n\n"
                        f"Example (three categories):\n"
                        f"{example_JSON}"
                    ),
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                }
            ]
        if model_source == "openAI":
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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "anthropic":
            import anthropic
            reply = None
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = message.content[0].text
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")
        #if no valid image path is provided
        elif  valid_image == False:
            reply = "invalid image path"
            print("Skipped NaN input or invalid path")
            #extracted_jsons.append("""{"no_valid_path": 1}""")
            link1.append("Error processing input: {e}")
        else:
            raise ValueError("Unknown source! Choose from OpenAI, Perplexity, or Mistral")
            # in situation that no JSON is found
        if reply is not None:
            if reply == "invalid image path":
                extracted_jsons.append("""{"no_valid_path": 1}""")
            else:
                extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
                if extracted_json:
                    cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                    extracted_jsons.append(cleaned_json)
                else:
                    error_message = """{"1":"e"}"""
                    extracted_jsons.append(error_message)
                    print(error_message)
        else:
            error_message = """{"1":"e"}"""
            extracted_jsons.append(error_message)
            print(error_message)

        # --- Safety Save ---
        if safety:
            #print(f"Saving CSV to: {save_directory}")
            # Save progress so far
            temp_df = pd.DataFrame({
                'image_input': image_files[:i+1],
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
        'image_input': (
            image_files.reset_index(drop=True) if isinstance(image_files, (pd.DataFrame, pd.Series)) 
            else pd.Series(image_files)
        ),
        'link1': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)

    if to_csv:
        if save_directory is None:
            save_directory = os.getcwd()
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)
    
    return categorized_data

#image score function
def image_score_drawing(
    reference_image_description,
    image_input,
    reference_image,
    api_key,
    columns="numbered",
    user_model="gpt-4o-2024-11-20",
    creativity=None,
    to_csv=False,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import glob
    import base64
    from pathlib import Path

    if save_directory is not None and not os.path.isdir(save_directory):
    # Directory doesn't exist - raise an exception to halt execution
        raise FileNotFoundError(f"Directory {save_directory} doesn't exist")

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

    model_source = model_source.lower() # eliminating case sensitivity 

    if not isinstance(image_input, list):
        # If image_input is a filepath (string)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))
    
        print(f"Found {len(image_files)} images.")
    else:
        # If image_files is already a list
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")
    
    with open(reference_image, 'rb') as f:
        reference = base64.b64encode(f.read()).decode('utf-8')
        reference_image = f"data:image/{reference_image.split('.')[-1]};base64,{reference}"
    
    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Categorising images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration
        
    # Only open the file if path is valid
        if os.path.isdir(img_path):
            encoded = "Not a Valid Image, contains file path"
        else:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                    encoded = f"Error: {str(e)}"
    # Handle extension safely
        if encoded.startswith("Error:") or encoded == "Not a Valid Image, contains file path":
            encoded_image = encoded
            valid_image = False
            
        else:
            ext = Path(img_path).suffix.lstrip(".").lower()
            encoded_image = f"data:image/{ext};base64,{encoded}"
            valid_image = True
    
    # Handle extension safely
        ext = Path(img_path).suffix.lstrip(".").lower()
        encoded_image = f"data:image/{ext};base64,{encoded}"

        if model_source == "openai" or model_source == "mistral":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual similarity assessment system.\n"
                        f"Task ► Compare these two images:\n"
                        f"1. REFERENCE (left): {reference_image_description}\n"
                        f"2. INPUT (right): User-provided drawing\n\n"
                        f"Rating criteria:\n"
                        f"1: No meaningful similarity (fundamentally different)\n"
                        f"2: Barely recognizable similarity (25% match)\n" 
                        f"3: Partial match (50% key features)\n"
                        f"4: Strong alignment (75% features)\n"
                        f"5: Near-perfect match (90%+ similarity)\n\n"
                        f"Output format ► Return ONLY:\n"
                        "{\n"
                        '  "score": [1-5],\n'
                        '  "summary": "reason you scored"\n'
                        "}\n\n"
                        f"Critical rules:\n"
                        f"- Score must reflect shape, proportions, and key details\n"
                        f"- List only concrete matching elements from reference\n"
                        f"- No markdown or additional text"
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": reference_image, "detail": "high"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": encoded_image, "detail": "high"}
                }
            ]

        elif model_source == "anthropic":  # Changed to elif
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual similarity assessment system.\n"
                        f"Task ► Compare these two images:\n"
                        f"1. REFERENCE (left): {reference_image_description}\n"
                        f"2. INPUT (right): User-provided drawing\n\n"
                        f"Rating criteria:\n"
                        f"1: No meaningful similarity (fundamentally different)\n"
                        f"2: Barely recognizable similarity (25% match)\n" 
                        f"3: Partial match (50% key features)\n"
                        f"4: Strong alignment (75% features)\n"
                        f"5: Near-perfect match (90%+ similarity)\n\n"
                        f"Output format ► Return ONLY:\n"
                        "{\n"
                        '  "score": [1-5],\n'
                        '  "summary": "reason you scored"\n'
                        "}\n\n"
                        f"Critical rules:\n"
                        f"- Score must reflect shape, proportions, and key details\n"
                        f"- List only concrete matching elements from reference\n"
                        f"- No markdown or additional text"
                    )
                },
                {
                    "type": "image",  # Added missing type
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": reference
                    }
                },
                {
                    "type": "image",  # Added missing type
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg", 
                        "data": encoded
                    }
                }
            ]

            
        if model_source == "openai":
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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = message.content[0].text  # Anthropic returns content as list
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")
        #if no valid image path is provided
        elif  valid_image == False:
            reply = "invalid image path"
            print("Skipped NaN input or invalid path")
            #extracted_jsons.append("""{"no_valid_path": 1}""")
            link1.append("Error processing input: {e}")
        else:
            raise ValueError("Unknown source! Choose from OpenAI, Perplexity, or Mistral")
            # in situation that no JSON is found
        if reply is not None:
            if reply == "invalid image path":
                extracted_jsons.append("""{"no_valid_path": 1}""")
            else:
                extracted_json = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
                if extracted_json:
                    cleaned_json = extracted_json[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '').replace("  ", '')
                    extracted_jsons.append(cleaned_json)
                else:
                    error_message = """{"1":"e"}"""
                    extracted_jsons.append(error_message)
                    print(error_message)
        else:
            error_message = """{"1":"e"}"""
            extracted_jsons.append(error_message)
            print(error_message)

        # --- Safety Save ---
        if safety:
            # Save progress so far
            temp_df = pd.DataFrame({
                'image_input': image_files[:i+1],
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
        'image_input': (
            image_files.reset_index(drop=True) if isinstance(image_files, (pd.DataFrame, pd.Series)) 
            else pd.Series(image_files)
        ),
        'link1': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)

    if to_csv:
        if save_directory is None:
            save_directory = os.getcwd()
        categorized_data.to_csv(os.path.join(save_directory, filename), index=False)
    
    return categorized_data

# image features function
def image_features(
    image_description, 
    image_input,
    features_to_extract,
    api_key,
    user_model="gpt-4o-2024-11-20",
    creativity=None,
    to_csv=False,
    safety=False,
    filename="categorized_data.csv",
    save_directory=None,
    model_source="OpenAI"
):
    import os
    import json
    import pandas as pd
    import regex
    from tqdm import tqdm
    import glob
    import base64
    from pathlib import Path

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

    model_source = model_source.lower() # eliminating case sensitivity 

    if not isinstance(image_input, list):
        # If image_input is a filepath (string)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_input, ext)))
    
        print(f"Found {len(image_files)} images.")
    else:
        # If image_files is already a list
        image_files = image_input
        print(f"Provided a list of {len(image_input)} images.")
    
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(features_to_extract))
    cat_num = len(features_to_extract)
    category_dict = {str(i+1): "0" for i in range(cat_num)}
    example_JSON = json.dumps(category_dict, indent=4)
    
    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Scoring images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration
        
    # Only open the file if path is valid
        if os.path.isdir(img_path):
            encoded = "Not a Valid Image, contains file path"
        else:
            try:
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                    encoded = f"Error: {str(e)}"
    # Handle extension safely
        if encoded.startswith("Error:") or encoded == "Not a Valid Image, contains file path":
            encoded_image = encoded
            valid_image = False
            
        else:
            ext = Path(img_path).suffix.lstrip(".").lower()
            encoded_image = f"data:image/{ext};base64,{encoded}"
            valid_image = True

        if model_source == "openai" or model_source == "mistral":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual question answering assistant.\n"
                        f"Task ► Analyze the attached image and answer these specific questions:\n\n"
                        f"Image context: {image_description}\n\n"
                        f"Questions to answer:\n{categories_str}\n\n"
                        f"Output format ► Return **only** a JSON object where:\n"
                        f"- Keys are question numbers ('1', '2', ...)\n"
                        f"- Values are concise answers (numbers, short phrases)\n\n"
                        f"Example for 3 questions:\n"
                        "{\n"
                        '  "1": "4",\n'
                        '  "2": "blue",\n'
                        '  "3": "yes"\n'
                        "}\n\n"
                        f"Important rules:\n"
                        f"1. Answer directly - no explanations\n"
                        f"2. Use exact numerical values when possible\n"
                        f"3. For yes/no questions, use 'yes' or 'no'\n"
                        f"4. Never add extra keys or formatting"
                        ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_image, "detail": "high"},
                            },
            ]
        elif model_source == "anthropic":
            prompt = [
                {
                    "type": "text",
                    "text": (
                        f"You are a visual question answering assistant.\n"
                        f"Task ► Analyze the attached image and answer these specific questions:\n\n"
                        f"Image context: {image_description}\n\n"
                        f"Questions to answer:\n{categories_str}\n\n"
                        f"Output format ► Return **only** a JSON object where:\n"
                        f"- Keys are question numbers ('1', '2', ...)\n"
                        f"- Values are concise answers (numbers, short phrases)\n\n"
                        f"Example for 3 questions:\n"
                        "{\n"
                        '  "1": "4",\n'
                        '  "2": "blue",\n'
                        '  "3": "yes"\n'
                        "}\n\n"
                        f"Important rules:\n"
                        f"1. Answer directly - no explanations\n"
                        f"2. Use exact numerical values when possible\n"
                        f"3. For yes/no questions, use 'yes' or 'no'\n"
                        f"4. Never add extra keys or formatting"
                    )
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                }
            ]
        if model_source == "openai":
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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif model_source == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    **({"temperature": creativity} if creativity is not None else {})
                )
                reply = message.content[0].text  # Anthropic returns content as list
                link1.append(reply)
            except Exception as e:
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

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
                if "model" in str(e).lower():
                    raise ValueError(f"Invalid OpenAI model '{user_model}': {e}")
                else:
                    print("An error occurred: {e}")
                    link1.append("Error processing input: {e}")

        elif  valid_image == False:
            print("Skipped NaN input or invalid path")
            reply = None
            link1.append("Error processing input: {e}")
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
            #print(f"Saving CSV to: {save_directory}")
            # Save progress so far
            temp_df = pd.DataFrame({
                'image_input': image_files[:i+1],
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
        'image_input': (
            image_files.reset_index(drop=True) if isinstance(image_files, (pd.DataFrame, pd.Series)) 
            else pd.Series(image_files)
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