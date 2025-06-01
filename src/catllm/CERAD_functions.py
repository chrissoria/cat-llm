# a function specifically for CERAD Constructional Praxis test
# specifically for pictures of drawings of shapes like circles, diamonds, rectangles, and cubes

"""
Areas for improvement:
1. Prompt refinement: adjusting the prompt so that it produces a more accurate score.
2. Image preprocessing: adjusting the images so that they are easier to be analyzed by the models.
3. Model selection: using a different model that is better suited for image analysis.
4. Model Ensembling: using multiple models and combining their scores to produce a more accurate score.
5. Prompt ensembling: using multiple prompts and combining their scores to produce a more accurate score.
6. Post-processing: adjusting the way scores are calculated after the model has output its assessment.
7. Efficiency: optimizing the code to run faster, cheaper, and more efficiently.
8. Drawn-format versatility: making the function more versatile to handle different scenarios, such as shapes drawn on tablets.
9. Image input flexibility: allowing the function to accept images in various formats, such as URLs or raw image data.
10. Test variety: expanding or adding functions to handle score more tests relevant for cogntive assesment, such as the MMSE.
11. Error handling: improving error handling to better manage unexpected inputs or model failures.
"""
def cerad_drawn_score(
    shape, 
    image_input,
    api_key,
    user_model="gpt-4o-2024-11-20",
    creativity=0,
    reference_in_image=False,
    provide_reference=False,
    safety=False,
    filename="categorized_data.csv",
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

    shape = shape.lower()

    if shape == "circle":
        categories = ["The image contains a drawing that clearly represents a circle",
                    "The drawing does not resemble a circle",
                    "The drawing resembles a circle",
                    "The circle is closed",
                    "The circle is almost closed",
                    "The circle is circular",
                    "The circle is almost circular",
                    "None of the above descriptions apply"]
    elif shape == "diamond":
        categories = ["The image contains a drawing that clearly represents a diamond shape",
                    "It has a drawing of a square",
                    "A drawn shape DOES NOT resemble a diamond",
                    "A drawn shape resembles a diamond",
                    "The drawn shape has 4 sides",
                    "The drawn shape sides are about equal",
                    "If a diamond is drawn it's more elaborate than a simple diamond (such as overlapping diamonds or a diamond with an extras lines inside)",
                    "None of the above descriptions apply"]
    elif shape == "rectangles" or shape == "overlapping rectangles":
        categories = ["The image contains a drawing that clearly represents overlapping rectangles",
                    "A drawn shape DOES NOT resemble a overlapping rectangles",
                    "A drawn shape resembles a overlapping rectangles",
                    "Rectangle 1 has 4 sides",
                    "Rectangle 2 has 4 sides",
                    "The rectangles are overlapping",
                    "The rectangles overlap contains a longer vertical rectangle with top and bottom portruding",
                    "None of the above descriptions apply"]
    elif shape == "cube":
        categories = ["The image contains a drawing that clearly represents a cube (3D box shape)",
                    "The image does NOT contain any drawing that resembles a cube or 3D box",
                    "The image contains a WELL-DRAWN recognizable cube with proper 3D perspective",
                    "If a cube is present: the front face appears as a square or diamond shape",
                    "If a cube is present: internal/hidden edges are visible (showing 3D depth, not just an outline)",
                    "If a cube is present: the front and back faces appear parallel to each other",
                    "The image contains only a 2D square (flat shape, no 3D appearance)",
                    "None of the above descriptions apply"]
    else:
        raise ValueError("Invalid shape! Choose from 'circle', 'diamond', 'rectangles', or 'cube'.")

    image_extensions = [
    '*.png', '*.jpg', '*.jpeg',
    '*.gif', '*.webp', '*.svg', '*.svgz', '*.avif', '*.apng',
    '*.tif', '*.tiff', '*.bmp',
    '*.heif', '*.heic', '*.ico',
    '*.psd'
    ]

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
    
    link1 = []
    extracted_jsons = []

    for i, img_path in enumerate(tqdm(image_files, desc="Scoring images"), start=0):
    # Check validity first
        if img_path is None or not os.path.exists(img_path):
            link1.append("Skipped NaN input or invalid path")
            extracted_jsons.append("""{"no_valid_image": 1}""")
            continue  # Skip the rest of the loop iteration
        
    # Only open the file if path is valid
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
    
    # Handle extension safely
        ext = Path(img_path).suffix.lstrip(".").lower()
        encoded_image = f"data:image/{ext};base64,{encoded}"

        if reference_in_image:
            reference_text = f"This image contains a perfect reference image of a {shape}. Next to is a drawing that is meant to be similar to the reference {shape}.\n\n"
        else:
            reference_text = f"Image is expected to show within it a drawing of a {shape}.\n\n"
    
        prompt = [
            {
                "type": "text",
                "text": (
                    f"You are an image-tagging assistant trained in the CERAD Constructional Praxis test.\n"
                    f"Task ► Examine the attached image and decide, **for each category below**, "
                    f"whether it is PRESENT (1) or NOT PRESENT (0).\n\n"
                    f"{reference_text}"
                    f"Categories:\n{categories_str}\n\n"
                    f"Output format ► Respond with **only** a JSON object whose keys are the "
                    f"quoted category numbers ('1', '2', …) and whose values are 1 or 0. "
                    f"No additional keys, comments, or text.\n\n"
                    f"Example:\n"
                    f"{example_JSON}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": encoded_image, "detail": "high"},
            },
        ]
        if model_source == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=creativity
                )
                reply = response_obj.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")

        elif model_source == "Perplexity":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            try:
                response_obj = client.chat.completions.create(
                    model=user_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=creativity
                )
                reply = response_obj.choices[0].message.content
                link1.append(reply)
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")
        elif model_source == "Anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                message = client.messages.create(
                    model=user_model,
                    max_tokens=1024,
                    temperature=creativity,
                    messages=[{"role": "user", "content": prompt}]
                )
                reply = message.content[0].text  # Anthropic returns content as list
                link1.append(reply)
            except Exception as e:
                print(f"An error occurred: {e}")
                link1.append(f"Error processing input: {e}")
        elif model_source == "Mistral":
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            try:
                response = client.chat.complete(
                    model=user_model,
                    messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=creativity
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
            if filename is None:
                filepath = os.path.join(os.getcwd(), 'catllm_data.csv')
            else:
                filepath = filename
            temp_df.to_csv(filepath, index=False)

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
        'image_input': image_files,
        'link1': pd.Series(link1).reset_index(drop=True),
        'json': pd.Series(extracted_jsons).reset_index(drop=True)
    })
    categorized_data = pd.concat([categorized_data, normalized_data], axis=1)
    columns_to_convert = ["1", "2", "3", "4", "5", "6", "7"]
    categorized_data[columns_to_convert] = categorized_data[columns_to_convert].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    if shape == "circle":

        categorized_data = categorized_data.rename(columns={
            "1": "drawing_present",
            "2": "not_similar",
            "3": "similar",
            "4": "cir_closed",
            "5": "cir_almost_closed",
            "6": "cir_round",
            "7": "cir_almost_round",
            "8": "none"
        })

        categorized_data['score'] = categorized_data['cir_almost_closed'] + categorized_data['cir_closed'] + categorized_data['cir_round'] + categorized_data['cir_almost_round']
        categorized_data.loc[categorized_data['none'] == 1, 'score'] = 0
        categorized_data.loc[(categorized_data['drawing_present'] == 0) & (categorized_data['score'] == 0), 'score'] = 0

    elif shape == "diamond":

        categorized_data = categorized_data.rename(columns={
            "1": "drawing_present",
            "2": "diamond_square",
            "3": "not_similar",
            "4": "similar",
            "5": "diamond_4_sides",
            "6": "diamond_equal_sides",
            "7": "complex_diamond",
            "8": "none"
        })

        categorized_data['score'] = categorized_data['diamond_4_sides'] + categorized_data['diamond_equal_sides'] + categorized_data['similar']

        categorized_data.loc[categorized_data['none'] == 1, 'score'] = 0
        categorized_data.loc[(categorized_data['diamond_square'] == 1) & (categorized_data['score'] == 0), 'score'] = 2

    elif shape == "rectangles" or shape == "overlapping rectangles":

        categorized_data = categorized_data.rename(columns={
            "1":"drawing_present",
            "2": "not_similar",
            "3": "similar",
            "4": "r1_4_sides",
            "5": "r2_4_sides",
            "6": "rectangles_overlap",
            "7": "rectangles_cross",
            "8": "none"
        })

        categorized_data['score'] = 0
        categorized_data.loc[(categorized_data['r1_4_sides'] == 1) & (categorized_data['r2_4_sides'] == 1), 'score'] = 1
        categorized_data.loc[(categorized_data['rectangles_overlap'] == 1) & (categorized_data['rectangles_cross'] == 1), 'score'] += 1
        categorized_data.loc[categorized_data['none'] == 1, 'score'] = 0

    elif shape == "cube":

        categorized_data = categorized_data.rename(columns={
            "1": "drawing_present",
            "2": "not_similar",
            "3": "similar", 
            "4": "cube_front_face",
            "5": "cube_internal_lines",
            "6": "cube_opposite_sides",
            "7": "square_only",
            "8": "none"
        })

        categorized_data['score'] = categorized_data['cube_front_face'] + categorized_data['cube_internal_lines'] + categorized_data['cube_opposite_sides'] + categorized_data['similar']
        categorized_data.loc[categorized_data['similar'] == 1, 'score'] = categorized_data['score'] + 1
        categorized_data.loc[categorized_data['none'] == 1, 'score'] = 0
        categorized_data.loc[(categorized_data['drawing_present'] == 0) & (categorized_data['score'] == 0), 'score'] = 0
        categorized_data.loc[(categorized_data['not_similar'] == 1) & (categorized_data['score'] == 0), 'score'] = 0
        categorized_data.loc[categorized_data['score'] > 4, 'score'] = 4

    else:
        raise ValueError("Invalid shape! Choose from 'circle', 'diamond', 'rectangles', or 'cube'.")

    categorized_data.loc[categorized_data['no_valid_image'] == 1, 'score'] = None

    if filename is not None:
        categorized_data.to_csv(filename, index=False)
    
    return categorized_data