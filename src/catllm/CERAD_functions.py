# CERAD Constructional Praxis scoring function
# Scores drawings of shapes (circles, diamonds, rectangles, cubes) using image_multi_class

"""
This function wraps image_multi_class to implement CERAD Constructional Praxis test scoring.
It defines shape-specific categories, calls image_multi_class for the LLM classification,
then applies algorithmic scoring rules to convert binary classifications into CERAD scores.

Supported shapes:
- circle: max score 2
- diamond: max score 3
- rectangles (overlapping): max score 2
- cube: max score 4
"""

# Only export cerad_drawn_score (not the internal import of image_multi_class)
__all__ = ["cerad_drawn_score"]

from .image_functions import image_multi_class


def cerad_drawn_score(
    shape,
    image_input,
    api_key,
    user_model="gpt-4o",
    creativity=None,
    safety=False,
    chain_of_thought=True,
    filename=None,
    save_directory=None,
    model_source="auto"
):
    """
    Score CERAD Constructional Praxis drawings using image_multi_class.

    Args:
        shape (str): Shape to score - 'circle', 'diamond', 'rectangles', or 'cube'
        image_input: Directory path or list of image file paths
        api_key (str): API key for the model provider
        user_model (str): Model to use (default: gpt-4o)
        creativity (float): Temperature setting (None uses model default)
        safety (bool): If True, saves progress after each image
        chain_of_thought (bool): If True, uses step-by-step reasoning
        filename (str): Output filename for CSV
        save_directory (str): Directory to save results
        model_source (str): Provider - 'auto', 'openai', 'anthropic', 'google', 'mistral'

    Returns:
        pd.DataFrame: Results with binary category columns and computed 'score' column
    """
    import pandas as pd
    from pathlib import Path

    shape = shape.lower()
    shape = "rectangles" if shape == "overlapping rectangles" else shape

    # Define shape-specific categories
    if shape == "circle":
        categories = [
            "The image contains a drawing that clearly represents a circle",
            "The image does NOT contain any drawing that resembles a circle",
            "The image contains a drawing that resembles a circle",
            "The circle is closed",
            "The circle is almost closed",
            "The circle is circular",
            "The circle is almost circular",
            "None of the above descriptions apply"
        ]
        image_description = "A hand-drawn circle for CERAD Constructional Praxis assessment"

    elif shape == "diamond":
        categories = [
            "The image contains a drawing that clearly represents a diamond shape",
            "It has a drawing of a square",
            "A drawn shape DOES NOT resemble a diamond",
            "A drawn shape resembles a diamond",
            "The drawn shape has 4 sides",
            "The drawn shape sides are about equal",
            "If a diamond is drawn it's more elaborate than a simple diamond (such as overlapping diamonds or a diamond with an extras lines inside)",
            "None of the above descriptions apply"
        ]
        image_description = "A hand-drawn diamond shape for CERAD Constructional Praxis assessment"

    elif shape == "rectangles":
        categories = [
            "The image contains a drawing that clearly represents overlapping rectangles",
            "The image does NOT contain any drawing that resembles overlapping rectangles",
            "The image contains a drawing that resembles overlapping rectangles",
            "If rectangle 1 is present and it has 4 sides",
            "If rectangle 2 is present and it has 4 sides",
            "The drawn rectangles are overlapping",
            "The drawn rectangles overlap to form a longer vertical rectangle with top and bottom sticking out",
            "None of the above descriptions apply"
        ]
        image_description = "Hand-drawn overlapping rectangles for CERAD Constructional Praxis assessment"

    elif shape == "cube":
        categories = [
            "The image contains a drawing that clearly represents a cube (3D box shape)",
            "The image does NOT contain any drawing that resembles a cube or 3D box",
            "The image contains a WELL-DRAWN recognizable cube with proper 3D perspective",
            "If a cube is present: the front face appears as a square or diamond shape",
            "If a cube is present: internal/hidden edges are visible (showing 3D depth, not just an outline)",
            "If a cube is present: the front and back faces appear parallel to each other",
            "The image contains only a 2D square (flat shape, no 3D appearance)",
            "None of the above descriptions apply"
        ]
        image_description = "A hand-drawn cube for CERAD Constructional Praxis assessment"

    else:
        raise ValueError("Invalid shape! Choose from 'circle', 'diamond', 'rectangles', or 'cube'.")

    # Call image_multi_class to get binary classifications
    result_df = image_multi_class(
        image_description=image_description,
        image_input=image_input,
        categories=categories,
        api_key=api_key,
        user_model=user_model,
        creativity=creativity,
        safety=safety,
        chain_of_thought=chain_of_thought,
        filename=filename,
        save_directory=save_directory,
        model_source=model_source
    )

    # Rename category columns to meaningful names based on shape
    if shape == "circle":
        column_mapping = {
            "category_1": "drawing_present",
            "category_2": "not_similar",
            "category_3": "similar",
            "category_4": "cir_closed",
            "category_5": "cir_almost_closed",
            "category_6": "cir_round",
            "category_7": "cir_almost_round",
            "category_8": "none"
        }
        result_df = result_df.rename(columns=column_mapping)

        # Calculate score
        result_df['score'] = (
            result_df['cir_almost_closed'].fillna(0) +
            result_df['cir_closed'].fillna(0) +
            result_df['cir_round'].fillna(0) +
            result_df['cir_almost_round'].fillna(0)
        )
        result_df.loc[result_df['none'] == 1, 'score'] = 0
        result_df.loc[(result_df['drawing_present'] == 0) & (result_df['score'] == 0), 'score'] = 0
        result_df.loc[result_df['score'] > 2, 'score'] = 2

    elif shape == "diamond":
        column_mapping = {
            "category_1": "drawing_present",
            "category_2": "diamond_square",
            "category_3": "not_similar",
            "category_4": "similar",
            "category_5": "diamond_4_sides",
            "category_6": "diamond_equal_sides",
            "category_7": "complex_diamond",
            "category_8": "none"
        }
        result_df = result_df.rename(columns=column_mapping)

        # Fix cases where model outputs 4 instead of 1
        result_df.loc[result_df['diamond_4_sides'] > 1, 'diamond_4_sides'] = 1

        result_df['score'] = (
            result_df['diamond_4_sides'].fillna(0) +
            result_df['diamond_equal_sides'].fillna(0) +
            result_df['similar'].fillna(0) +
            result_df['diamond_square'].fillna(0)
        )
        result_df.loc[result_df['none'] == 1, 'score'] = 0
        result_df.loc[result_df['score'] > 3, 'score'] = 3

    elif shape == "rectangles":
        column_mapping = {
            "category_1": "drawing_present",
            "category_2": "not_similar",
            "category_3": "similar",
            "category_4": "r1_4_sides",
            "category_5": "r2_4_sides",
            "category_6": "rectangles_overlap",
            "category_7": "rectangles_cross",
            "category_8": "none"
        }
        result_df = result_df.rename(columns=column_mapping)

        result_df['score'] = (
            result_df['rectangles_overlap'].fillna(0) +
            result_df['similar'].fillna(0) +
            result_df['rectangles_cross'].fillna(0)
        )
        result_df.loc[result_df['none'] == 1, 'score'] = 0
        result_df.loc[result_df['score'] > 2, 'score'] = 2

    elif shape == "cube":
        column_mapping = {
            "category_1": "drawing_present",
            "category_2": "not_similar",
            "category_3": "similar",
            "category_4": "cube_front_face",
            "category_5": "cube_internal_lines",
            "category_6": "cube_opposite_sides",
            "category_7": "square_only",
            "category_8": "none"
        }
        result_df = result_df.rename(columns=column_mapping)

        result_df['score'] = (
            result_df['cube_front_face'].fillna(0) +
            result_df['cube_internal_lines'].fillna(0) +
            result_df['cube_opposite_sides'].fillna(0) +
            result_df['similar'].fillna(0)
        )
        result_df.loc[result_df['similar'] == 1, 'score'] = result_df['score'] + 1
        result_df.loc[result_df['none'] == 1, 'score'] = 0
        result_df.loc[(result_df['drawing_present'] == 0) & (result_df['score'] == 0), 'score'] = 0
        result_df.loc[(result_df['not_similar'] == 1) & (result_df['score'] == 0), 'score'] = 0
        result_df.loc[result_df['score'] > 4, 'score'] = 4

    # Convert score to integer
    result_df['score'] = result_df['score'].astype('Int64')

    # Add image filename column
    result_df['image_file'] = result_df['image_input'].apply(lambda x: Path(x).name if pd.notna(x) else None)

    # Handle processing errors - set score to NA
    if 'processing_status' in result_df.columns:
        result_df.loc[result_df['processing_status'] == 'error', 'score'] = pd.NA

    # Save final results if filename provided
    if filename:
        import os
        save_path = os.path.join(save_directory, filename) if save_directory else filename
        result_df.to_csv(save_path, index=False)

    return result_df
