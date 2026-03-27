"""
Cognitive domain (cat-cog): CERAD drawing scoring.
Special case -- only supports cerad_score function.
"""

import streamlit as st
from components.file_upload import render_image_upload


def render_domain_panel(function_id):
    """Render input panel for the Cognitive domain.

    Only supports image upload for CERAD scoring.
    """
    result = {
        "input_data": None,
        "input_type": "image",
        "description": "CERAD drawing",
        "original_filename": "cerad_images",
        "mode": None,
        "df": None,
        "domain_kwargs": {},
    }

    # The CERAD page handles its own image upload and shape selection,
    # so this panel is minimal. The function page (cerad_page.py) does the work.
    st.info("Select a shape and upload drawing image(s) below.")

    return result
