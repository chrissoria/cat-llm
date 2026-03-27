"""
General domain (cat-stack): no extra parameters, standard file upload only.
"""

import streamlit as st
from components.file_upload import render_csv_upload, render_pdf_upload, render_image_upload


def render_domain_panel(function_id):
    """Render input panel for the General domain.

    Args:
        function_id: Current function ("classify", "extract", "explore", "summarize").

    Returns:
        dict with keys:
            input_data: list or None
            input_type: "text", "pdf", or "image"
            description: str
            original_filename: str
            mode: str or None (PDF mode)
            df: DataFrame or None
            domain_kwargs: dict of domain-specific kwargs (empty for general)
    """
    input_type_choice = st.radio(
        "Input Type",
        options=["Text Data (CSV/Excel)", "PDF Documents", "Images"],
        horizontal=True,
        key="input_type_radio",
    )

    result = {
        "input_data": None,
        "input_type": "text",
        "description": "",
        "original_filename": "data",
        "mode": None,
        "df": None,
        "domain_kwargs": {},
    }

    if input_type_choice == "Text Data (CSV/Excel)":
        result["input_type"] = "text"
        input_data, description, filename, df = render_csv_upload()
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["df"] = df

    elif input_type_choice == "PDF Documents":
        result["input_type"] = "pdf"
        input_data, description, filename, mode = render_pdf_upload()
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["mode"] = mode

    else:  # Images
        result["input_type"] = "image"
        input_data, description, filename = render_image_upload()
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename

    return result
