"""
Survey domain (cat-survey): text-only CSV/Excel uploader + survey_question.
"""

import streamlit as st
from components.file_upload import render_csv_upload


def render_domain_panel(function_id):
    """Render input panel for the Survey domain.

    Survey responses are inherently textual, so this panel skips the
    multimodal Input Type radio and goes straight to the CSV/Excel uploader.
    """
    input_data, description, filename, df = render_csv_upload()

    survey_question = st.text_input(
        "Survey Question",
        placeholder="e.g., 'What is the primary reason you moved to your current neighborhood?'",
        help="The question respondents were asked. Helps the LLM understand context.",
        key="survey_question",
    )

    return {
        "input_data": input_data,
        "input_type": "text",
        "description": description,
        "original_filename": filename,
        "mode": None,
        "df": df,
        "domain_kwargs": {"survey_question": survey_question},
    }
