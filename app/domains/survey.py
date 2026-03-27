"""
Survey domain (cat-survey): adds survey_question input.
"""

import streamlit as st
from domains.general import render_domain_panel as render_general_panel


def render_domain_panel(function_id):
    """Render input panel for the Survey domain.

    Same as General but adds a 'Survey Question' text input.
    """
    result = render_general_panel(function_id)

    survey_question = st.text_input(
        "Survey Question",
        placeholder="e.g., 'What is the primary reason you moved to your current neighborhood?'",
        help="The question respondents were asked. Helps the LLM understand context.",
        key="survey_question",
    )

    result["domain_kwargs"]["survey_question"] = survey_question
    return result
