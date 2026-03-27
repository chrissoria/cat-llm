"""
Session state initialization and reset helpers.
"""

import streamlit as st
from config import MAX_CATEGORIES, INITIAL_CATEGORIES


def init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        # Navigation
        "domain": "general",
        "function": "classify",
        # API keys (persisted across domain/function switches)
        "api_keys": {},
        # Classify state
        "categories": [""] * MAX_CATEGORIES,
        "category_count": INITIAL_CATEGORIES,
        "task_mode": None,
        "extracted_categories": None,
        "extraction_params": None,
        "ensemble_num_runs": 3,
        # Data
        "survey_data": None,
        "pdf_data": None,
        "image_data": None,
        "example_loaded": False,
        "pdf_name_map": {},
        # Results
        "results": None,
        # UI
        "show_code_modal": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results():
    """Clear results and related state."""
    st.session_state.results = None
    st.session_state.show_code_modal = False


def reset_all():
    """Full reset of all working state."""
    st.session_state.categories = [""] * MAX_CATEGORIES
    st.session_state.category_count = INITIAL_CATEGORIES
    st.session_state.task_mode = None
    st.session_state.extracted_categories = None
    st.session_state.extraction_params = None
    st.session_state.results = None
    st.session_state.show_code_modal = False
    st.session_state.example_loaded = False
    st.session_state.pdf_name_map = {}
