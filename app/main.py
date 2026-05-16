"""
CatLLM Unified App — Entry point.

Run with: streamlit run app/main.py
"""

import base64
import os
import sys

# Add app directory to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st


def _logo_data_uri(filename: str = "logo_mark.png") -> str:
    path = os.path.join(os.path.dirname(__file__), "assets", filename)
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


LOGO_MARK_URI = _logo_data_uri()

# Page config must be first Streamlit call
st.set_page_config(
    page_title="CatLLM",
    page_icon="🐱",
    layout="wide",
)

from components.css import inject_css
from components.header import render_header
from components.update_check import check_for_updates, render_update_banner
from components.model_selection import render_api_keys_sidebar
from components.results_display import (
    render_classify_results,
    render_summarize_results,
    render_extract_results,
    render_explore_results,
)
from session import init_session_state, reset_all
from domains.registry import get_domain_ids, get_domain_label, get_functions
from components.settings import show_settings_dialog, init_settings
from components.history import render_history_sidebar

# Domain panel modules
from domains import general, survey, social_media, academic, policy, web, cognitive

# Function page modules
from functions import classify_page, summarize_page, extract_page, explore_page, cerad_page


# ---------------------------------------------------------------------------
# Domain -> panel module mapping
# ---------------------------------------------------------------------------
DOMAIN_PANELS = {
    "general": general,
    "survey": survey,
    "social_media": social_media,
    "academic": academic,
    "policy": policy,
    "web": web,
    "cognitive": cognitive,
}

# Function -> page module mapping
FUNCTION_PAGES = {
    "classify": classify_page,
    "extract": extract_page,
    "explore": explore_page,
    "summarize": summarize_page,
    "cerad_score": cerad_page,
}


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
inject_css()
init_session_state()
init_settings()
check_for_updates()

# ---------------------------------------------------------------------------
# Sidebar: domain selector, function selector, API keys
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:0.65rem; margin: 0 0 1.25rem 0; padding-bottom:0.75rem; border-bottom:1px solid rgba(232,237,245,0.10);">
      <img src="{LOGO_MARK_URI}" alt="" style="height:46px; width:auto; display:block; flex-shrink:0; transform:translateY(-4px);">
      <span style="font-family:'IBM Plex Sans',sans-serif; font-size:1.85rem; line-height:1; font-weight:600; letter-spacing:-0.025em; color:#E8EDF5;">CatLLM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

domain_ids = get_domain_ids()
domain_labels = [get_domain_label(d) for d in domain_ids]

selected_domain_label = st.sidebar.selectbox(
    "Domain",
    options=domain_labels,
    index=0,
    key="domain_selector",
)
selected_domain = domain_ids[domain_labels.index(selected_domain_label)]

# Update domain in session state; reset results when domain changes
if st.session_state.get("domain") != selected_domain:
    st.session_state.domain = selected_domain
    st.session_state.results = None
    st.session_state.task_mode = None

# Function selector (filtered by domain)
available_fns = get_functions(selected_domain)
fn_ids = list(available_fns.keys())
fn_labels = [available_fns[f]["label"] for f in fn_ids]

selected_fn_label = st.sidebar.selectbox(
    "Function",
    options=fn_labels,
    index=0,
    key="function_selector",
)
selected_fn = fn_ids[fn_labels.index(selected_fn_label)]

# Update function in session state; reset results when function changes
if st.session_state.get("function") != selected_fn:
    st.session_state.function = selected_fn
    st.session_state.results = None
    st.session_state.task_mode = None

# Input Type — only meaningful for General (other domains have fixed input types).
if selected_domain == "general":
    st.sidebar.selectbox(
        "Input Type",
        options=["Text Data (CSV/Excel)", "PDF Documents", "Images"],
        index=0,
        key="input_type_sidebar",
    )

st.sidebar.markdown("---")

# API key inputs
render_api_keys_sidebar()

st.sidebar.markdown("---")

# History section
with st.sidebar.expander("History"):
    render_history_sidebar()

st.sidebar.markdown("---")
if st.sidebar.button("Reset All", use_container_width=True):
    reset_all()
    st.rerun()
if st.sidebar.button("\u2699 Settings", use_container_width=True):
    show_settings_dialog()


# ---------------------------------------------------------------------------
# Main content: update banner (if any) + two columns (input | output)
# ---------------------------------------------------------------------------
render_update_banner()

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown(f"### {get_domain_label(selected_domain)} — {selected_fn_label}")

    # Render domain panel (data input + domain-specific params)
    domain_module = DOMAIN_PANELS.get(selected_domain, general)
    domain_panel = domain_module.render_domain_panel(selected_fn)

    # Render function page (task-specific UI + run button)
    fn_module = FUNCTION_PAGES.get(selected_fn)
    if fn_module:
        fn_module.render(selected_domain, domain_panel)


with col_output:
    st.markdown("### Results")

    results = st.session_state.get("results")
    if results:
        task_type = results.get("task_type", "classify")
        if task_type == "classify":
            render_classify_results(results)
        elif task_type == "summarize":
            render_summarize_results(results)
        elif task_type == "extract":
            render_extract_results(results)
        elif task_type == "explore":
            render_explore_results(results)
        # CERAD results are rendered inline by cerad_page
    else:
        st.info("Configure your task on the left and click the action button to see results here.")


# ---------------------------------------------------------------------------
# Footer: colophon expander
# ---------------------------------------------------------------------------
st.markdown("---")
render_header()
