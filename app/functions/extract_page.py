"""
Extract function page: standalone category extraction.
"""

import time
import streamlit as st

from config import render_single_model_selector, get_model_source
from components.progress import ProgressTracker
from components.code_generation import generate_extract_code
from components.results_display import render_extract_results
from domains.registry import get_callable


def render(domain_id, domain_panel):
    """Render the extract function page."""
    input_data = domain_panel["input_data"]
    input_type = domain_panel["input_type"]
    description = domain_panel["description"]
    mode = domain_panel.get("mode")
    domain_kwargs = domain_panel.get("domain_kwargs", {})

    st.markdown("---")
    st.markdown("### Category Extraction Options")

    max_categories = st.slider("Number of Categories to Extract", min_value=3, max_value=25, value=12)
    specificity = st.selectbox("Specificity", options=["Broad", "Moderate", "Narrow"], index=0)
    focus = st.text_input("Focus (optional)", placeholder="e.g., 'emotional responses'", key="extract_focus")

    st.markdown("### Model Selection")
    model, api_key, model_source, key_error = render_single_model_selector(key_prefix="extract_standalone_")

    if st.button("Extract Categories", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload data first")
            return

        if key_error:
            st.error(key_error)
            return
        tracker = ProgressTracker("Extracting categories")

        extract_fn = get_callable(domain_id, "extract")
        if extract_fn is None:
            st.error(f"Extract not available for domain '{domain_id}'")
            return

        items_list = input_data if isinstance(input_data, list) else [input_data]
        num_items = len(items_list)
        divisions = max(1, min(5, num_items // 15))

        extract_kwargs = {
            "input_data": items_list,
            "api_key": api_key,
            "input_type": input_type,
            "description": description,
            "user_model": model,
            "model_source": model_source,
            "max_categories": int(max_categories),
            "specificity": specificity.lower(),
            "progress_callback": tracker.get_callback(),
        }
        if mode:
            extract_kwargs["mode"] = mode
        if focus and focus.strip():
            extract_kwargs["focus"] = focus.strip()
        extract_kwargs.update(domain_kwargs)

        try:
            extract_result = extract_fn(**extract_kwargs)
            categories = extract_result.get("top_categories", [])
            processing_time = time.time() - tracker.start_time
            tracker.complete(f"Completed in {processing_time:.1f}s")

            code = generate_extract_code(
                domain_id, input_type, description, model, model_source,
                int(max_categories), mode=mode, domain_kwargs=domain_kwargs,
            )

            st.session_state.results = {
                "categories": categories,
                "counts_df": extract_result.get("counts_df"),
                "code": code,
                "task_type": "extract",
            }
            from components.history import save_run
            save_run(st.session_state.results, st.session_state.get("settings"))
            st.success(f"Extracted {len(categories)} categories in {processing_time:.1f}s")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
