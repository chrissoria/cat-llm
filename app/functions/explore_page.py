"""
Explore function page: saturation analysis for category discovery.
"""

import time
import streamlit as st

from config import render_single_model_selector, get_model_source
from components.progress import ProgressTracker
from components.results_display import render_explore_results
from domains.registry import get_callable


def render(domain_id, domain_panel):
    """Render the explore function page."""
    input_data = domain_panel["input_data"]
    input_type = domain_panel["input_type"]
    description = domain_panel["description"]
    mode = domain_panel.get("mode")
    domain_kwargs = domain_panel.get("domain_kwargs", {})

    st.markdown("---")
    st.markdown("### Exploration Options")
    st.markdown("Explore discovers categories and checks for saturation (whether new categories keep appearing).")

    st.markdown("### Model Selection")
    model, api_key, model_source, key_error = render_single_model_selector(key_prefix="explore_")

    if st.button("Explore Categories", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload data first")
            return

        if key_error:
            st.error(key_error)
            return
        tracker = ProgressTracker("Exploring categories")

        explore_fn = get_callable(domain_id, "explore")
        if explore_fn is None:
            st.error(f"Explore not available for domain '{domain_id}'")
            return

        items_list = input_data if isinstance(input_data, list) else [input_data]

        explore_kwargs = {
            "input_data": items_list,
            "api_key": api_key,
            "description": description,
            "user_model": model,
            "model_source": model_source,
        }
        if mode:
            explore_kwargs["mode"] = mode
        explore_kwargs.update(domain_kwargs)

        try:
            categories = explore_fn(**explore_kwargs)
            processing_time = time.time() - tracker.start_time
            tracker.complete(f"Completed in {processing_time:.1f}s")

            st.session_state.results = {
                "categories": categories if isinstance(categories, list) else [],
                "code": f"""import catllm

result = catllm.explore(
    input_data=your_data,
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}"
)
print(result)
""",
                "task_type": "explore",
            }
            from components.history import save_run
            save_run(st.session_state.results, st.session_state.get("settings"))
            st.success(f"Discovered {len(categories) if isinstance(categories, list) else 0} categories")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
