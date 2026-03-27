"""
Summarize function page: focus, max_length, instructions, run + results.
"""

import os
import sys
import time
import tempfile
import pandas as pd
import streamlit as st

import catllm
from config import render_single_model_selector, get_model_source
from components.progress import ProgressTracker
from components.reports import generate_methodology_report_pdf
from components.code_generation import generate_summarize_code
from components.results_display import render_summarize_results
from components.file_upload import count_pdf_pages
from domains.registry import get_callable


def render(domain_id, domain_panel):
    """Render the summarize function page."""
    input_data = domain_panel["input_data"]
    input_type = domain_panel["input_type"]
    description = domain_panel["description"]
    original_filename = domain_panel["original_filename"]
    mode = domain_panel.get("mode")
    domain_kwargs = domain_panel.get("domain_kwargs", {})

    st.markdown("---")
    st.markdown("### Summarization Options")

    focus = st.text_input("Focus (optional)", placeholder="e.g., 'main arguments', 'key findings'")
    max_length = st.number_input("Max Summary Length (words, 0=no limit)", min_value=0, max_value=1000, value=0)
    max_length = max_length if max_length > 0 else None
    instructions = st.text_input("Additional Instructions (optional)", placeholder="e.g., 'use bullet points'")

    # Policy-specific: format and tone
    format_val = None
    tone_val = None
    if domain_id == "policy":
        format_val = st.selectbox("Format", options=["paragraph", "bullets", "one-liner", "structured", "report"], key="policy_format")
        tone_val = st.selectbox("Tone", options=["eli5", "legal"], key="policy_tone")
        domain_kwargs["format"] = format_val
        domain_kwargs["tone"] = tone_val

    st.markdown("### Model Selection")
    model, api_key, model_source, key_error = render_single_model_selector(key_prefix="summarize_")

    if st.button("Summarize Data", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload data first")
            return

        if key_error:
            st.error(key_error)
            return
        items_list = input_data if isinstance(input_data, list) else [input_data]
        tracker = ProgressTracker("Processing item")

        summarize_fn = get_callable(domain_id, "summarize")
        if summarize_fn is None:
            st.error(f"Summarize not available for domain '{domain_id}'")
            return

        summarize_kwargs = {
            "input_data": items_list,
            "api_key": api_key,
            "description": description,
            "user_model": model,
            "model_source": model_source,
            "progress_callback": tracker.get_callback(),
        }
        if mode:
            summarize_kwargs["mode"] = mode
        if focus and focus.strip():
            summarize_kwargs["focus"] = focus.strip()
        if max_length:
            summarize_kwargs["max_length"] = max_length
        if instructions and instructions.strip():
            summarize_kwargs["instructions"] = instructions.strip()
        summarize_kwargs.update(domain_kwargs)

        try:
            start_time = time.time()
            result_df = summarize_fn(**summarize_kwargs)
            processing_time = time.time() - start_time
            tracker.complete(f"Completed {len(result_df)} items in {processing_time:.1f}s")

            # Replace temp PDF paths
            if input_type == "pdf" and "pdf_path" in result_df.columns:
                pdf_name_map = st.session_state.get("pdf_name_map", {})
                def replace_temp_path(val):
                    if pd.isna(val):
                        return val
                    val_str = str(val)
                    for temp_path, orig_name in pdf_name_map.items():
                        if temp_path in val_str:
                            return val_str.replace(temp_path, orig_name + ".pdf")
                    return val_str
                result_df["pdf_path"] = result_df["pdf_path"].apply(replace_temp_path)

            # Save CSV
            with tempfile.NamedTemporaryFile(mode="w", suffix="_summarized.csv", delete=False) as f:
                result_df.to_csv(f.name, index=False)
                csv_path = f.name

            # Success rate
            if "processing_status" in result_df.columns:
                success_rate = (result_df["processing_status"] == "success").sum() / len(result_df) * 100
            else:
                success_rate = 100.0

            catllm_version = getattr(catllm, "__version__", "unknown")
            python_version = sys.version.split()[0]

            # Report
            pdf_path = generate_methodology_report_pdf(
                task_type="summarize", model=model, column_name=description,
                num_rows=len(result_df), model_source=model_source,
                filename=original_filename, success_rate=success_rate,
                result_df=result_df, processing_time=processing_time,
                catllm_version=catllm_version, python_version=python_version,
                input_type=input_type, description=description, domain=domain_id,
                focus=focus if focus else None, max_length=max_length,
            )

            # Code
            code = generate_summarize_code(
                domain_id, input_type, description, model, model_source,
                focus=focus if focus else None, max_length=max_length,
                instructions=instructions if instructions else None, mode=mode,
                domain_kwargs=domain_kwargs,
            )

            st.session_state.results = {
                "df": result_df,
                "csv_path": csv_path,
                "pdf_path": pdf_path,
                "code": code,
                "status": f"Summarized {len(result_df)} items in {processing_time:.1f}s",
                "task_type": "summarize",
            }
            st.success(f"Summarized {len(result_df)} items in {processing_time:.1f}s")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
