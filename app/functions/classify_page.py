"""
Classify function page: categories (manual/auto-extract), classify modes, run + results.
"""

import os
import sys
import time
import tempfile
import pandas as pd
import streamlit as st

import catllm
from config import MAX_CATEGORIES, INITIAL_CATEGORIES, get_model_source, resolve_api_key, render_single_model_selector
from components.model_selection import render_model_selector
from components.progress import ProgressTracker
from components.reports import generate_methodology_report_pdf
from components.code_generation import generate_classify_code, generate_extract_code
from components.results_display import render_classify_results
from domains.registry import get_callable


def render(domain_id, domain_panel):
    """Render the classify function page.

    Args:
        domain_id: Current domain ID string.
        domain_panel: Dict returned by the domain's render_domain_panel().
    """
    input_data = domain_panel["input_data"]
    input_type = domain_panel["input_type"]
    description = domain_panel["description"]
    original_filename = domain_panel["original_filename"]
    mode = domain_panel.get("mode")
    domain_kwargs = domain_panel.get("domain_kwargs", {})

    st.markdown("---")

    # Task selection
    st.markdown("### What would you like to do?")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Enter Categories Manually", use_container_width=True):
            st.session_state.task_mode = "manual"
    with col_btn2:
        if st.button("Auto-extract Categories", use_container_width=True):
            st.session_state.task_mode = "auto_extract"

    # Auto-extract flow
    if st.session_state.task_mode == "auto_extract":
        _render_auto_extract(domain_id, input_data, input_type, description, mode, domain_kwargs)

    # Manual category entry + classify
    if st.session_state.task_mode == "manual":
        _render_manual_classify(domain_id, input_data, input_type, description, original_filename, mode, domain_kwargs)


def _render_auto_extract(domain_id, input_data, input_type, description, mode, domain_kwargs):
    """Render auto-extract UI and run extraction."""
    st.markdown("### Auto-extract Categories")
    st.markdown("We'll analyze your data to discover the main categories.")

    max_categories = st.slider("Number of Categories to Extract", min_value=3, max_value=25, value=12)
    specificity = st.selectbox("How specific?", options=["Broad", "Moderate", "Narrow"], index=0)
    focus = st.text_input("Focus (optional)", placeholder="e.g., 'emotional responses', 'financial factors'")

    # Model selection (single model for extraction)
    st.markdown("### Model Selection")
    model, api_key, model_source_val, key_error = render_single_model_selector(key_prefix="extract_")

    if st.button("Extract Categories", type="primary"):
        if input_data is None:
            st.error("Please upload data first")
        elif key_error:
            st.error(key_error)
        else:
            model_source = get_model_source(model)
            tracker = ProgressTracker("Extracting categories")

            extract_fn = get_callable(domain_id, "extract")
            if extract_fn is None:
                st.error(f"Extract not available for domain '{domain_id}'")
                return

            extract_kwargs = {
                "input_data": input_data,
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
            # Add domain-specific kwargs
            extract_kwargs.update(domain_kwargs)

            try:
                extract_result = extract_fn(**extract_kwargs)
                categories = extract_result.get("top_categories", [])
                processing_time = time.time() - tracker.start_time
                tracker.complete(f"Completed in {processing_time:.1f}s")

                if categories:
                    st.success(f"Extracted {len(categories)} categories in {processing_time:.1f}s")
                    st.session_state.extracted_categories = categories
                    st.session_state.extraction_params = {
                        "model": model,
                        "model_source": model_source,
                        "max_categories": int(max_categories),
                        "input_type": input_type,
                        "description": description,
                        "mode": mode,
                    }
                    st.session_state.task_mode = "manual"
                    st.rerun()
                else:
                    st.error("No categories were extracted from the data")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def _render_manual_classify(domain_id, input_data, input_type, description, original_filename, mode, domain_kwargs):
    """Render manual category entry and classification."""
    st.markdown("### Categories")

    # Pre-fill with extracted categories
    if st.session_state.extracted_categories:
        for i, cat in enumerate(st.session_state.extracted_categories[:MAX_CATEGORIES]):
            st.session_state.categories[i] = cat
        st.session_state.category_count = min(len(st.session_state.extracted_categories), MAX_CATEGORIES)
        st.session_state.extracted_categories = None

    placeholders = [
        "e.g., Positive sentiment", "e.g., Negative sentiment", "e.g., Product feedback",
        "e.g., Service complaint", "e.g., Feature request", "e.g., Custom category",
    ]
    categories_entered = []
    for i in range(st.session_state.category_count):
        placeholder = placeholders[i] if i < len(placeholders) else "e.g., Custom category"
        cat_value = st.text_input(
            f"Category {i + 1}",
            value=st.session_state.categories[i],
            placeholder=placeholder,
            key=f"cat_{i}",
        )
        st.session_state.categories[i] = cat_value
        if cat_value.strip():
            categories_entered.append(cat_value.strip())

    if st.session_state.category_count < MAX_CATEGORIES:
        if st.button("+ Add More"):
            st.session_state.category_count += 1
            st.rerun()

    # Model selection with classify modes
    st.markdown("### Model Selection")
    model_sel = render_model_selector(mode="classify", key_prefix="cls_")

    classify_mode = model_sel["classify_mode"]
    models_list = model_sel["models_list"]
    models_tuples = model_sel["models_tuples"]
    ensemble_runs = model_sel["ensemble_runs"]
    model_temperatures = model_sel["model_temperatures"]
    consensus_threshold = model_sel["consensus_threshold"]

    if st.button("Categorize Data", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload data first")
        elif not categories_entered:
            st.error("Please enter at least one category")
        elif model_sel["error"]:
            st.error(model_sel["error"])
        elif classify_mode == "Model Comparison" and len(models_list) < 2:
            st.error("Please select at least 2 models for comparison mode")
        elif classify_mode == "Ensemble" and len(models_list) < 3:
            st.error("Please select at least 3 models for ensemble mode")
        else:
            _run_classify(
                domain_id, input_data, input_type, description, original_filename,
                mode, categories_entered, models_tuples, models_list, classify_mode,
                consensus_threshold, model_temperatures, ensemble_runs, domain_kwargs,
            )


def _run_classify(
    domain_id, input_data, input_type, description, original_filename,
    mode, categories, models_tuples, models_list, classify_mode,
    consensus_threshold, model_temperatures, ensemble_runs, domain_kwargs,
):
    """Execute the classification and store results."""
    items_list = input_data if isinstance(input_data, list) else [input_data]
    tracker = ProgressTracker("Processing item" if input_type != "pdf" else "Processing page")

    classify_fn = get_callable(domain_id, "classify")
    if classify_fn is None:
        st.error(f"Classify not available for domain '{domain_id}'")
        return

    classify_kwargs = {
        "input_data": items_list,
        "categories": categories,
        "models": models_tuples,
        "description": description,
        "progress_callback": tracker.get_callback(),
    }
    if mode:
        classify_kwargs["mode"] = mode
    if classify_mode == "Ensemble":
        classify_kwargs["consensus_threshold"] = consensus_threshold
    # Add domain-specific kwargs
    classify_kwargs.update(domain_kwargs)

    try:
        start_time = time.time()
        result_df = classify_fn(**classify_kwargs)
        processing_time = time.time() - start_time
        tracker.complete(f"Completed {len(result_df)} items in {processing_time:.1f}s")

        # Replace temp PDF paths
        if input_type == "pdf" and "pdf_input" in result_df.columns:
            pdf_name_map = st.session_state.get("pdf_name_map", {})
            def replace_temp_path(val):
                if pd.isna(val):
                    return val
                val_str = str(val)
                for temp_path, orig_name in pdf_name_map.items():
                    temp_name = os.path.basename(temp_path).replace(".pdf", "")
                    if temp_name in val_str:
                        return val_str.replace(temp_name, orig_name)
                return val_str
            result_df["pdf_input"] = result_df["pdf_input"].apply(replace_temp_path)

        # Save CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix="_classified.csv", delete=False) as f:
            result_df.to_csv(f.name, index=False)
            csv_path = f.name

        # Success rate
        if "processing_status" in result_df.columns:
            success_rate = (result_df["processing_status"] == "success").sum() / len(result_df) * 100
        else:
            success_rate = 100.0

        # Version info
        catllm_version = getattr(catllm, "__version__", "unknown")
        python_version = sys.version.split()[0]

        # Report model string
        report_model = models_list[0] if len(models_list) == 1 else ", ".join(models_list)
        report_source = models_tuples[0][1] if len(models_list) == 1 else f"{classify_mode} ({len(models_list)} models)"

        # Generate code
        code = generate_classify_code(
            domain_id, input_type, description, categories, report_model, report_source,
            mode=mode, classify_mode=classify_mode, models_list=models_list,
            consensus_threshold=consensus_threshold, model_temperatures=model_temperatures,
            ensemble_runs=ensemble_runs if ensemble_runs else None,
            domain_kwargs=domain_kwargs,
        )

        # Generate report
        pdf_path = generate_methodology_report_pdf(
            task_type="classify", categories=categories, model=report_model,
            column_name=description, num_rows=len(result_df), model_source=report_source,
            filename=original_filename, success_rate=success_rate, result_df=result_df,
            processing_time=processing_time, catllm_version=catllm_version,
            python_version=python_version, input_type=input_type, description=description,
            classify_mode=classify_mode, models_list=models_list, code=code,
            consensus_threshold=consensus_threshold if classify_mode == "Ensemble" else None,
            domain=domain_id,
        )

        st.session_state.results = {
            "df": result_df,
            "csv_path": csv_path,
            "pdf_path": pdf_path,
            "code": code,
            "status": f"Classified {len(result_df)} items in {processing_time:.1f}s",
            "categories": categories,
            "classify_mode": classify_mode,
            "models_list": models_list,
            "model_temperatures": model_temperatures,
            "ensemble_runs": ensemble_runs if ensemble_runs else None,
            "task_type": "classify",
        }
        st.success(f"Classified {len(result_df)} items in {processing_time:.1f}s")
        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")
