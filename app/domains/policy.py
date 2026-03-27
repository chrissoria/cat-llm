"""
Policy domain (cat-pol): registered data sources, doc_type, date filters.
"""

import streamlit as st
from components.file_upload import render_csv_upload, render_pdf_upload


def render_domain_panel(function_id):
    """Render input panel for the Policy domain."""
    result = {
        "input_data": None,
        "input_type": "text",
        "description": "",
        "original_filename": "data",
        "mode": None,
        "df": None,
        "domain_kwargs": {},
    }

    data_source = st.radio(
        "Data Source",
        options=["Upload CSV/Excel", "Upload PDF", "Fetch from Source"],
        horizontal=True,
        key="pol_data_source",
    )

    if data_source == "Upload CSV/Excel":
        input_data, description, filename, df = render_csv_upload(key_prefix="pol_")
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["df"] = df

    elif data_source == "Upload PDF":
        result["input_type"] = "pdf"
        input_data, description, filename, mode = render_pdf_upload(key_prefix="pol_")
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["mode"] = mode

    else:
        # Fetch from registered source
        st.markdown("#### Policy Data Source")

        # Try to get available sources
        try:
            import catllm
            sources = catllm.list_policy_sources()
            source_ids = list(sources.keys()) if isinstance(sources, dict) else []
        except Exception:
            source_ids = [
                "city_san_diego", "city_la", "city_berkeley",
                "federal_laws", "trump_truth_social",
            ]

        source = st.selectbox("Source", options=source_ids, key="pol_source")
        result["domain_kwargs"]["source"] = source

        doc_type = st.text_input("Document Type (optional)", placeholder="e.g., 'ordinance', 'resolution'",
                                 key="pol_doc_type")
        if doc_type:
            result["domain_kwargs"]["doc_type"] = doc_type

        col1, col2 = st.columns(2)
        with col1:
            since = st.date_input("Since", value=None, key="pol_since")
            if since:
                result["domain_kwargs"]["since"] = str(since)
        with col2:
            until = st.date_input("Until", value=None, key="pol_until")
            if until:
                result["domain_kwargs"]["until"] = str(until)

        n = st.number_input("Max Documents", min_value=1, max_value=500, value=50, key="pol_n")
        result["domain_kwargs"]["n"] = n

        result["description"] = f"{source} documents"
        result["original_filename"] = f"{source}_data"

    # Context
    doc_context = st.text_input("Document Context (optional)",
                                placeholder="e.g., 'City council proceedings from San Diego'",
                                key="pol_context")
    if doc_context:
        result["domain_kwargs"]["document_context"] = doc_context

    return result
