"""
Academic domain (cat-ademic): OpenAlex paper fetching or upload.
"""

import streamlit as st
from components.file_upload import render_csv_upload


def render_domain_panel(function_id):
    """Render input panel for the Academic domain."""
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
        options=["Upload CSV/Excel", "Fetch from OpenAlex"],
        horizontal=True,
        key="acad_data_source",
    )

    if data_source == "Upload CSV/Excel":
        input_data, description, filename, df = render_csv_upload(key_prefix="acad_")
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["df"] = df
    else:
        st.markdown("#### OpenAlex Search")
        journal_name = st.text_input("Journal Name", placeholder="e.g., 'Nature'", key="acad_journal")
        if journal_name:
            result["domain_kwargs"]["journal_name"] = journal_name

        journal_issn = st.text_input("Journal ISSN (optional)", placeholder="e.g., '0028-0836'", key="acad_issn")
        if journal_issn:
            result["domain_kwargs"]["journal_issn"] = journal_issn

        journal_field = st.text_input("Field (optional)", placeholder="e.g., 'Computer Science'", key="acad_field")
        if journal_field:
            result["domain_kwargs"]["journal_field"] = journal_field

        topic_name = st.text_input("Topic (optional)", placeholder="e.g., 'machine learning'", key="acad_topic")
        if topic_name:
            result["domain_kwargs"]["topic_name"] = topic_name

        col1, col2 = st.columns(2)
        with col1:
            paper_limit = st.number_input("Max Papers", min_value=1, max_value=500, value=50, key="acad_limit")
            result["domain_kwargs"]["paper_limit"] = paper_limit
        with col2:
            polite_email = st.text_input("Email (polite pool)", placeholder="your@email.com", key="acad_email")
            if polite_email:
                result["domain_kwargs"]["polite_email"] = polite_email

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From Date", value=None, key="acad_date_from")
            if date_from:
                result["domain_kwargs"]["date_from"] = str(date_from)
        with col2:
            date_to = st.date_input("To Date", value=None, key="acad_date_to")
            if date_to:
                result["domain_kwargs"]["date_to"] = str(date_to)

        result["description"] = journal_name or topic_name or "academic papers"
        result["original_filename"] = "openalex_data"

    # Context fields
    st.markdown("#### Context (optional)")
    col1, col2 = st.columns(2)
    with col1:
        journal_ctx = st.text_input("Journal", key="acad_journal_ctx", placeholder="e.g., 'Nature'")
        if journal_ctx:
            result["domain_kwargs"]["journal"] = journal_ctx
    with col2:
        field_ctx = st.text_input("Field", key="acad_field_ctx", placeholder="e.g., 'Neuroscience'")
        if field_ctx:
            result["domain_kwargs"]["field"] = field_ctx

    research_focus = st.text_input("Research Focus", key="acad_focus",
                                   placeholder="e.g., 'deep learning applications in healthcare'")
    if research_focus:
        result["domain_kwargs"]["research_focus"] = research_focus

    return result
