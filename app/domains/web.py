"""
Web domain (cat-web): URL list or CSV upload, source_domain, content_type.
"""

import streamlit as st
from components.file_upload import render_csv_upload


def render_domain_panel(function_id):
    """Render input panel for the Web domain."""
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
        options=["Upload CSV/Excel", "Enter URLs"],
        horizontal=True,
        key="web_data_source",
    )

    if data_source == "Upload CSV/Excel":
        input_data, description, filename, df = render_csv_upload(key_prefix="web_")
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["df"] = df
    else:
        urls_text = st.text_area(
            "URLs (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            height=150,
            key="web_urls",
        )
        if urls_text:
            urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
            if urls:
                result["input_data"] = urls
                result["description"] = "web content"
                result["original_filename"] = "urls"
                st.success(f"{len(urls)} URL(s) entered")

    # Web-specific fields
    st.markdown("#### Web Context (optional)")
    col1, col2 = st.columns(2)
    with col1:
        source_domain = st.text_input("Source Domain", placeholder="e.g., 'nytimes.com'", key="web_domain")
        if source_domain:
            result["domain_kwargs"]["source_domain"] = source_domain
    with col2:
        content_type = st.text_input("Content Type", placeholder="e.g., 'news article'", key="web_content_type")
        if content_type:
            result["domain_kwargs"]["content_type"] = content_type

    timeout = st.number_input("Fetch Timeout (seconds)", min_value=5, max_value=120, value=30, key="web_timeout")
    result["domain_kwargs"]["timeout"] = timeout

    # Web metadata key-value pairs
    with st.expander("Additional Metadata"):
        st.caption("Add key-value pairs to inject into prompts.")
        meta_key = st.text_input("Key", key="web_meta_key", placeholder="e.g., 'section'")
        meta_val = st.text_input("Value", key="web_meta_val", placeholder="e.g., 'technology'")
        if meta_key and meta_val:
            result["domain_kwargs"]["web_metadata"] = {meta_key: meta_val}

    return result
