"""
Social Media domain (cat-vader): fetch from platforms or upload data.
"""

import streamlit as st
from components.file_upload import render_csv_upload


def render_domain_panel(function_id):
    """Render input panel for the Social Media domain."""
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
        options=["Upload CSV/Excel", "Fetch from Platform"],
        horizontal=True,
        key="sm_data_source",
    )

    if data_source == "Upload CSV/Excel":
        input_data, description, filename, df = render_csv_upload(key_prefix="sm_")
        result["input_data"] = input_data
        result["description"] = description
        result["original_filename"] = filename
        result["df"] = df
    else:
        # Fetch from platform
        platform = st.selectbox(
            "Platform",
            options=["threads", "bluesky", "reddit", "mastodon", "youtube"],
            key="sm_platform",
        )
        result["domain_kwargs"]["sm_source"] = platform

        handle = st.text_input("Handle / Username", placeholder="e.g., @username", key="sm_handle")
        if handle:
            result["domain_kwargs"]["sm_handle"] = handle

        sm_limit = st.number_input("Max Posts to Fetch", min_value=1, max_value=500, value=50, key="sm_limit")
        result["domain_kwargs"]["sm_limit"] = sm_limit

        col1, col2 = st.columns(2)
        with col1:
            sm_months = st.number_input("Months Back", min_value=0, max_value=24, value=0, key="sm_months",
                                        help="0 = no limit")
            if sm_months > 0:
                result["domain_kwargs"]["sm_months"] = sm_months
        with col2:
            sm_days = st.number_input("Days Back", min_value=0, max_value=365, value=0, key="sm_days",
                                      help="0 = no limit")
            if sm_days > 0:
                result["domain_kwargs"]["sm_days"] = sm_days

        # YouTube-specific options
        if platform == "youtube":
            yt_content = st.selectbox("Content Type", options=["video", "comments"], key="sm_yt_content")
            result["domain_kwargs"]["sm_youtube_content"] = yt_content
            if yt_content == "video":
                yt_transcript = st.checkbox("Include Transcripts", key="sm_yt_transcript")
                result["domain_kwargs"]["sm_youtube_transcript"] = yt_transcript

        # Credentials
        with st.expander("Platform Credentials"):
            st.caption("Required for some platforms (e.g., Reddit client_id/secret, Threads token).")
            cred_key = st.text_input("Credential / Token", type="password", key="sm_credential")
            if cred_key:
                if platform == "reddit":
                    client_id = st.text_input("Reddit Client ID", key="sm_reddit_id")
                    client_secret = st.text_input("Reddit Client Secret", type="password", key="sm_reddit_secret")
                    if client_id and client_secret:
                        result["domain_kwargs"]["sm_credentials"] = {
                            "client_id": client_id,
                            "client_secret": client_secret,
                        }
                else:
                    result["domain_kwargs"]["sm_credentials"] = {"token": cred_key}

        result["description"] = f"{platform} posts"
        result["original_filename"] = f"{platform}_data"

    # Context fields
    st.markdown("#### Context (optional)")
    col1, col2 = st.columns(2)
    with col1:
        platform_ctx = st.text_input("Platform", key="sm_platform_ctx",
                                     placeholder="e.g., 'Twitter'")
        if platform_ctx:
            result["domain_kwargs"]["platform"] = platform_ctx
    with col2:
        handle_ctx = st.text_input("Handle", key="sm_handle_ctx",
                                   placeholder="e.g., '@elonmusk'")
        if handle_ctx:
            result["domain_kwargs"]["handle"] = handle_ctx

    hashtags = st.text_input("Hashtags (comma-separated)", key="sm_hashtags",
                             placeholder="e.g., '#AI, #tech'")
    if hashtags:
        result["domain_kwargs"]["hashtags"] = [h.strip() for h in hashtags.split(",") if h.strip()]

    return result
