"""
CatLLM brand CSS theme.
"""

import streamlit as st


def inject_css():
    """Inject the CatLLM CSS theme into the page."""
    st.markdown(CSS, unsafe_allow_html=True)


CSS = """
<style>
/* Import Garamond font and apply globally */
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');

*:not([class*="icon"]):not([data-testid="stIconMaterial"]):not(svg):not(path) {
    font-family: 'EB Garamond', Garamond, Georgia, serif !important;
    font-size: 17px !important;
}

/* Preserve Streamlit icon fonts */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-size: 24px !important;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers with gradient accent */
h1 {
    background: linear-gradient(90deg, #E8A33C 0%, #D4872C 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* Card-like sections */
.stExpander {
    border: 1px solid #E8D5B5;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(232, 163, 60, 0.08);
}

/* File uploader styling */
.stFileUploader {
    border-radius: 12px;
}

.stFileUploader > div > div {
    border: 2px dashed #E8A33C;
    border-radius: 12px;
    background: linear-gradient(135deg, #FEFCF9 0%, #F5EFE6 100%);
}

/* Button styling */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
    border: 2px solid #E8A33C;
    background: #FEFCF9;
    color: #D4872C;
}

/* Tall button for example dataset (matches file uploader height) */
.tall-button .stButton > button {
    min-height: 107px;
    border-radius: 12px;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(232, 163, 60, 0.3);
    background: #F5EFE6;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E8A33C 0%, #D4872C 100%);
    border: none;
    color: white;
}

/* Success/info messages */
.stSuccess {
    background-color: #E8F5E9;
    border-left: 4px solid #4CAF50;
    border-radius: 0 8px 8px 0;
}

.stInfo {
    background-color: #FFF8E8;
    border-left: 4px solid #E8A33C;
    border-radius: 0 8px 8px 0;
}

/* Radio buttons */
.stRadio > div {
    gap: 0.5rem;
    display: flex;
    width: 100%;
}

.stRadio > div > label {
    background: #F5EFE6;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    flex: 1;
    text-align: center;
    justify-content: center;
}

.stRadio > div > label:hover {
    border-color: #E8A33C;
}

/* Text inputs */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #E8D5B5;
}

.stTextInput > div > div > input:focus {
    border-color: #E8A33C;
    box-shadow: 0 0 0 2px rgba(232, 163, 60, 0.2);
}

/* Select boxes */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #E8A33C 0%, #D4872C 100%);
    border-radius: 10px;
}

/* Slider */
.stSlider > div > div > div {
    background: #E8A33C;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #E8D5B5, transparent);
    margin: 1.5rem 0;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 12px;
    border: 1px solid #E8D5B5;
}

/* Metric cards */
.stMetric {
    background: linear-gradient(135deg, #FEFCF9 0%, #F5EFE6 100%);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #E8D5B5;
}

/* Download buttons */
.stDownloadButton > button {
    background: #F5EFE6;
    border: 1px solid #E8A33C;
    color: #D4872C;
}

.stDownloadButton > button:hover {
    background: #E8A33C;
    color: white;
}

/* Multiselect */
.stMultiSelect > div > div {
    border-radius: 8px;
}

/* Status indicator */
.stStatus {
    border-radius: 12px;
}

/* Column gaps */
[data-testid="column"] {
    padding: 0 0.5rem;
}

/* Logo and title alignment */
[data-testid="column"]:first-child img {
    border-radius: 8px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FEFCF9 0%, #F5EFE6 100%);
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
}
</style>
"""
