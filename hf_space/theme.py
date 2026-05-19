"""
CatLLM brand CSS theme for the HuggingFace Space.

Mirrors the catllm.com site palette and the desktop app: dark mode with
teal (primary) and orange (secondary) accents, paired with IBM Plex Sans
(display), DM Sans (body), and DM Mono (codes & metadata).

Kept in sync with app/components/css.py — copy that file over when the
desktop theme changes.
"""

import base64
import os

import streamlit as st


def inject_css():
    """Inject the CatLLM CSS theme into the page."""
    st.markdown(CSS, unsafe_allow_html=True)


def _load_logo_uri(filename: str = "logo_mark.png") -> str:
    """Return a base64 data URI for the cat-shield mark, or empty string."""
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except FileNotFoundError:
        return ""


LOGO_MARK_URI = _load_logo_uri()


def render_wordmark(subtitle: str = "Research-grade text classification with LLMs"):
    """Render the CatLLM cat-shield + IBM Plex Sans wordmark + a short subtitle."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.75rem; margin: 0 0 0.5rem 0; padding-bottom:0.75rem; border-bottom:1px solid rgba(232,237,245,0.10);">
          <img src="{LOGO_MARK_URI}" alt="" style="height:54px; width:auto; display:block; flex-shrink:0; transform:translateY(-4px);">
          <div style="display:flex; flex-direction:column; gap:0.15rem;">
            <span style="font-family:'IBM Plex Sans',sans-serif; font-size:2.1rem; line-height:1; font-weight:600; letter-spacing:-0.025em; color:#E8EDF5;">CatLLM</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:#8A95A8;">{subtitle}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400&display=swap');

:root {
  --bg:          #1A1A1A;
  --bg-2:        #1F1F1F;
  --bg-3:        #262626;
  --bg-4:        #2E2E2E;
  --teal:        #00DCC3;
  --teal-dim:    #00A89A;
  --teal-glow:   rgba(0, 220, 195, 0.12);
  --teal-glow2:  rgba(0, 220, 195, 0.04);
  --orange:      #FF8C3C;
  --orange-dim:  #D87024;
  --orange-glow: rgba(255, 140, 60, 0.12);
  --text:        #E8EDF5;
  --text-2:      #8A95A8;
  --text-3:      #4E5A6D;
  --hairline:    rgba(232, 237, 245, 0.10);
  --hairline-strong: rgba(232, 237, 245, 0.22);
  --shadow:      rgba(0, 0, 0, 0.45);
}

/* ===================================================================
   Global typography — DM Sans body, IBM Plex Sans display, DM Mono code.
   =================================================================== */
*:not([class*="icon"]):not([data-testid="stIconMaterial"]):not(svg):not(path):not(.material-icons):not(.material-symbols-rounded) {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
    font-size: 15px;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-size: 22px !important;
}

h1, h2, h3, h4, .clm-display {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: var(--text) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text) !important;
}

h1 { font-size: 2.5rem !important; line-height: 1.05 !important; }
h2 { font-size: 1.85rem !important; line-height: 1.1; }
h3 { font-size: 1.25rem !important; line-height: 1.2; font-weight: 500 !important; }
h4 { font-size: 1.05rem !important; line-height: 1.25; font-weight: 500 !important; }

em, i, .italic {
    font-style: italic;
    color: var(--text-2);
}

code, kbd, pre, .mono, .clm-mono,
.stCodeBlock, .stCode, [class*="language-"] {
    font-family: 'DM Mono', ui-monospace, 'JetBrains Mono', monospace !important;
    font-size: 0.85em;
}

/* ===================================================================
   Page background — deep ink with subtle teal/orange auras.
   =================================================================== */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(ellipse at top left,    rgba(0, 220, 195, 0.06) 0%, transparent 55%),
        radial-gradient(ellipse at bottom right, rgba(255, 140, 60, 0.05) 0%, transparent 55%) !important;
    background-attachment: fixed, fixed !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: rgba(26, 26, 26, 0.75) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--hairline) !important;
}

.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

/* ===================================================================
   Buttons — soft surface tile, teal halo on hover.
   =================================================================== */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    border: 1px solid var(--hairline-strong) !important;
    background: var(--bg-3) !important;
    color: var(--text) !important;
    padding: 0.65rem 1.2rem !important;
    transition: background 140ms ease, border-color 140ms ease, color 140ms ease, box-shadow 140ms ease, transform 140ms ease !important;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.03) inset, 0 1px 2px var(--shadow) !important;
    white-space: normal !important;
    line-height: 1.3 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: var(--teal-dim) !important;
    background: var(--bg-4) !important;
    color: var(--teal) !important;
    box-shadow: 0 0 0 1px var(--teal-glow), 0 8px 24px var(--teal-glow2), 0 1px 0 rgba(255,255,255,0.04) inset !important;
}

.stButton > button:active,
.stDownloadButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(1px) !important;
}

/* Disabled state — keep dark surface, dim the text (overrides Streamlit
   default white-on-white that's unreadable in dark mode). */
.stButton > button:disabled,
.stDownloadButton > button:disabled,
.stFormSubmitButton > button:disabled,
.stButton > button[disabled],
.stDownloadButton > button[disabled],
.stFormSubmitButton > button[disabled] {
    background: var(--bg-2) !important;
    color: var(--text-3) !important;
    border-color: var(--hairline) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
    color: var(--bg) !important;
    font-weight: 600 !important;
    box-shadow: 0 0 0 1px var(--teal-glow), 0 6px 18px var(--teal-glow) !important;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--teal-dim) !important;
    border-color: var(--teal-dim) !important;
    color: var(--bg) !important;
}

.tall-button .stButton > button {
    min-height: 107px;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}

/* ===================================================================
   File uploader.
   =================================================================== */
.stFileUploader {
    margin-top: 1.25rem !important;
}

.stFileUploader > div > div,
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--hairline-strong) !important;
    border-radius: 10px !important;
    background: var(--bg-2) !important;
    transition: border-color 160ms ease, background 160ms ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal-dim) !important;
    background: var(--bg-3) !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--text-2) !important;
}

/* "Browse files" button inside the dropzone — overrides Streamlit's
   default light secondary-button styling. */
[data-testid="stFileUploaderDropzone"] button,
.stFileUploader button {
    background: var(--bg-3) !important;
    color: var(--text) !important;
    border: 1px solid var(--hairline-strong) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    box-shadow: none !important;
    transition: background 140ms ease, border-color 140ms ease, color 140ms ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
.stFileUploader button:hover {
    background: var(--bg-4) !important;
    border-color: var(--teal-dim) !important;
    color: var(--teal) !important;
}

/* Dropzone cloud icon */
[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--text-2) !important;
    color: var(--text-2) !important;
}

/* ===================================================================
   Inputs.
   =================================================================== */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stChatInput textarea {
    border-radius: 8px !important;
    border: 1px solid var(--hairline-strong) !important;
    background: var(--bg-2) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 140ms ease, box-shadow 140ms ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow) !important;
    outline: none !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: 8px !important;
    border: 1px solid var(--hairline-strong) !important;
    background: var(--bg-2) !important;
}

.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: var(--text-3) !important;
}

[data-baseweb="tag"] {
    border-radius: 6px !important;
    background: var(--teal-glow) !important;
    color: var(--teal) !important;
    border: 1px solid var(--teal-dim) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Select dropdown menu — BaseWeb popovers used by Streamlit selectbox /
   multiselect / date-picker. Default Streamlit theme leaves these light. */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"],
[data-baseweb="select-dropdown"],
ul[role="listbox"] {
    background: var(--bg-3) !important;
    background-color: var(--bg-3) !important;
    border: 1px solid var(--hairline-strong) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    box-shadow: 0 8px 24px var(--shadow) !important;
}

[data-baseweb="menu"] li,
[role="option"],
[role="listbox"] li {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.55rem 0.85rem !important;
}

[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"],
[role="listbox"] li:hover {
    background: var(--bg-4) !important;
    color: var(--teal) !important;
}

/* The closed selectbox itself — make sure background stays dark. */
[data-baseweb="select"] > div,
[data-baseweb="select"] [data-baseweb="select-control"] {
    background: var(--bg-2) !important;
    color: var(--text) !important;
}

[data-baseweb="select"] input,
[data-baseweb="select"] [data-baseweb="select-content"] {
    color: var(--text) !important;
    background: transparent !important;
}

/* ===================================================================
   Radio — pill row.
   =================================================================== */
.stRadio > div {
    gap: 0.5rem;
    display: flex;
    flex-wrap: wrap;
}

.stRadio > div > label {
    background: var(--bg-2);
    padding: 0.55rem 0.95rem;
    border-radius: 8px;
    border: 1px solid var(--hairline-strong);
    transition: all 150ms ease;
    flex: 1;
    min-width: min-content;
    text-align: center;
    justify-content: center;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    cursor: pointer;
    color: var(--text-2) !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
    hyphens: none !important;
}

.stRadio > div > label *,
.stRadio [data-baseweb="radio"] p,
.stRadio [data-testid="stMarkdownContainer"] p {
    word-break: normal !important;
    overflow-wrap: normal !important;
    hyphens: none !important;
    white-space: normal !important;
}

.stRadio > div > label:hover {
    border-color: var(--teal-dim);
    color: var(--text) !important;
}

.stRadio > div > label:has(input:checked) {
    background: var(--teal-glow) !important;
    border-color: var(--teal) !important;
    color: var(--teal) !important;
}
.stRadio > div > label:has(input:checked) * {
    color: var(--teal) !important;
}

/* ===================================================================
   Expanders.
   =================================================================== */
.stExpander,
[data-testid="stExpander"] {
    border: 1px solid var(--hairline-strong) !important;
    border-radius: 10px !important;
    background: var(--bg-2) !important;
    box-shadow: none !important;
}

.stExpander summary,
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
    padding: 0.75rem 1rem !important;
}

.stExpander summary:hover {
    color: var(--teal) !important;
}

/* ===================================================================
   Messages.
   =================================================================== */
.stSuccess {
    background: rgba(0, 220, 195, 0.08) !important;
    border: 1px solid var(--teal-dim) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stInfo {
    background: var(--bg-2) !important;
    border: 1px solid var(--hairline-strong) !important;
    border-left: 3px solid var(--teal) !important;
    border-radius: 8px !important;
    color: var(--text-2) !important;
}
.stWarning {
    background: rgba(255, 140, 60, 0.06) !important;
    border: 1px solid var(--orange-dim) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stError {
    background: rgba(255, 95, 87, 0.06) !important;
    border: 1px solid #ff5f57 !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ===================================================================
   Metric cards.
   =================================================================== */
.stMetric, [data-testid="stMetric"] {
    background: var(--bg-2) !important;
    padding: 1rem 1.1rem !important;
    border-radius: 10px !important;
    border: 1px solid var(--hairline-strong) !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
    color: var(--text-3) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 2rem !important;
    line-height: 1.05 !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    color: var(--teal) !important;
}

/* ===================================================================
   Dataframe.
   =================================================================== */
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid var(--hairline-strong) !important;
}

.stDataFrame thead th {
    background: var(--bg-3) !important;
    color: var(--text-2) !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid var(--hairline-strong) !important;
}

.stDataFrame tbody td {
    background: var(--bg-2) !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--hairline) !important;
}

/* ===================================================================
   Progress & slider.
   =================================================================== */
.stProgress > div > div {
    background: var(--teal) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: var(--bg-3) !important;
    border-radius: 4px !important;
}

.stSlider [role="slider"] {
    background: var(--teal) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 0 1px var(--teal-dim);
    border-radius: 50% !important;
}
.stSlider > div > div > div > div {
    background: var(--teal) !important;
}

/* ===================================================================
   Dividers.
   =================================================================== */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--hairline-strong) !important;
    margin: 1.75rem 0 !important;
}

/* ===================================================================
   Code blocks — teal-tinted dark.
   =================================================================== */
.stCodeBlock, pre {
    border-radius: 10px !important;
    border: 1px solid var(--hairline-strong) !important;
    background: #0F0F0F !important;
}

.stCodeBlock code, pre code {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ===================================================================
   Tabs.
   =================================================================== */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--hairline-strong) !important;
    gap: 0 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.1rem !important;
    border-radius: 0 !important;
    color: var(--text-2) !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ===================================================================
   Sidebar.
   =================================================================== */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--hairline-strong) !important;
}

/* Collapse Streamlit's generous default top padding in the sidebar so the
   wordmark sits close to the top edge. */
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
[data-testid="stSidebar"] [data-testid="stSidebarNav"] + div {
    padding-top: 0.25rem !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
    gap: 0.6rem !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-child > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Also collapse the global block-container top padding on the main column
   so the layout reads tighter overall. */
[data-testid="stSidebar"] .stMarkdown:first-child,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:first-child {
    margin-top: 0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    font-size: 1.2rem !important;
    color: var(--text) !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
    color: var(--text-3) !important;
    font-weight: 400 !important;
}

/* ===================================================================
   Links.
   =================================================================== */
a, a:visited {
    color: var(--teal) !important;
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 3px;
    text-decoration-color: var(--teal-dim);
    transition: color 140ms ease, text-decoration-color 140ms ease;
}

a:hover {
    color: var(--teal) !important;
    text-decoration-color: var(--teal) !important;
}

/* ===================================================================
   Status & toast.
   =================================================================== */
.stStatus, [data-testid="stStatusWidget"] {
    border-radius: 10px !important;
    border: 1px solid var(--hairline-strong) !important;
    background: var(--bg-2) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: var(--bg);
}
::-webkit-scrollbar-thumb {
    background: var(--bg-4);
    border-radius: 5px;
    border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-3);
}

footer { visibility: hidden; }

/* ===================================================================
   Utility classes.
   =================================================================== */
.clm-kicker {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-3);
}
</style>
"""
