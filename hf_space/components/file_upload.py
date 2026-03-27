"""
File upload panels for CSV/Excel, PDF, and image inputs.
"""

import os
import tempfile
import pandas as pd
import streamlit as st


def count_pdf_pages(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception:
        return 1


def extract_text_from_pdfs(pdf_paths):
    """Extract text from all pages of all PDFs."""
    import fitz
    all_texts = []
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text().strip()
                if text:
                    all_texts.append(text)
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
    return all_texts


def extract_pdf_pages(pdf_paths, pdf_name_map, mode="image"):
    """Extract individual pages from PDFs as images or text."""
    import fitz
    pages = []
    for pdf_path in pdf_paths:
        orig_name = pdf_name_map.get(pdf_path, os.path.basename(pdf_path).replace(".pdf", ""))
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                page_label = f"{orig_name}_p{page_num}"
                if mode == "text":
                    text = page.get_text().strip()
                    if text:
                        pages.append((text, page_label, "text"))
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    pix.save(img_path)
                    if mode == "both":
                        text = page.get_text().strip()
                        pages.append((img_path, page_label, "image", text))
                    else:
                        pages.append((img_path, page_label, "image"))
            doc.close()
        except Exception as e:
            print(f"Error extracting pages from {pdf_path}: {e}")
    return pages


PDF_MODE_OPTIONS = ["Image (visual documents)", "Text (text-heavy)", "Both (comprehensive)"]
PDF_MODE_MAP = {
    "Image (visual documents)": "image",
    "Text (text-heavy)": "text",
    "Both (comprehensive)": "both",
}


def render_csv_upload(key_prefix=""):
    """Render CSV/Excel file upload panel.

    Returns:
        (input_data, description, original_filename, df) or (None, None, None, None).
    """
    upload_col, example_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key=f"{key_prefix}survey_file",
        )
    with example_col:
        st.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="tall-button">', unsafe_allow_html=True)
        if st.button("Try Example Dataset", key=f"{key_prefix}example_btn", use_container_width=True):
            st.session_state.example_loaded = True
        st.markdown("</div>", unsafe_allow_html=True)

    columns = []
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            columns = df.columns.tolist()
            st.success(f"Loaded {len(df):,} rows")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif st.session_state.get("example_loaded"):
        try:
            example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "example_data.csv")
            df = pd.read_csv(example_path)
            columns = df.columns.tolist()
            st.success(f"Loaded example dataset ({len(df)} rows)")
        except Exception:
            pass

    selected_column = st.selectbox(
        "Column to Process",
        options=columns if columns else ["Upload a file first"],
        disabled=not columns,
        key=f"{key_prefix}survey_column",
    )

    description = selected_column if columns else ""
    original_filename = uploaded_file.name if uploaded_file else "example_data.csv"

    if df is not None and columns and selected_column in columns:
        return df[selected_column].tolist(), description, original_filename, df

    return None, description, original_filename, df


def render_pdf_upload(key_prefix=""):
    """Render PDF upload panel.

    Returns:
        (input_data, description, original_filename, pdf_mode) or (None, ..., ..., ...).
    """
    pdf_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"{key_prefix}pdf_files",
    )

    pdf_description = st.text_input(
        "Document Description",
        placeholder="e.g., 'research papers', 'interview transcripts'",
        help="Helps the LLM understand context",
        key=f"{key_prefix}pdf_desc",
    )

    pdf_mode = st.radio(
        "Processing Mode",
        options=PDF_MODE_OPTIONS,
        key=f"{key_prefix}pdf_mode",
    )

    if pdf_files:
        input_data = []
        pdf_name_map = {}
        for f in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                input_data.append(tmp.name)
                pdf_name_map[tmp.name] = f.name.replace(".pdf", "")
        st.session_state.pdf_name_map = pdf_name_map
        description = pdf_description or "document"
        st.success(f"Uploaded {len(pdf_files)} PDF file(s)")
        return input_data, description, "pdf_files", PDF_MODE_MAP.get(pdf_mode, "image")

    return None, pdf_description or "document", "pdf_files", PDF_MODE_MAP.get(pdf_mode, "image")


def render_image_upload(key_prefix=""):
    """Render image upload panel.

    Returns:
        (input_data, description, original_filename) or (None, ..., ...).
    """
    image_files = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True,
        key=f"{key_prefix}image_files",
    )

    image_description = st.text_input(
        "Image Description",
        placeholder="e.g., 'product photos', 'social media posts'",
        help="Helps the LLM understand context",
        key=f"{key_prefix}image_desc",
    )

    if image_files:
        input_data = []
        for f in image_files:
            suffix = "." + f.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                input_data.append(tmp.name)
        description = image_description or "images"
        st.success(f"Uploaded {len(image_files)} image file(s)")
        return input_data, description, "image_files"

    return None, image_description or "images", "image_files"
