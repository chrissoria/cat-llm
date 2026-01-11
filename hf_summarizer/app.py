"""
Streamlit app - CatLLM Survey Response Summarizer
Based on the classifier app but focused on text/PDF summarization
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import time
import sys
from datetime import datetime

# Import catllm
try:
    import catllm
    CATLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import catllm: {e}")
    CATLLM_AVAILABLE = False

MAX_FILE_SIZE_MB = 100

def count_pdf_pages(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception:
        return 1  # Default to 1 if can't read


# Free models - display name -> actual API model name
FREE_MODELS_MAP = {
    "Qwen3 235B": "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "DeepSeek V3.1": "deepseek-ai/DeepSeek-V3.1:novita",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct:groq",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "GPT-4o Mini": "gpt-4o-mini",
    "Mistral Medium": "mistral-medium-2505",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Grok 4 Fast": "grok-4-fast-non-reasoning",
}
FREE_MODEL_DISPLAY_NAMES = list(FREE_MODELS_MAP.keys())

# Paid models (user provides their own API key)
PAID_MODEL_CHOICES = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "mistral-large-latest",
]

# Models routed through HuggingFace
HF_ROUTED_MODELS = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-3.3-70B-Instruct:groq",
]


def is_free_model(model, model_tier):
    """Check if using free tier (Space pays for API)."""
    return model_tier == "Free Models"


def get_model_source(model):
    """Auto-detect model source."""
    model_lower = model.lower()
    if "gpt" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    elif "mistral" in model_lower and ":novita" not in model_lower:
        return "mistral"
    elif any(x in model_lower for x in [":novita", ":groq", "qwen", "llama", "deepseek"]):
        return "huggingface"
    elif "sonar" in model_lower:
        return "perplexity"
    elif "grok" in model_lower:
        return "xai"
    return "huggingface"


def get_api_key(model, model_tier, api_key_input):
    """Get the appropriate API key based on model and tier."""
    if is_free_model(model, model_tier):
        if model in HF_ROUTED_MODELS:
            return os.environ.get("HF_API_KEY", ""), "HuggingFace"
        elif "gpt" in model.lower():
            return os.environ.get("OPENAI_API_KEY", ""), "OpenAI"
        elif "gemini" in model.lower():
            return os.environ.get("GOOGLE_API_KEY", ""), "Google"
        elif "mistral" in model.lower():
            return os.environ.get("MISTRAL_API_KEY", ""), "Mistral"
        elif "claude" in model.lower():
            return os.environ.get("ANTHROPIC_API_KEY", ""), "Anthropic"
        elif "sonar" in model.lower():
            return os.environ.get("PERPLEXITY_API_KEY", ""), "Perplexity"
        elif "grok" in model.lower():
            return os.environ.get("XAI_API_KEY", ""), "xAI"
        else:
            return os.environ.get("HF_API_KEY", ""), "HuggingFace"
    else:
        if api_key_input and api_key_input.strip():
            return api_key_input.strip(), "User"
        return "", "User"


def generate_summarize_code(input_type, description, model, model_source, focus=None, max_length=None, instructions=None, mode=None):
    """Generate Python code for summarization."""
    focus_param = f',\n    focus="{focus}"' if focus else ''
    length_param = f',\n    max_length={max_length}' if max_length else ''
    instructions_param = f',\n    instructions="{instructions}"' if instructions else ''

    if input_type == "text":
        return f'''import catllm
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Summarize the text column
result = catllm.summarize(
    input_data=df["your_column"].tolist(),
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}"{focus_param}{length_param}{instructions_param}
)

# View results
print(result)
result.to_csv("summarized_results.csv", index=False)
'''
    else:  # pdf
        mode_param = f',\n    mode="{mode}"' if mode else ''
        return f'''import catllm

# Summarize PDF documents
result = catllm.summarize(
    input_data="path/to/your/pdfs/",
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}"{mode_param}{focus_param}{length_param}{instructions_param}
)

# View results
print(result)
result.to_csv("summarized_results.csv", index=False)
'''


def generate_methodology_report_pdf(model, column_name, num_rows, model_source, filename, success_rate,
                          result_df=None, processing_time=None,
                          catllm_version=None, python_version=None,
                          input_type="text", description=None, focus=None, max_length=None):
    """Generate a PDF methodology report for summarization."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    pdf_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_methodology_report.pdf', delete=False)
    doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles['Normal']

    story = []

    report_title = "CatLLM Summarization Report"
    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("About This Report", heading_style))
    about_text = """This methodology report documents the automated summarization process. \
CatLLM uses LLMs to generate concise summaries of text or PDF documents, providing \
consistent and reproducible results."""
    story.append(Paragraph(about_text, normal_style))
    story.append(Spacer(1, 15))

    # Summary section
    story.append(Paragraph("Summarization Summary", heading_style))
    story.append(Spacer(1, 10))

    summary_data = [
        ["Source File", filename],
        ["Source Column/Type", column_name],
        ["Model Used", model],
        ["Model Source", model_source],
        ["Items Summarized", str(num_rows)],
        ["Success Rate", f"{success_rate:.2f}%"],
    ]
    if focus:
        summary_data.append(["Focus", focus])
    if max_length:
        summary_data.append(["Max Length", f"{max_length} words"])

    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    if processing_time is not None:
        story.append(Paragraph("Processing Time", heading_style))
        rows_per_min = (num_rows / processing_time) * 60 if processing_time > 0 else 0
        avg_time = processing_time / num_rows if num_rows > 0 else 0

        time_data = [
            ["Total Processing Time", f"{processing_time:.1f} seconds"],
            ["Average Time per Item", f"{avg_time:.2f} seconds"],
            ["Processing Rate", f"{rows_per_min:.1f} items/minute"],
        ]
        time_table = Table(time_data, colWidths=[180, 270])
        time_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(time_table)

    story.append(Spacer(1, 15))
    story.append(Paragraph("Version Information", heading_style))
    version_data = [
        ["CatLLM Version", catllm_version or "unknown"],
        ["Python Version", python_version or "unknown"],
        ["Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    ]
    version_table = Table(version_data, colWidths=[180, 270])
    version_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(version_table)

    story.append(Spacer(1, 30))
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316", normal_style))

    doc.build(story)
    return pdf_file.name


# Page config
st.set_page_config(
    page_title="CatLLM - Research Data Summarizer",
    page_icon="🐱",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = None
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None

# Logo and title
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.title("CatLLM - Research Data Summarizer")
    st.markdown("Generate concise summaries of survey responses and PDF documents using LLMs.")

# About section
with st.expander("About This App"):
    st.markdown("""
**Privacy Notice:** Your data is sent to third-party LLM APIs for summarization. Do not upload sensitive, confidential, or personally identifiable information (PII).

---

**CatLLM** is an open-source Python package for processing text and document data using Large Language Models.

### What It Does
- **Summarize Text**: Generate concise summaries of survey responses or text data
- **Summarize PDFs**: Extract key information from PDF documents page-by-page
- **Focus Summaries**: Guide the model to focus on specific aspects of your data

### Beta Test - We Want Your Feedback!
This app is currently in **beta** and **free to use** while CatLLM is under review for publication, made possible by **Bashir Ahmed's generous fellowship support**.

- Found a bug? Have a feature request? Please open an issue on [GitHub](https://github.com/chrissoria/cat-llm)
- Reach out directly: [chrissoria@berkeley.edu](mailto:chrissoria@berkeley.edu)

### Links
- **PyPI**: [pip install cat-llm](https://pypi.org/project/cat-llm/)
- **GitHub**: [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)
- **Classifier App**: [CatLLM Survey Classifier](https://huggingface.co/spaces/CatLLM/survey-classifier)

### Citation
If you use CatLLM in your research, please cite:
```
Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316
```
""")

# Main layout
col_input, col_output = st.columns([1, 1])

with col_input:
    # Input type selector
    input_type_choice = st.radio(
        "Input Type",
        options=["Survey Responses", "PDF Documents"],
        horizontal=True,
        key="input_type_radio"
    )

    # Initialize variables
    input_data = None
    input_type_selected = "text"
    description = ""
    original_filename = "data"
    pdf_mode = "Image (visual documents)"

    if input_type_choice == "Survey Responses":
        input_type_selected = "text"

        uploaded_file = st.file_uploader(
            "Upload Data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="survey_file"
        )

        if st.button("Try Example Dataset", key="example_btn"):
            st.session_state.example_loaded = True

        columns = []
        df = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                columns = df.columns.tolist()
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        elif hasattr(st.session_state, 'example_loaded') and st.session_state.example_loaded:
            try:
                df = pd.read_csv("example_data.csv")
                columns = df.columns.tolist()
                st.success(f"Loaded example dataset ({len(df)} rows)")
            except:
                pass

        selected_column = st.selectbox(
            "Column to Summarize",
            options=columns if columns else ["Upload a file first"],
            disabled=not columns,
            key="survey_column"
        )

        description = selected_column if columns else ""
        original_filename = uploaded_file.name if uploaded_file else "example_data.csv"

        if df is not None and columns and selected_column in columns:
            input_data = df[selected_column].tolist()

    else:  # PDF Documents
        input_type_selected = "pdf"

        pdf_files = st.file_uploader(
            "Upload PDF Document(s)",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_files"
        )

        pdf_description = st.text_input(
            "Document Description",
            placeholder="e.g., 'research papers', 'interview transcripts'",
            help="Helps the LLM understand context",
            key="pdf_desc"
        )

        pdf_mode = st.radio(
            "Processing Mode",
            options=["Image (visual documents)", "Text (text-heavy)", "Both (comprehensive)"],
            key="pdf_mode"
        )

        if pdf_files:
            input_data = []
            pdf_name_map = {}  # Map temp paths to original filenames
            for f in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(f.read())
                    input_data.append(tmp.name)
                    pdf_name_map[tmp.name] = f.name.replace('.pdf', '')
            st.session_state.pdf_name_map = pdf_name_map
            description = pdf_description or "document"
            original_filename = "pdf_files"
            st.success(f"Uploaded {len(pdf_files)} PDF file(s)")

    st.markdown("---")

    # Summarization options
    st.markdown("### Summarization Options")

    focus = st.text_input(
        "Focus (optional)",
        placeholder="e.g., 'main arguments', 'emotional content', 'key findings'",
        help="Guide the model to focus on specific aspects"
    )

    max_length = st.number_input(
        "Maximum Summary Length (words, optional)",
        min_value=0,
        max_value=1000,
        value=0,
        help="Leave at 0 for no limit"
    )
    max_length = max_length if max_length > 0 else None

    instructions = st.text_input(
        "Additional Instructions (optional)",
        placeholder="e.g., 'use bullet points', 'include quotes'",
        help="Custom instructions for the summarization"
    )

    st.markdown("---")

    # Model selection
    st.markdown("### Model Selection")
    model_tier = st.radio(
        "Model Tier",
        options=["Free Models", "Bring Your Own Key"],
        key="model_tier"
    )

    if model_tier == "Free Models":
        model_display = st.selectbox("Model", options=FREE_MODEL_DISPLAY_NAMES, key="model")
        model = FREE_MODELS_MAP[model_display]
        api_key = ""
    else:
        model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key="model_paid")
        api_key = st.text_input("API Key", type="password", key="api_key")

    # Summarize button
    if st.button("Summarize Data", type="primary", use_container_width=True):
        if input_data is None:
            st.error("Please upload data first")
        else:
            mode = None
            if input_type_selected == "pdf":
                mode_mapping = {
                    "Image (visual documents)": "image",
                    "Text (text-heavy)": "text",
                    "Both (comprehensive)": "both"
                }
                mode = mode_mapping.get(pdf_mode, "image")

            actual_api_key, provider = get_api_key(model, model_tier, api_key)
            if not actual_api_key:
                st.error(f"{provider} API key not configured")
            else:
                model_source = get_model_source(model)
                items_list = input_data if isinstance(input_data, list) else [input_data]

                # Calculate estimated time
                num_items = len(items_list)
                if input_type_selected == "pdf":
                    total_pages = sum(count_pdf_pages(p) for p in items_list)
                    est_seconds = total_pages * 5
                else:
                    est_seconds = max(10, num_items * 2)

                est_time_str = f"{est_seconds:.0f}s" if est_seconds < 60 else f"{est_seconds/60:.1f}m"

                # Progress UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                def progress_callback(current_idx, total, label=None):
                    progress = current_idx / total if total > 0 else 0
                    progress_bar.progress(min(progress, 1.0))

                    elapsed = time.time() - start_time
                    if current_idx > 0:
                        avg_time = elapsed / current_idx
                        eta_seconds = avg_time * (total - current_idx)
                        eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f" | ETA: {eta_seconds/60:.1f}m"
                    else:
                        eta_str = ""

                    label_str = f" ({label})" if label else ""
                    status_text.text(f"Processing item {current_idx+1} of {total}{label_str} ({progress*100:.0f}%){eta_str}")

                try:
                    # Build kwargs for summarize
                    summarize_kwargs = {
                        "input_data": items_list,
                        "api_key": actual_api_key,
                        "description": description,
                        "user_model": model,
                        "model_source": model_source,
                        "progress_callback": progress_callback,
                    }
                    if mode:
                        summarize_kwargs["mode"] = mode
                    if focus and focus.strip():
                        summarize_kwargs["focus"] = focus.strip()
                    if max_length:
                        summarize_kwargs["max_length"] = max_length
                    if instructions and instructions.strip():
                        summarize_kwargs["instructions"] = instructions.strip()

                    result_df = catllm.summarize(**summarize_kwargs)

                    processing_time = time.time() - start_time
                    total_items = len(result_df)
                    progress_bar.progress(1.0)
                    status_text.text(f"Completed {total_items} items in {processing_time:.1f}s")

                    # Replace temp paths with original filenames for PDF input
                    if input_type_selected == "pdf" and 'pdf_path' in result_df.columns:
                        pdf_name_map = st.session_state.get('pdf_name_map', {})
                        def replace_temp_path(val):
                            if pd.isna(val):
                                return val
                            val_str = str(val)
                            for temp_path, orig_name in pdf_name_map.items():
                                if temp_path in val_str:
                                    return val_str.replace(temp_path, orig_name + '.pdf')
                            return val_str
                        result_df['pdf_path'] = result_df['pdf_path'].apply(replace_temp_path)

                    # Save CSV
                    with tempfile.NamedTemporaryFile(mode='w', suffix='_summarized.csv', delete=False) as f:
                        result_df.to_csv(f.name, index=False)
                        csv_path = f.name

                    # Calculate success rate
                    if 'processing_status' in result_df.columns:
                        success_count = (result_df['processing_status'] == 'success').sum()
                        success_rate = (success_count / len(result_df)) * 100
                    else:
                        success_rate = 100.0

                    # Get version info
                    try:
                        catllm_version = catllm.__version__
                    except AttributeError:
                        catllm_version = "unknown"
                    python_version = sys.version.split()[0]

                    # Generate methodology report
                    pdf_path = generate_methodology_report_pdf(
                        model=model,
                        column_name=description,
                        num_rows=total_items,
                        model_source=model_source,
                        filename=original_filename,
                        success_rate=success_rate,
                        result_df=result_df,
                        processing_time=processing_time,
                        catllm_version=catllm_version,
                        python_version=python_version,
                        input_type=input_type_selected,
                        description=description,
                        focus=focus if focus else None,
                        max_length=max_length
                    )

                    # Generate code
                    code = generate_summarize_code(
                        input_type_selected, description, model, model_source,
                        focus=focus if focus else None,
                        max_length=max_length,
                        instructions=instructions if instructions else None,
                        mode=mode
                    )

                    st.session_state.results = {
                        'df': result_df,
                        'csv_path': csv_path,
                        'pdf_path': pdf_path,
                        'code': code,
                        'status': f"Summarized {total_items} items in {processing_time:.1f}s",
                    }
                    st.success(f"Summarized {total_items} items in {processing_time:.1f}s")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col_output:
    st.markdown("### Results")

    if st.session_state.results:
        results = st.session_state.results

        # Placeholder for future chart
        st.info("Summary visualization coming soon!")

        # Results dataframe
        display_df = results['df'].copy()
        cols_to_hide = ['model_response', 'json', 'raw_response', 'raw_json']
        display_df = display_df.drop(columns=[c for c in cols_to_hide if c in display_df.columns])
        st.dataframe(display_df, use_container_width=True)

        # Downloads
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open(results['csv_path'], 'rb') as f:
                st.download_button(
                    "Download Results (CSV)",
                    data=f,
                    file_name="summarized_results.csv",
                    mime="text/csv"
                )
        with col_dl2:
            with open(results['pdf_path'], 'rb') as f:
                st.download_button(
                    "Download Methodology Report (PDF)",
                    data=f,
                    file_name="methodology_report.pdf",
                    mime="application/pdf"
                )

        # Code
        with st.expander("See the Code"):
            st.code(results['code'], language='python')
    else:
        st.info("Upload data and click 'Summarize Data' to see results here.")

# Bottom buttons
col_reset, col_code = st.columns(2)
with col_reset:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.results = None
        if hasattr(st.session_state, 'example_loaded'):
            del st.session_state.example_loaded
        st.rerun()

with col_code:
    if st.session_state.results:
        if st.button("See in Code", use_container_width=True):
            st.session_state.show_code_modal = True

# Code modal/dialog
if st.session_state.get('show_code_modal') and st.session_state.results:
    st.markdown("---")
    st.markdown("### Reproducibility Code")
    st.markdown("Use this code to reproduce the summarization with the CatLLM Python package:")
    st.code(st.session_state.results['code'], language='python')
    if st.button("Close"):
        st.session_state.show_code_modal = False
        st.rerun()
