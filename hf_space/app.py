"""
Streamlit app - CatLLM Survey Response Classifier
Migrated from Gradio for better mobile support
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Import catllm
try:
    import catllm
    CATLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import catllm: {e}")
    CATLLM_AVAILABLE = False

MAX_CATEGORIES = 10
INITIAL_CATEGORIES = 3
MAX_FILE_SIZE_MB = 100

# Free models (uses Space secrets - no user API key needed)
FREE_MODEL_CHOICES = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-3.3-70B-Instruct:groq",
    "gemini-2.5-flash",
    "gpt-4o",
    "mistral-medium-2505",
    "claude-3-haiku-20240307",
    "grok-4-fast-non-reasoning",
]

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


def calculate_total_file_size(files):
    """Calculate total size of uploaded files in MB."""
    if files is None:
        return 0
    if not isinstance(files, list):
        files = [files]

    total_bytes = 0
    for f in files:
        try:
            if hasattr(f, 'size'):
                total_bytes += f.size
            elif hasattr(f, 'name'):
                total_bytes += os.path.getsize(f.name)
        except (OSError, AttributeError):
            pass
    return total_bytes / (1024 * 1024)


def generate_extract_code(input_type, description, model, model_source, max_categories, mode=None):
    """Generate Python code for category extraction."""
    if input_type == "text":
        return f'''import catllm
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Extract categories from the text column
result = catllm.extract(
    input_data=df["{description}"].tolist(),
    api_key="YOUR_API_KEY",
    input_type="text",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}",
    max_categories={max_categories}
)

# View extracted categories
print(result["top_categories"])
print(result["counts_df"])
'''
    elif input_type == "pdf":
        mode_line = f',\n    mode="{mode}"' if mode else ''
        return f'''import catllm

# Extract categories from PDF documents
result = catllm.extract(
    input_data="path/to/your/pdfs/",
    api_key="YOUR_API_KEY",
    input_type="pdf",
    description="{description}"{mode_line},
    user_model="{model}",
    model_source="{model_source}",
    max_categories={max_categories}
)

# View extracted categories
print(result["top_categories"])
print(result["counts_df"])
'''
    else:  # image
        return f'''import catllm

# Extract categories from images
result = catllm.extract(
    input_data="path/to/your/images/",
    api_key="YOUR_API_KEY",
    input_type="image",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}",
    max_categories={max_categories}
)

# View extracted categories
print(result["top_categories"])
print(result["counts_df"])
'''


def generate_classify_code(input_type, description, categories, model, model_source, mode=None):
    """Generate Python code for classification."""
    categories_str = ",\n    ".join([f'"{cat}"' for cat in categories])

    if input_type == "text":
        return f'''import catllm
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Define categories
categories = [
    {categories_str}
]

# Classify the text data
result = catllm.classify(
    input_data=df["{description}"].tolist(),
    categories=categories,
    api_key="YOUR_API_KEY",
    input_type="text",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''
    elif input_type == "pdf":
        mode_line = f',\n    mode="{mode}"' if mode else ''
        return f'''import catllm

# Define categories
categories = [
    {categories_str}
]

# Classify PDF documents
result = catllm.classify(
    input_data="path/to/your/pdfs/",
    categories=categories,
    api_key="YOUR_API_KEY",
    input_type="pdf",
    description="{description}"{mode_line},
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''
    else:  # image
        return f'''import catllm

# Define categories
categories = [
    {categories_str}
]

# Classify images
result = catllm.classify(
    input_data="path/to/your/images/",
    categories=categories,
    api_key="YOUR_API_KEY",
    input_type="image",
    description="{description}",
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''


def generate_methodology_report_pdf(categories, model, column_name, num_rows, model_source, filename, success_rate,
                          result_df=None, processing_time=None, prompt_template=None,
                          data_quality=None, catllm_version=None, python_version=None,
                          task_type="assign", extracted_categories_df=None, max_categories=None,
                          input_type="text", description=None):
    """Generate a PDF methodology report."""
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
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=9, leftIndent=20, spaceAfter=3)

    story = []

    if task_type == "extract_and_assign":
        report_title = "CatLLM Extraction &amp; Classification Report"
    else:
        report_title = "CatLLM Classification Report"

    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("About This Report", heading_style))

    if task_type == "extract_and_assign":
        about_text = """This methodology report documents the automated category extraction and classification process. \
CatLLM first discovers categories from your data using LLMs, then classifies each item into those categories."""
    else:
        about_text = """This methodology report documents the classification process for reproducibility and transparency. \
CatLLM restricts the prompt to a standard template that is impartial to the researcher's inclinations, ensuring \
consistent and reproducible results."""

    story.append(Paragraph(about_text, normal_style))
    story.append(Spacer(1, 15))

    if categories:
        story.append(Paragraph("Category Mapping", heading_style))
        story.append(Paragraph("Each category column contains binary values: 1 = present, 0 = not present", normal_style))
        story.append(Spacer(1, 8))

        category_data = [["Column Name", "Category Description"]]
        for i, cat in enumerate(categories, 1):
            category_data.append([f"category_{i}", cat])

        cat_table = Table(category_data, colWidths=[120, 330])
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(cat_table)
        story.append(Spacer(1, 15))

    story.append(Spacer(1, 30))
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316", normal_style))

    # Summary section
    story.append(PageBreak())
    story.append(Paragraph("Classification Summary", title_style))
    story.append(Spacer(1, 15))

    summary_data = [
        ["Source File", filename],
        ["Source Column", column_name],
        ["Model Used", model],
        ["Model Source", model_source],
        ["Rows Classified", str(num_rows)],
        ["Number of Categories", str(len(categories)) if categories else "0"],
        ["Success Rate", f"{success_rate:.2f}%"],
    ]
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
            ["Average Time per Response", f"{avg_time:.2f} seconds"],
            ["Processing Rate", f"{rows_per_min:.1f} rows/minute"],
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

    doc.build(story)
    return pdf_file.name


def run_auto_extract(input_type, input_data, description, max_categories_val,
                     model_tier, model, api_key_input, mode=None, progress_callback=None):
    """Extract categories from data."""
    if not CATLLM_AVAILABLE:
        return None, "catllm package not available"

    actual_api_key, provider = get_api_key(model, model_tier, api_key_input)
    if not actual_api_key:
        return None, f"{provider} API key not configured"

    model_source = get_model_source(model)

    try:
        if isinstance(input_data, list):
            num_items = len(input_data)
        else:
            num_items = 1

        if input_type == "image":
            divisions = min(3, max(1, num_items // 5))
            categories_per_chunk = 12
        else:
            divisions = max(1, num_items // 15)
            divisions = min(divisions, 5)
            chunk_size = num_items // max(1, divisions)
            categories_per_chunk = min(10, chunk_size - 1)

        extract_kwargs = {
            'input_data': input_data,
            'api_key': actual_api_key,
            'input_type': input_type,
            'description': description,
            'user_model': model,
            'model_source': model_source,
            'divisions': divisions,
            'categories_per_chunk': categories_per_chunk,
            'max_categories': int(max_categories_val)
        }
        if mode:
            extract_kwargs['mode'] = mode

        extract_result = catllm.extract(**extract_kwargs)
        categories = extract_result.get('top_categories', [])

        if not categories:
            return None, "No categories were extracted"

        return categories, f"Extracted {len(categories)} categories successfully!"

    except Exception as e:
        return None, f"Error: {str(e)}"


def run_classify_data(input_type, input_data, description, categories,
                      model_tier, model, api_key_input, mode=None,
                      original_filename="data", column_name="text",
                      progress_callback=None):
    """Classify data with user-provided categories."""
    if not CATLLM_AVAILABLE:
        return None, None, None, None, "catllm package not available"

    if not categories:
        return None, None, None, None, "Please enter at least one category"

    actual_api_key, provider = get_api_key(model, model_tier, api_key_input)
    if not actual_api_key:
        return None, None, None, None, f"{provider} API key not configured"

    model_source = get_model_source(model)

    try:
        start_time = time.time()

        classify_kwargs = {
            'input_data': input_data,
            'categories': categories,
            'api_key': actual_api_key,
            'input_type': input_type,
            'description': description,
            'user_model': model,
            'model_source': model_source
        }
        if mode:
            classify_kwargs['mode'] = mode

        result = catllm.classify(**classify_kwargs)

        processing_time = time.time() - start_time
        num_items = len(result)

        # Save CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='_classified.csv', delete=False) as f:
            result.to_csv(f.name, index=False)
            csv_path = f.name

        # Calculate success rate
        if 'processing_status' in result.columns:
            success_count = (result['processing_status'] == 'success').sum()
            success_rate = (success_count / len(result)) * 100
        else:
            success_rate = 100.0

        # Get version info
        try:
            catllm_version = catllm.__version__
        except AttributeError:
            catllm_version = "unknown"
        python_version = sys.version.split()[0]

        # Generate methodology report
        report_pdf_path = generate_methodology_report_pdf(
            categories=categories,
            model=model,
            column_name=column_name,
            num_rows=num_items,
            model_source=model_source,
            filename=original_filename,
            success_rate=success_rate,
            result_df=result,
            processing_time=processing_time,
            catllm_version=catllm_version,
            python_version=python_version,
            task_type="assign",
            input_type=input_type,
            description=description
        )

        # Generate reproducibility code
        code = generate_classify_code(input_type, description, categories, model, model_source, mode)

        return result, csv_path, report_pdf_path, code, f"Classified {num_items} items in {processing_time:.1f}s"

    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"


def create_distribution_chart(result_df, categories):
    """Create a bar chart showing category distribution."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(categories) * 0.8)))

    dist_data = []
    total_rows = len(result_df)
    for i, cat in enumerate(categories, 1):
        col_name = f"category_{i}"
        if col_name in result_df.columns:
            count = int(result_df[col_name].sum())
            pct = (count / total_rows) * 100 if total_rows > 0 else 0
            dist_data.append({"Category": cat, "Percentage": round(pct, 1)})

    categories_list = [d["Category"] for d in dist_data][::-1]
    percentages = [d["Percentage"] for d in dist_data][::-1]

    bars = ax.barh(categories_list, percentages, color='#2563eb')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage (%)', fontsize=11)
    ax.set_title('Category Distribution (%)', fontsize=14, fontweight='bold')

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{pct:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    return fig


# Page config
st.set_page_config(
    page_title="CatLLM - Research Data Classifier",
    page_icon="🐱",
    layout="wide"
)

# Initialize session state
if 'categories' not in st.session_state:
    st.session_state.categories = [''] * MAX_CATEGORIES
if 'category_count' not in st.session_state:
    st.session_state.category_count = INITIAL_CATEGORIES
if 'task_mode' not in st.session_state:
    st.session_state.task_mode = None
if 'extracted_categories' not in st.session_state:
    st.session_state.extracted_categories = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "survey"
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = None
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None

# Logo and title
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.title("CatLLM - Research Data Classifier")
    st.markdown("Extract categories from or classify text data, PDFs, and images using LLMs.")

# About section
with st.expander("About This App"):
    st.markdown("""
**Privacy Notice:** Your data is sent to third-party LLM APIs for classification. Do not upload sensitive, confidential, or personally identifiable information (PII).

---

**CatLLM** is an open-source Python package for classifying text and document data using Large Language Models.

### What It Does
- **Extract Categories**: Discover themes and categories in your data automatically
- **Assign Categories**: Classify data into your predefined categories
- **Extract & Assign**: Let CatLLM discover categories, then classify all your data

### Beta Test - We Want Your Feedback!
This app is currently in **beta** and **free to use** while CatLLM is under review for publication, made possible by **Bashir Ahmed's generous fellowship support**.

- Found a bug? Have a feature request? Please open an issue on [GitHub](https://github.com/chrissoria/cat-llm)
- Reach out directly: [chrissoria@berkeley.edu](mailto:chrissoria@berkeley.edu)

### Links
- **PyPI**: [pip install cat-llm](https://pypi.org/project/cat-llm/)
- **GitHub**: [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)

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
        options=["Survey Responses", "PDF Documents", "Images"],
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
            "Column to Process",
            options=columns if columns else ["Upload a file first"],
            disabled=not columns,
            key="survey_column"
        )

        description = selected_column if columns else ""
        original_filename = uploaded_file.name if uploaded_file else "example_data.csv"

        if df is not None and columns and selected_column in columns:
            input_data = df[selected_column].tolist()

    elif input_type_choice == "PDF Documents":
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
            for f in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(f.read())
                    input_data.append(tmp.name)
            description = pdf_description or "document"
            original_filename = "pdf_files"
            st.success(f"Uploaded {len(pdf_files)} PDF file(s)")

    else:  # Images
        input_type_selected = "image"

        image_files = st.file_uploader(
            "Upload Images",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            accept_multiple_files=True,
            key="image_files"
        )

        image_description = st.text_input(
            "Image Description",
            placeholder="e.g., 'product photos', 'social media posts'",
            help="Helps the LLM understand context",
            key="image_desc"
        )

        if image_files:
            input_data = []
            for f in image_files:
                suffix = '.' + f.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    input_data.append(tmp.name)
            description = image_description or "images"
            original_filename = "image_files"
            st.success(f"Uploaded {len(image_files)} image file(s)")

    st.markdown("---")

    # Task selection
    st.markdown("### What would you like to do?")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        manual_mode = st.button("Enter Categories Manually", use_container_width=True)
    with col_btn2:
        auto_mode = st.button("Auto-extract Categories", use_container_width=True)

    if manual_mode:
        st.session_state.task_mode = "manual"
    if auto_mode:
        st.session_state.task_mode = "auto_extract"

    # Auto-extract settings
    if st.session_state.task_mode == "auto_extract":
        st.markdown("### Auto-extract Categories")
        st.markdown("We'll analyze your data to discover the main categories.")

        max_categories = st.slider(
            "Number of Categories to Extract",
            min_value=3,
            max_value=25,
            value=12,
            help="How many categories should be identified in your data"
        )

        # Model selection for extraction
        st.markdown("### Model Selection")
        model_tier = st.radio(
            "Model Tier",
            options=["Free Models", "Bring Your Own Key"],
            key="extract_model_tier"
        )

        if model_tier == "Free Models":
            model = st.selectbox("Model", options=FREE_MODEL_CHOICES, key="extract_model")
            api_key = ""
            st.info("**Free tier** - no API key required!")
        else:
            model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key="extract_model_paid")
            api_key = st.text_input("API Key", type="password", key="extract_api_key")

        if st.button("Extract Categories", type="primary"):
            if input_data is None:
                st.error("Please upload data first")
            else:
                with st.spinner("Extracting categories..."):
                    mode = None
                    if input_type_selected == "pdf":
                        mode_mapping = {
                            "Image (visual documents)": "image",
                            "Text (text-heavy)": "text",
                            "Both (comprehensive)": "both"
                        }
                        mode = mode_mapping.get(pdf_mode, "image")

                    categories, status = run_auto_extract(
                        input_type_selected, input_data, description,
                        max_categories, model_tier, model, api_key, mode
                    )

                    if categories:
                        st.session_state.extracted_categories = categories
                        st.session_state.task_mode = "manual"  # Switch to manual to show categories
                        st.success(status)
                        st.rerun()
                    else:
                        st.error(status)

    # Category inputs (shown for manual mode or after extraction)
    if st.session_state.task_mode == "manual":
        st.markdown("### Categories")
        st.markdown("Enter your classification categories below.")

        # Pre-fill with extracted categories if available
        if st.session_state.extracted_categories:
            for i, cat in enumerate(st.session_state.extracted_categories[:MAX_CATEGORIES]):
                st.session_state.categories[i] = cat
            st.session_state.category_count = min(len(st.session_state.extracted_categories), MAX_CATEGORIES)
            st.session_state.extracted_categories = None  # Clear after use

        placeholder_examples = [
            "e.g., Positive sentiment",
            "e.g., Negative sentiment",
            "e.g., Product feedback",
            "e.g., Service complaint",
            "e.g., Feature request",
            "e.g., Custom category"
        ]

        categories_entered = []
        for i in range(st.session_state.category_count):
            placeholder = placeholder_examples[i] if i < len(placeholder_examples) else "e.g., Custom category"
            cat_value = st.text_input(
                f"Category {i+1}",
                value=st.session_state.categories[i],
                placeholder=placeholder,
                key=f"cat_{i}"
            )
            st.session_state.categories[i] = cat_value
            if cat_value.strip():
                categories_entered.append(cat_value.strip())

        if st.session_state.category_count < MAX_CATEGORIES:
            if st.button("+ Add More"):
                st.session_state.category_count += 1
                st.rerun()

        st.markdown("### Model Selection")
        model_tier = st.radio(
            "Model Tier",
            options=["Free Models", "Bring Your Own Key"],
            key="classify_model_tier"
        )

        if model_tier == "Free Models":
            model = st.selectbox("Model", options=FREE_MODEL_CHOICES, key="classify_model")
            api_key = ""
            st.info("**Free tier** - no API key required!")
        else:
            model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key="classify_model_paid")
            api_key = st.text_input("API Key", type="password", key="classify_api_key")

        if st.button("Classify Data", type="primary", use_container_width=True):
            if input_data is None:
                st.error("Please upload data first")
            elif not categories_entered:
                st.error("Please enter at least one category")
            else:
                with st.spinner("Classifying data... This may take a few minutes."):
                    mode = None
                    if input_type_selected == "pdf":
                        mode_mapping = {
                            "Image (visual documents)": "image",
                            "Text (text-heavy)": "text",
                            "Both (comprehensive)": "both"
                        }
                        mode = mode_mapping.get(pdf_mode, "image")

                    result_df, csv_path, pdf_path, code, status = run_classify_data(
                        input_type_selected, input_data, description,
                        categories_entered, model_tier, model, api_key, mode,
                        original_filename, description
                    )

                    if result_df is not None:
                        st.session_state.results = {
                            'df': result_df,
                            'csv_path': csv_path,
                            'pdf_path': pdf_path,
                            'code': code,
                            'status': status,
                            'categories': categories_entered
                        }
                        st.success(status)
                        st.rerun()
                    else:
                        st.error(status)

with col_output:
    st.markdown("### Results")

    if st.session_state.results:
        results = st.session_state.results

        # Distribution chart
        fig = create_distribution_chart(results['df'], results['categories'])
        st.pyplot(fig)

        # Results dataframe
        st.dataframe(results['df'], use_container_width=True)

        # Downloads
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open(results['csv_path'], 'rb') as f:
                st.download_button(
                    "Download Results (CSV)",
                    data=f,
                    file_name="classified_results.csv",
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
        st.info("Upload data, select categories, and click 'Classify Data' to see results here.")

# Reset button
if st.button("Reset", type="secondary"):
    st.session_state.categories = [''] * MAX_CATEGORIES
    st.session_state.category_count = INITIAL_CATEGORIES
    st.session_state.task_mode = None
    st.session_state.extracted_categories = None
    st.session_state.results = None
    if hasattr(st.session_state, 'example_loaded'):
        del st.session_state.example_loaded
    st.rerun()
