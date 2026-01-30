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


def extract_text_from_pdfs(pdf_paths):
    """Extract text from all pages of all PDFs, returning list of page texts."""
    import fitz  # PyMuPDF
    all_texts = []
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text().strip()
                if text:  # Only add non-empty pages
                    all_texts.append(text)
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
    return all_texts


def extract_pdf_pages(pdf_paths, pdf_name_map, mode="image"):
    """
    Extract individual pages from PDFs.
    Returns list of (page_data, page_label) tuples.
    For image mode: page_data is path to temp image file
    For text mode: page_data is extracted text
    """
    import fitz  # PyMuPDF
    pages = []

    for pdf_path in pdf_paths:
        orig_name = pdf_name_map.get(pdf_path, os.path.basename(pdf_path).replace('.pdf', ''))
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                page_label = f"{orig_name}_p{page_num}"

                if mode == "text":
                    # Extract text
                    text = page.get_text().strip()
                    if text:
                        pages.append((text, page_label, "text"))
                else:
                    # Render as image (for image or both mode)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                    img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
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

# Free models - display name -> actual API model name
FREE_MODELS_MAP = {
    "GPT-4o Mini": "gpt-4o-mini",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct:groq",
    "Qwen 2.5": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1:novita",
    "Mistral Medium": "mistral-medium-2505",
    "Grok 4 Fast": "grok-4-fast-non-reasoning",
}
FREE_MODEL_DISPLAY_NAMES = list(FREE_MODELS_MAP.keys())
FREE_MODEL_CHOICES = list(FREE_MODELS_MAP.values())  # Keep for backward compat

# Paid models (user provides their own API key)
PAID_MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "mistral-large-latest",
]

# Models routed through HuggingFace
HF_ROUTED_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct:groq",
    "deepseek-ai/DeepSeek-R1:novita",
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


def generate_full_code(extraction_params, classify_params):
    """Generate combined extract + classify code when categories were auto-extracted."""
    ext = extraction_params
    cls = classify_params

    # Determine input data placeholder
    if ext['input_type'] == "text":
        input_placeholder = 'df["your_column"].tolist()'
        load_data = '''import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")
'''
    elif ext['input_type'] == "pdf":
        input_placeholder = '"path/to/your/pdfs/"'
        load_data = ''
    else:
        input_placeholder = '"path/to/your/images/"'
        load_data = ''

    mode_param = f',\n    mode="{ext["mode"]}"' if ext.get('mode') else ''

    # Build extract code
    extract_code = f'''# Step 1: Extract categories from your data
extract_result = catllm.extract(
    input_data={input_placeholder},
    api_key="YOUR_API_KEY",
    description="{ext['description']}",
    user_model="{ext['model']}",
    max_categories={ext['max_categories']}{mode_param}
)

categories = extract_result["top_categories"]
print(f"Extracted {{len(categories)}} categories: {{categories}}")
'''

    # Build classify code based on mode
    if cls['classify_mode'] == "Single Model":
        classify_mode_param = f',\n    mode="{cls["mode"]}"' if cls.get('mode') and ext['input_type'] == "pdf" else ''
        classify_code = f'''
# Step 2: Classify data using extracted categories
result = catllm.classify(
    input_data={input_placeholder},
    categories=categories,
    api_key="YOUR_API_KEY",
    description="{cls['description']}",
    user_model="{cls['model']}"{classify_mode_param}
)'''
    else:
        # Multi-model mode
        models_str = ",\n        ".join([f'("{m}", "auto", "YOUR_API_KEY")' for m in cls['models_list']])
        classify_mode_param = f',\n    mode="{cls["mode"]}"' if cls.get('mode') and ext['input_type'] == "pdf" else ''
        threshold_str = "majority" if cls['consensus_threshold'] == 0.5 else "two-thirds" if cls['consensus_threshold'] == 0.67 else "unanimous"
        consensus_param = f',\n    consensus_threshold="{threshold_str}"' if cls['classify_mode'] == "Ensemble" else ''

        classify_code = f'''
# Step 2: Classify data using extracted categories with {"ensemble voting" if cls['classify_mode'] == "Ensemble" else "model comparison"}
models = [
        {models_str}
]

result = catllm.classify(
    input_data={input_placeholder},
    categories=categories,
    models=models,
    description="{cls['description']}"{classify_mode_param}{consensus_param}
)'''

    return f'''import catllm
{load_data}
{extract_code}
{classify_code}

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''


def generate_classify_code(input_type, description, categories, model, model_source, mode=None, classify_mode="Single Model", models_list=None, consensus_threshold=0.5):
    """Generate Python code for classification."""
    categories_str = ",\n    ".join([f'"{cat}"' for cat in categories])

    # Determine input data placeholder based on type
    if input_type == "text":
        input_placeholder = 'df["your_column"].tolist()'
        load_data = '''import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")
'''
    elif input_type == "pdf":
        input_placeholder = '"path/to/your/pdfs/"'
        load_data = ''
    else:  # image
        input_placeholder = '"path/to/your/images/"'
        load_data = ''

    # Generate code based on classification mode
    if classify_mode == "Single Model":
        # Single model mode
        mode_param = f',\n    mode="{mode}"' if mode and input_type == "pdf" else ''
        return f'''import catllm
{load_data}
# Define categories
categories = [
    {categories_str}
]

# Classify data (input type is auto-detected)
result = catllm.classify(
    input_data={input_placeholder},
    categories=categories,
    api_key="YOUR_API_KEY",
    description="{description}",
    user_model="{model}"{mode_param}
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''
    else:
        # Multi-model mode (Comparison or Ensemble)
        if models_list:
            models_str = ",\n        ".join([f'("{m}", "auto", "YOUR_API_KEY")' for m in models_list])
        else:
            models_str = '("gpt-4o", "auto", "YOUR_API_KEY"),\n        ("claude-sonnet-4-5-20250929", "auto", "YOUR_API_KEY")'

        mode_param = f',\n    mode="{mode}"' if mode and input_type == "pdf" else ''
        # Map numeric threshold back to string for cleaner code
        threshold_str = "majority" if consensus_threshold == 0.5 else "two-thirds" if consensus_threshold == 0.67 else "unanimous"
        consensus_param = f',\n    consensus_threshold="{threshold_str}"' if classify_mode == "Ensemble" else ''

        return f'''import catllm
{load_data}
# Define categories
categories = [
    {categories_str}
]

# Define models for {"ensemble voting" if classify_mode == "Ensemble" else "comparison"}
models = [
        {models_str}
]

# Classify with multiple models
result = catllm.classify(
    input_data={input_placeholder},
    categories=categories,
    models=models,
    description="{description}"{mode_param}{consensus_param}
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''


def generate_methodology_report_pdf(categories, model, column_name, num_rows, model_source, filename, success_rate,
                          result_df=None, processing_time=None, prompt_template=None,
                          data_quality=None, catllm_version=None, python_version=None,
                          task_type="assign", extracted_categories_df=None, max_categories=None,
                          input_type="text", description=None, classify_mode="Single Model",
                          models_list=None, code=None):
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
        ["Classification Mode", classify_mode],
        ["Model(s) Used", model],
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

    # Reproducibility Code section
    if code:
        story.append(PageBreak())
        story.append(Paragraph("Reproducibility Code", title_style))
        story.append(Paragraph("Use this Python code to reproduce the classification with the CatLLM package:", normal_style))
        story.append(Spacer(1, 10))

        # Split code into lines and add as code-formatted paragraphs
        for line in code.strip().split('\n'):
            # Escape special characters for reportlab
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if escaped_line.strip():
                story.append(Paragraph(escaped_line, code_style))
            else:
                story.append(Spacer(1, 6))

    # Visualizations section
    if result_df is not None and categories:
        from reportlab.platypus import Image
        import io

        # Distribution chart (new page)
        story.append(PageBreak())
        story.append(Paragraph("Category Distribution", title_style))
        try:
            fig1 = create_distribution_chart(result_df, categories, classify_mode, models_list)
            img_buffer1 = io.BytesIO()
            fig1.savefig(img_buffer1, format='png', dpi=150, bbox_inches='tight')
            img_buffer1.seek(0)
            plt.close(fig1)

            # Save to temp file for reportlab
            img_temp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img_temp1.write(img_buffer1.read())
            img_temp1.close()

            img1 = Image(img_temp1.name, width=450, height=250)
            story.append(img1)
            story.append(Spacer(1, 10))
            story.append(Paragraph("Note: Categories are not mutually exclusive—each item can belong to multiple categories.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Could not generate distribution chart: {str(e)}", normal_style))

        # Classification matrix (new page)
        story.append(PageBreak())
        story.append(Paragraph("Classification Matrix", title_style))
        try:
            fig2 = create_classification_heatmap(result_df, categories, classify_mode, models_list)
            img_buffer2 = io.BytesIO()
            fig2.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
            img_buffer2.seek(0)
            plt.close(fig2)

            # Save to temp file for reportlab
            img_temp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img_temp2.write(img_buffer2.read())
            img_temp2.close()

            img2 = Image(img_temp2.name, width=450, height=300)
            story.append(img2)
            story.append(Spacer(1, 10))
            story.append(Paragraph("Orange = category present, Black = not present. Each row represents one response.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Could not generate classification matrix: {str(e)}", normal_style))

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
            'survey_input': input_data,
            'categories': categories,
            'models': [(model, model_source, actual_api_key)],
            'input_description': description,
        }
        if mode:
            classify_kwargs['pdf_mode'] = mode

        result = catllm.classify_ensemble(**classify_kwargs)

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


def sanitize_model_name(model: str) -> str:
    """Convert model name to column-safe suffix (matches catllm logic)."""
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_').lower()
    return sanitized[:40]


def create_classification_heatmap(result_df, categories, classify_mode="Single Model", models_list=None):
    """Create a binary heatmap showing classification for each row.

    Args:
        result_df: DataFrame with classification results
        categories: List of category names
        classify_mode: "Single Model", "Model Comparison", or "Ensemble"
        models_list: List of model names (for multi-model modes)
    """
    import numpy as np

    total_rows = len(result_df)
    if total_rows == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Build the binary matrix based on classify_mode
    if classify_mode == "Ensemble":
        # Use consensus columns
        col_names = [f"category_{i}_consensus" for i in range(1, len(categories) + 1)]
    elif classify_mode == "Model Comparison" and models_list:
        # Use first model's columns for the heatmap
        sanitized = sanitize_model_name(models_list[0])
        col_names = [f"category_{i}_{sanitized}" for i in range(1, len(categories) + 1)]
    else:
        # Single model
        col_names = [f"category_{i}" for i in range(1, len(categories) + 1)]

    # Extract the binary matrix
    matrix_data = []
    for col in col_names:
        if col in result_df.columns:
            matrix_data.append(result_df[col].astype(int).values)
        else:
            matrix_data.append(np.zeros(total_rows, dtype=int))

    matrix = np.array(matrix_data).T  # Rows = responses, Cols = categories

    # Create figure with appropriate sizing
    fig_height = max(4, min(20, total_rows * 0.15))
    fig_width = max(8, len(categories) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create custom colormap: black (0) and orange (1) - CatLLM theme
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1a1a1a', '#E8A33C'])

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    # Set labels - remove y-axis numbers for cleaner look
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Categories', fontsize=11)
    ax.set_ylabel(f'Responses (n={total_rows})', fontsize=11)
    ax.set_yticks([])  # Remove y-axis tick marks

    title = 'Classification Matrix'
    if classify_mode == "Ensemble":
        title += ' (Ensemble Consensus)'
    elif classify_mode == "Model Comparison":
        title += f' ({models_list[0].split("/")[-1].split(":")[0][:20]})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a1a1a', edgecolor='white', label='Not Present'),
        Patch(facecolor='#E8A33C', edgecolor='white', label='Present')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    return fig


def create_distribution_chart(result_df, categories, classify_mode="Single Model", models_list=None):
    """Create a bar chart showing category distribution.

    Args:
        result_df: DataFrame with classification results
        categories: List of category names
        classify_mode: "Single Model", "Model Comparison", or "Ensemble"
        models_list: List of model names (for multi-model modes)
    """
    import numpy as np

    total_rows = len(result_df)
    if total_rows == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Define colors for different models
    model_colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#be185d', '#65a30d']

    if classify_mode == "Single Model":
        # Single model: use category_1, category_2, etc.
        fig, ax = plt.subplots(figsize=(10, max(4, len(categories) * 0.8)))

        dist_data = []
        for i, cat in enumerate(categories, 1):
            col_name = f"category_{i}"
            if col_name in result_df.columns:
                count = int(result_df[col_name].sum())
                pct = (count / total_rows) * 100
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

    elif classify_mode == "Ensemble":
        # Ensemble: use category_1_consensus, category_2_consensus, etc.
        fig, ax = plt.subplots(figsize=(10, max(4, len(categories) * 0.8)))

        dist_data = []
        for i, cat in enumerate(categories, 1):
            col_name = f"category_{i}_consensus"
            if col_name in result_df.columns:
                count = int(result_df[col_name].sum())
                pct = (count / total_rows) * 100
                dist_data.append({"Category": cat, "Percentage": round(pct, 1)})

        categories_list = [d["Category"] for d in dist_data][::-1]
        percentages = [d["Percentage"] for d in dist_data][::-1]

        bars = ax.barh(categories_list, percentages, color='#16a34a')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_title('Ensemble Consensus Distribution (%)', fontsize=14, fontweight='bold')

        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=10)

    else:  # Model Comparison
        # Model Comparison: grouped bars for each model
        if not models_list:
            models_list = []

        sanitized_names = [sanitize_model_name(m) for m in models_list]
        n_models = len(sanitized_names)
        n_categories = len(categories)

        fig, ax = plt.subplots(figsize=(12, max(5, n_categories * 1.2)))

        # Gather data for each model
        bar_height = 0.8 / n_models
        y_positions = np.arange(n_categories)

        for model_idx, (model_name, sanitized) in enumerate(zip(models_list, sanitized_names)):
            model_pcts = []
            for i in range(1, n_categories + 1):
                col_name = f"category_{i}_{sanitized}"
                if col_name in result_df.columns:
                    count = int(result_df[col_name].sum())
                    pct = (count / total_rows) * 100
                else:
                    pct = 0
                model_pcts.append(pct)

            # Reverse for horizontal bar chart
            model_pcts = model_pcts[::-1]
            offset = (model_idx - n_models / 2 + 0.5) * bar_height
            color = model_colors[model_idx % len(model_colors)]

            # Use shorter display name
            display_name = model_name.split('/')[-1].split(':')[0][:20]
            bars = ax.barh(y_positions + offset, model_pcts, bar_height * 0.9,
                          label=display_name, color=color, alpha=0.85)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories[::-1])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_title('Category Distribution by Model (%)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig


# Page config
st.set_page_config(
    page_title="CatLLM - Research Data Classifier",
    page_icon="🐱",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
/* Import Garamond font and apply globally */
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');

* {
    font-family: 'EB Garamond', Garamond, Georgia, serif !important;
    font-size: 17px !important;
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
</style>
""", unsafe_allow_html=True)

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
if 'extraction_params' not in st.session_state:
    st.session_state.extraction_params = None  # Stores params when categories are auto-extracted

# Logo and title - use HTML for better alignment
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 10px;">
    <img src="https://huggingface.co/spaces/CatLLM/survey-classifier/resolve/main/logo.png" width="100" style="border-radius: 8px;">
    <div>
        <div style="font-size: 2.2rem; font-weight: 700; color: #333; font-family: 'EB Garamond', Garamond, Georgia, serif; line-height: 1.1;">CatLLM</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #E8A33C; font-family: 'EB Garamond', Garamond, Georgia, serif; margin-bottom: 4px;">NLP for Survey Research</div>
        <div style="font-size: 1rem; font-weight: 400; color: #666; font-family: 'EB Garamond', Garamond, Georgia, serif;">Research-grade categorization of survey responses, PDFs, and images using AI models.</div>
        <div style="font-size: 0.85rem; font-weight: 400; color: #888; font-family: 'EB Garamond', Garamond, Georgia, serif; margin-top: 4px;">Developed at UC Berkeley</div>
    </div>
</div>
""", unsafe_allow_html=True)

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

        upload_col, example_col = st.columns([3, 1])
        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload Data (CSV or Excel)",
                type=['csv', 'xlsx', 'xls'],
                key="survey_file"
            )
        with example_col:
            st.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)  # Match "Upload Data" label height
            st.markdown('<div class="tall-button">', unsafe_allow_html=True)
            if st.button("Try Example Dataset", key="example_btn", use_container_width=True):
                st.session_state.example_loaded = True
            st.markdown('</div>', unsafe_allow_html=True)

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
            pdf_name_map = {}  # Map temp paths to original filenames
            for f in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(f.read())
                    input_data.append(tmp.name)
                    pdf_name_map[tmp.name] = f.name.replace('.pdf', '')  # Store original name without extension
            st.session_state.pdf_name_map = pdf_name_map
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

        specificity = st.selectbox(
            "How specific should categories be?",
            options=["Broad", "Moderate", "Narrow"],
            index=0,
            help="Broad = general themes, Moderate = balanced detail, Narrow = highly specific categories"
        )

        focus = st.text_input(
            "What should categories be focused around? (optional)",
            placeholder="e.g., 'decisions to move', 'emotional responses', 'financial factors'",
            help="Guide the model to prioritize extracting categories related to this focus"
        )

        # Model selection for extraction
        st.markdown("### Model Selection")
        model_tier = st.radio(
            "Model Tier",
            options=["Free Models", "Bring Your Own Key"],
            key="extract_model_tier"
        )

        if model_tier == "Free Models":
            model_display = st.selectbox("Model", options=FREE_MODEL_DISPLAY_NAMES, key="extract_model")
            model = FREE_MODELS_MAP[model_display]  # Convert to actual model name
            api_key = ""
        else:
            model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key="extract_model_paid")
            api_key = st.text_input("API Key", type="password", key="extract_api_key")

        if st.button("Extract Categories", type="primary"):
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

                    # Calculate estimated time based on input size
                    num_items = len(input_data) if isinstance(input_data, list) else 1
                    if input_type_selected == "pdf":
                        # PDFs take longer - estimate ~5s per page
                        total_pages = sum(count_pdf_pages(p) for p in (input_data if isinstance(input_data, list) else [input_data]))
                        est_seconds = total_pages * 5
                    elif input_type_selected == "image":
                        # Images ~4s each
                        est_seconds = num_items * 4
                    else:
                        # Text ~2s per item, but batched
                        est_seconds = max(10, num_items * 0.5)

                    # Progress tracking UI
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    # Progress callback for extraction
                    def extract_progress_callback(current_step, total_steps, step_label):
                        progress = current_step / total_steps if total_steps > 0 else 0
                        progress_bar.progress(min(progress, 1.0))

                        elapsed = time.time() - start_time
                        if current_step > 0:
                            avg_time = elapsed / current_step
                            eta_seconds = avg_time * (total_steps - current_step)
                            eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f" | ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_str = ""

                        status_text.text(f"Extracting categories: {step_label} ({progress*100:.0f}%){eta_str}")

                    extract_kwargs = {
                        'input_data': input_data,
                        'api_key': actual_api_key,
                        'input_type': input_type_selected,
                        'description': description,
                        'user_model': model,
                        'model_source': model_source,
                        'max_categories': int(max_categories),
                        'specificity': specificity.lower(),
                        'progress_callback': extract_progress_callback,
                    }
                    if mode:
                        extract_kwargs['mode'] = mode
                    if focus and focus.strip():
                        extract_kwargs['focus'] = focus.strip()

                    try:
                        extract_result = catllm.extract(**extract_kwargs)
                        categories = extract_result.get('top_categories', [])

                        processing_time = time.time() - start_time
                        progress_bar.progress(1.0)
                        status_text.text(f"Completed in {processing_time:.1f}s")

                        if categories:
                            st.success(f"Extracted {len(categories)} categories in {processing_time:.1f}s")
                            st.session_state.extracted_categories = categories
                            # Store extraction params for code generation
                            st.session_state.extraction_params = {
                                'model': model,
                                'model_source': model_source,
                                'max_categories': int(max_categories),
                                'input_type': input_type_selected,
                                'description': description,
                                'mode': mode,
                            }
                            st.session_state.task_mode = "manual"
                            st.rerun()
                        else:
                            st.error("No categories were extracted from the data")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

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

        # Classification mode selector
        classify_mode = st.radio(
            "Classification Mode",
            options=["Single Model", "Model Comparison", "Ensemble"],
            horizontal=True,
            key="classify_mode",
            help="Single: one model. Comparison: see results from multiple models side-by-side. Ensemble: multiple models vote for consensus."
        )

        model_tier = st.radio(
            "Model Tier",
            options=["Free Models", "Bring Your Own Key"],
            key="classify_model_tier"
        )

        # Multi-model mode uses multiselect
        is_multi_model = classify_mode in ["Model Comparison", "Ensemble"]
        min_models = 3 if classify_mode == "Ensemble" else 2

        if model_tier == "Free Models":
            if is_multi_model:
                default_models = FREE_MODEL_DISPLAY_NAMES[:min_models] if len(FREE_MODEL_DISPLAY_NAMES) >= min_models else FREE_MODEL_DISPLAY_NAMES
                model_displays = st.multiselect(
                    f"Models (select {min_models}+)",
                    options=FREE_MODEL_DISPLAY_NAMES,
                    default=default_models,
                    key="classify_models_multi"
                )
                models_list = [FREE_MODELS_MAP[d] for d in model_displays]
            else:
                model_display = st.selectbox("Model", options=FREE_MODEL_DISPLAY_NAMES, key="classify_model")
                model = FREE_MODELS_MAP[model_display]  # Convert to actual model name
                models_list = [model]
            api_key = ""
        else:
            if is_multi_model:
                default_models = PAID_MODEL_CHOICES[:min_models] if len(PAID_MODEL_CHOICES) >= min_models else PAID_MODEL_CHOICES
                models_list = st.multiselect(
                    f"Models (select {min_models}+)",
                    options=PAID_MODEL_CHOICES,
                    default=default_models,
                    key="classify_models_multi_paid"
                )
            else:
                model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key="classify_model_paid")
                models_list = [model]
            api_key = st.text_input("API Key", type="password", key="classify_api_key")

        # Ensemble-specific options
        consensus_threshold = 0.5  # Default
        if classify_mode == "Ensemble":
            consensus_options = {
                "Majority (50%+)": 0.5,
                "Two-Thirds (67%+)": 0.67,
                "Unanimous (100%)": 1.0,
            }
            consensus_choice = st.radio(
                "Consensus Rule",
                options=list(consensus_options.keys()),
                horizontal=True,
                key="consensus_choice",
                help="How many models must agree for a category to be marked present"
            )
            consensus_threshold = consensus_options[consensus_choice]

        if st.button("Categorize Data", type="primary", use_container_width=True):
            if input_data is None:
                st.error("Please upload data first")
            elif not categories_entered:
                st.error("Please enter at least one category")
            elif classify_mode == "Model Comparison" and len(models_list) < 2:
                st.error("Please select at least 2 models for comparison mode")
            elif classify_mode == "Ensemble" and len(models_list) < 3:
                st.error("Please select at least 3 models for ensemble mode (needed for majority voting)")
            else:
                # Set up progress tracking
                mode = None
                if input_type_selected == "pdf":
                    mode_mapping = {
                        "Image (visual documents)": "image",
                        "Text (text-heavy)": "text",
                        "Both (comprehensive)": "both"
                    }
                    mode = mode_mapping.get(pdf_mode, "image")

                # Build models tuples list: [(model, source, api_key), ...]
                models_tuples = []
                api_key_error = None
                for m in models_list:
                    actual_key, provider = get_api_key(m, model_tier, api_key)
                    if not actual_key:
                        api_key_error = f"{provider} API key not configured for {m}"
                        break
                    m_source = get_model_source(m)
                    models_tuples.append((m, m_source, actual_key))

                if api_key_error:
                    st.error(api_key_error)
                else:
                    items_list = input_data if isinstance(input_data, list) else [input_data]

                    # Progress UI
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    # For PDFs, use progress callback
                    if input_type_selected == "pdf":
                        # Progress callback for PDF page-by-page updates
                        def pdf_progress_callback(current_idx, total_pages, page_label):
                            progress = current_idx / total_pages if total_pages > 0 else 0
                            progress_bar.progress(min(progress, 1.0))

                            elapsed = time.time() - start_time
                            if current_idx > 0:
                                avg_time = elapsed / current_idx
                                eta_seconds = avg_time * (total_pages - current_idx)
                                eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f" | ETA: {eta_seconds/60:.1f}m"
                            else:
                                eta_str = ""

                            status_text.text(f"Processing page {current_idx+1} of {total_pages} ({page_label}) ({progress*100:.0f}%){eta_str}")

                        try:
                            # Build kwargs for classify_ensemble
                            classify_kwargs = {
                                "survey_input": items_list,
                                "categories": categories_entered,
                                "models": models_tuples,
                                "input_description": description,
                                "pdf_mode": mode,
                                "progress_callback": pdf_progress_callback,
                            }
                            # Add consensus_threshold for ensemble mode
                            if classify_mode == "Ensemble":
                                classify_kwargs["consensus_threshold"] = consensus_threshold

                            result_df = catllm.classify_ensemble(**classify_kwargs)

                            processing_time = time.time() - start_time
                            total_items = len(result_df)
                            progress_bar.progress(1.0)
                            status_text.text(f"Completed {total_items} pages in {processing_time:.1f}s")

                            # Replace temp paths with original filenames in pdf_input column
                            if 'pdf_input' in result_df.columns:
                                pdf_name_map = st.session_state.get('pdf_name_map', {})
                                def replace_temp_path(val):
                                    if pd.isna(val):
                                        return val
                                    val_str = str(val)
                                    for temp_path, orig_name in pdf_name_map.items():
                                        # Check if the temp path's filename (without extension) is in the value
                                        temp_name = os.path.basename(temp_path).replace('.pdf', '')
                                        if temp_name in val_str:
                                            return val_str.replace(temp_name, orig_name)
                                    return val_str
                                result_df['pdf_input'] = result_df['pdf_input'].apply(replace_temp_path)

                            all_results = [result_df]

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            all_results = []

                    else:
                        # Non-PDF processing (text, images) - process all at once
                        total_items = len(items_list)

                        # Progress callback for item-by-item updates
                        def item_progress_callback(current_idx, total, item_label):
                            progress = current_idx / total if total > 0 else 0
                            progress_bar.progress(min(progress, 1.0))

                            elapsed = time.time() - start_time
                            if current_idx > 0:
                                avg_time = elapsed / current_idx
                                eta_seconds = avg_time * (total - current_idx)
                                eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f" | ETA: {eta_seconds/60:.1f}m"
                            else:
                                eta_str = ""

                            status_text.text(f"Processing item {current_idx+1} of {total} ({progress*100:.0f}%){eta_str}")

                        try:
                            # Build kwargs for classify_ensemble
                            classify_kwargs = {
                                "survey_input": items_list,
                                "categories": categories_entered,
                                "models": models_tuples,
                                "input_description": description,
                                "progress_callback": item_progress_callback,
                            }
                            # Add consensus_threshold for ensemble mode
                            if classify_mode == "Ensemble":
                                classify_kwargs["consensus_threshold"] = consensus_threshold

                            result_df = catllm.classify_ensemble(**classify_kwargs)
                            all_results = [result_df]

                            processing_time = time.time() - start_time
                            progress_bar.progress(1.0)
                            status_text.text(f"Completed {total_items} items in {processing_time:.1f}s")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            all_results = []
                            processing_time = time.time() - start_time

                    if all_results:
                        # Combine results
                        result_df = pd.concat(all_results, ignore_index=True)

                        # Save CSV
                        with tempfile.NamedTemporaryFile(mode='w', suffix='_classified.csv', delete=False) as f:
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

                        # For reports: create model string (single or list)
                        if len(models_list) == 1:
                            report_model = models_list[0]
                            report_model_source = models_tuples[0][1]
                        else:
                            report_model = ", ".join(models_list)
                            report_model_source = f"{classify_mode} ({len(models_list)} models)"

                        # Generate code first so we can include it in the PDF
                        # If categories were auto-extracted, include both extract and classify code
                        if st.session_state.extraction_params:
                            classify_params = {
                                'model': report_model,
                                'description': description,
                                'mode': mode,
                                'classify_mode': classify_mode,
                                'models_list': models_list,
                                'consensus_threshold': consensus_threshold,
                            }
                            code = generate_full_code(st.session_state.extraction_params, classify_params)
                        else:
                            code = generate_classify_code(
                                input_type_selected, description, categories_entered,
                                report_model, report_model_source, mode,
                                classify_mode=classify_mode, models_list=models_list,
                                consensus_threshold=consensus_threshold
                            )

                        # Generate methodology report with code included
                        pdf_path = generate_methodology_report_pdf(
                            categories=categories_entered,
                            model=report_model,
                            column_name=description,
                            num_rows=len(result_df),
                            model_source=report_model_source,
                            filename=original_filename,
                            success_rate=success_rate,
                            result_df=result_df,
                            processing_time=processing_time,
                            catllm_version=catllm_version,
                            python_version=python_version,
                            task_type="assign",
                            input_type=input_type_selected,
                            description=description,
                            classify_mode=classify_mode,
                            models_list=models_list,
                            code=code
                        )

                        st.session_state.results = {
                            'df': result_df,
                            'csv_path': csv_path,
                            'pdf_path': pdf_path,
                            'code': code,
                            'status': f"Classified {len(result_df)} items in {processing_time:.1f}s",
                            'categories': categories_entered,
                            'classify_mode': classify_mode,
                            'models_list': models_list,
                        }
                        st.success(f"Classified {len(result_df)} items in {processing_time:.1f}s")
                        st.rerun()
                    else:
                        st.error("No items were successfully classified")

with col_output:
    st.markdown("### Results")

    if st.session_state.results:
        results = st.session_state.results

        # Visualization selector
        viz_type = st.selectbox(
            "Visualization",
            options=["Category Distribution", "Classification Matrix"],
            key="viz_type",
            help="Distribution shows category percentages. Matrix shows each response's classifications."
        )

        if viz_type == "Category Distribution":
            fig = create_distribution_chart(
                results['df'],
                results['categories'],
                classify_mode=results.get('classify_mode', 'Single Model'),
                models_list=results.get('models_list', [])
            )
            st.pyplot(fig)
            st.caption("Note: Categories are not mutually exclusive—each item can belong to multiple categories.")
        else:
            fig = create_classification_heatmap(
                results['df'],
                results['categories'],
                classify_mode=results.get('classify_mode', 'Single Model'),
                models_list=results.get('models_list', [])
            )
            st.pyplot(fig)
            st.caption("Green = category present, Black = not present. Each row is one response.")

        # Results dataframe (hide technical columns from display)
        display_df = results['df'].copy()
        cols_to_hide = ['model_response', 'json', 'raw_response', 'raw_json']
        display_df = display_df.drop(columns=[c for c in cols_to_hide if c in display_df.columns])
        st.dataframe(display_df, use_container_width=True)

        # Downloads
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            with open(results['csv_path'], 'rb') as f:
                st.download_button(
                    "Download CSV",
                    data=f,
                    file_name="classified_results.csv",
                    mime="text/csv"
                )
        with col_dl2:
            with open(results['pdf_path'], 'rb') as f:
                st.download_button(
                    "Download Report",
                    data=f,
                    file_name="methodology_report.pdf",
                    mime="application/pdf"
                )
        with col_dl3:
            # Generate both plots and save to a single PDF
            import io
            from matplotlib.backends.backend_pdf import PdfPages

            plot_buffer = io.BytesIO()
            with PdfPages(plot_buffer) as pdf:
                # Distribution chart
                fig1 = create_distribution_chart(
                    results['df'],
                    results['categories'],
                    classify_mode=results.get('classify_mode', 'Single Model'),
                    models_list=results.get('models_list', [])
                )
                pdf.savefig(fig1, bbox_inches='tight')
                plt.close(fig1)

                # Classification matrix
                fig2 = create_classification_heatmap(
                    results['df'],
                    results['categories'],
                    classify_mode=results.get('classify_mode', 'Single Model'),
                    models_list=results.get('models_list', [])
                )
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)

            plot_buffer.seek(0)
            st.download_button(
                "Download Plots",
                data=plot_buffer,
                file_name="classification_plots.pdf",
                mime="application/pdf"
            )

        # Code
        with st.expander("See the Code"):
            st.code(results['code'], language='python')
    else:
        st.info("Upload data, select categories, and click 'Categorize Data' to see results here.")

# Bottom buttons
col_reset, col_code = st.columns(2)
with col_reset:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.categories = [''] * MAX_CATEGORIES
        st.session_state.category_count = INITIAL_CATEGORIES
        st.session_state.task_mode = None
        st.session_state.extracted_categories = None
        st.session_state.extraction_params = None
        st.session_state.results = None
        if hasattr(st.session_state, 'example_loaded'):
            del st.session_state.example_loaded
        st.rerun()

with col_code:
    if st.button("See in Code", use_container_width=True):
        st.session_state.show_code_modal = True

# Code modal/dialog
if st.session_state.get('show_code_modal'):
    st.markdown("---")
    st.markdown("### Reproducibility Code")
    st.markdown("Use this code to reproduce the classification with the CatLLM Python package:")

    # Use results code if available, otherwise generate from current parameters
    if st.session_state.results:
        code_to_show = st.session_state.results['code']
    else:
        # Get current categories from session state
        current_categories = [c for c in st.session_state.categories[:st.session_state.category_count] if c.strip()]

        # Determine current input type and description
        input_type_map = {"Survey Responses": "text", "PDF Documents": "pdf", "Images": "image"}
        current_input_type = input_type_map.get(st.session_state.get('input_type_radio', 'Survey Responses'), 'text')
        current_description = st.session_state.get('survey_column', '') or st.session_state.get('pdf_desc', '') or st.session_state.get('image_desc', '') or 'your_data'

        # Get current classification mode and models
        current_classify_mode = st.session_state.get('classify_mode', 'Single Model')
        current_model_tier = st.session_state.get('classify_model_tier', 'Free Models')

        if current_classify_mode in ["Model Comparison", "Ensemble"]:
            # Multi-model mode
            if current_model_tier == 'Free Models':
                model_displays = st.session_state.get('classify_models_multi', [])
                current_models_list = [FREE_MODELS_MAP.get(d, d) for d in model_displays]
            else:
                current_models_list = st.session_state.get('classify_models_multi_paid', [])
            current_model = ", ".join(current_models_list) if current_models_list else "gpt-4o-mini"
            current_model_source = f"{current_classify_mode} ({len(current_models_list)} models)"
        else:
            # Single model mode
            if current_model_tier == 'Free Models':
                model_display = st.session_state.get('classify_model', 'GPT-4o Mini')
                current_model = FREE_MODELS_MAP.get(model_display, 'gpt-4o-mini')
            else:
                current_model = st.session_state.get('classify_model_paid', 'gpt-4o-mini')
            current_models_list = [current_model]
            current_model_source = get_model_source(current_model)

        # Get consensus threshold for ensemble mode
        consensus_options = {"Majority (50%+)": 0.5, "Two-Thirds (67%+)": 0.67, "Unanimous (100%)": 1.0}
        current_consensus = consensus_options.get(st.session_state.get('consensus_choice', 'Majority (50%+)'), 0.5)

        # Get PDF mode if applicable
        current_mode = None
        if current_input_type == "pdf":
            mode_mapping = {
                "Image (visual documents)": "image",
                "Text (text-heavy)": "text",
                "Both (comprehensive)": "both"
            }
            current_mode = mode_mapping.get(st.session_state.get('pdf_mode', 'Image (visual documents)'), 'image')

        if current_categories:
            # Check if categories were auto-extracted
            if st.session_state.extraction_params:
                classify_params = {
                    'model': current_model,
                    'description': current_description,
                    'mode': current_mode,
                    'classify_mode': current_classify_mode,
                    'models_list': current_models_list,
                    'consensus_threshold': current_consensus,
                }
                code_to_show = generate_full_code(st.session_state.extraction_params, classify_params)
            else:
                code_to_show = generate_classify_code(
                    current_input_type, current_description, current_categories,
                    current_model, current_model_source, current_mode,
                    classify_mode=current_classify_mode, models_list=current_models_list,
                    consensus_threshold=current_consensus
                )
        else:
            code_to_show = '''import catllm

# Define your categories
categories = [
    "Category 1",
    "Category 2",
    # Add more categories...
]

# Classify your data
result = catllm.classify(
    input_data=df["your_column"].tolist(),
    categories=categories,
    api_key="YOUR_API_KEY",
    description="your_description",
    user_model="gpt-4o-mini"
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)
'''

    st.code(code_to_show, language='python')
    if st.button("Close"):
        st.session_state.show_code_modal = False
        st.rerun()
