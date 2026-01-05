"""
Gradio app - Conditional interface with Extract, Assign, and Extract & Assign modes
"""

import gradio as gr
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
MAX_FILE_SIZE_MB = 100  # Warn users if total file size exceeds this


def calculate_total_file_size(files):
    """Calculate total size of uploaded files in MB."""
    if files is None:
        return 0
    if not isinstance(files, list):
        files = [files]

    total_bytes = 0
    for f in files:
        try:
            file_path = f if isinstance(f, str) else f.name
            total_bytes += os.path.getsize(file_path)
        except (OSError, AttributeError):
            pass
    return total_bytes / (1024 * 1024)  # Convert to MB


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
    input_data="path/to/your/pdfs/",  # or list of PDF paths
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
    input_data="path/to/your/images/",  # or list of image paths
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
    input_data="path/to/your/pdfs/",  # or list of PDF paths
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
    input_data="path/to/your/images/",  # or list of image paths
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
    """Auto-detect model source. All HF router models (novita, groq, etc) use 'huggingface'."""
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


def generate_methodology_report_pdf(categories, model, column_name, num_rows, model_source, filename, success_rate,
                          result_df=None, processing_time=None, prompt_template=None,
                          data_quality=None, catllm_version=None, python_version=None,
                          task_type="assign", extracted_categories_df=None, max_categories=None,
                          input_type="text", description=None):
    """Generate a PDF methodology report for reproducibility and transparency.

    task_type: "extract", "assign", or "extract_and_assign"
    """
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

    # Title based on task type
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
CatLLM first discovers categories from your data using LLMs, then classifies each item into those categories. \
This two-phase approach combines exploratory category discovery with systematic classification, ensuring both \
data-driven category selection and reproducible assignments."""
    else:
        about_text = """This methodology report documents the classification process for reproducibility and transparency. \
CatLLM addresses an issue identified by researchers in "Prompt-Hacking: The New p-Hacking?" (Kosch &amp; Feger, 2025; \
arXiv:2504.14571): researchers could keep modifying prompts to obtain outputs that support desired conclusions, and \
this variability in pseudo-natural language poses a challenge for reproducibility since each prompt, even if only \
slightly altered, can yield different outputs, making it impossible to replicate findings reliably. CatLLM restricts \
the prompt to a standard template that is impartial to the researcher's inclinations, ensuring \
consistent and reproducible results."""

    story.append(Paragraph(about_text, normal_style))
    story.append(Spacer(1, 15))

    # For extract_and_assign, show extraction methodology first
    if task_type in ["extract", "extract_and_assign"]:
        story.append(Paragraph("Category Extraction Methodology", heading_style))
        extraction_text = f"""Categories were automatically extracted from your data using the following process:

1. Data was divided into chunks for analysis
2. Each chunk was analyzed by the LLM to identify themes and categories
3. Categories from all chunks were consolidated and deduplicated
4. Final categories were selected based on frequency and relevance
5. Maximum categories requested: {max_categories or 'default'}"""
        story.append(Paragraph(extraction_text.replace('\n', '<br/>'), normal_style))
        story.append(Spacer(1, 15))

    # Category mapping section - for assign and extract_and_assign
    if task_type in ["assign", "extract_and_assign"] and categories:
        story.append(Paragraph("Category Mapping", heading_style))
        story.append(Paragraph("Each category column contains binary values: 1 = present, 0 = not present", normal_style))
        story.append(Spacer(1, 8))

    # Category table - show for all task types that have categories
    if categories:
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

    # Output columns description - only for classification tasks
    if task_type in ["assign", "extract_and_assign"]:
        story.append(Paragraph("Other Output Columns", heading_style))
        other_cols = [
            ["Column Name", "Description"],
            ["survey_input", "The original text that was classified"],
            ["model_response", "Raw response from the LLM"],
            ["json", "Extracted JSON with category assignments"],
            ["processing_status", "'success' if classification worked, 'error' if failed"],
            ["categories_id", "Comma-separated list of assigned category numbers"],
        ]
        other_table = Table(other_cols, colWidths=[120, 330])
        other_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(other_table)

    story.append(Spacer(1, 30))
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316", normal_style))

    # Sample results - only for classification tasks
    if task_type in ["assign", "extract_and_assign"] and result_df is not None and len(result_df) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Sample Results (First 5 Rows)", title_style))
        story.append(Paragraph("Example classifications showing original text and assigned categories:", normal_style))
        story.append(Spacer(1, 15))

        sample_data = [["Original Text (truncated)", "Assigned Categories"]]
        sample_df = result_df.head(5)

        for _, row in sample_df.iterrows():
            original_text = str(row.get('survey_input', ''))[:80]
            if len(str(row.get('survey_input', ''))) > 80:
                original_text += "..."
            assigned = row.get('categories_id', '')
            if pd.isna(assigned) or assigned == '':
                assigned = "None"
            sample_data.append([original_text, str(assigned)])

        sample_table = Table(sample_data, colWidths=[320, 130])
        sample_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(sample_table)

    # Category distribution - only for classification tasks
    if task_type in ["assign", "extract_and_assign"]:
        story.append(PageBreak())
        story.append(Paragraph("Category Distribution", title_style))
        story.append(Paragraph("Count and percentage of responses assigned to each category:", normal_style))
        story.append(Spacer(1, 15))

        if result_df is not None:
            dist_data = [["Category", "Description", "Count", "Percentage"]]
            total_rows = len(result_df)

            for i, cat in enumerate(categories, 1):
                col_name = f"category_{i}"
                if col_name in result_df.columns:
                    count = int(result_df[col_name].sum())
                    pct = (count / total_rows) * 100 if total_rows > 0 else 0
                    dist_data.append([col_name, cat[:40], str(count), f"{pct:.1f}%"])
                else:
                    dist_data.append([col_name, cat[:40], "N/A", "N/A"])

            dist_table = Table(dist_data, colWidths=[80, 200, 60, 80])
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
            ]))
            story.append(dist_table)
            story.append(Spacer(1, 15))
            story.append(Paragraph(f"<i>Note: Percentages may sum to more than 100% as responses can be assigned to multiple categories.</i>", normal_style))

    # Summary section - adjust title based on task type
    story.append(PageBreak())
    if task_type == "extract_and_assign":
        story.append(Paragraph("Processing Summary", title_style))
    else:
        story.append(Paragraph("Classification Summary", title_style))
    story.append(Spacer(1, 15))

    # Build summary data based on task type
    if task_type == "assign":
        story.append(Paragraph("Classification Details", heading_style))
        summary_data = [
            ["Source File", filename],
            ["Source Column", column_name],
            ["Model Used", model],
            ["Model Source", model_source],
            ["Temperature", "default"],
            ["Rows Classified", str(num_rows)],
            ["Number of Categories", str(len(categories)) if categories else "0"],
            ["Success Rate", f"{success_rate:.2f}%"],
        ]
        if task_type == "extract_and_assign":
            summary_data.insert(4, ["Categories Auto-Extracted", "Yes"])
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

    if data_quality is not None:
        story.append(Paragraph("Data Quality Notes", heading_style))
        quality_data = [
            ["Empty/Null Inputs Skipped", str(data_quality.get('null_count', 0))],
            ["Average Text Length", f"{data_quality.get('avg_length', 0)} characters"],
            ["Min Text Length", f"{data_quality.get('min_length', 0)} characters"],
            ["Max Text Length", f"{data_quality.get('max_length', 0)} characters"],
            ["Responses with Errors", str(data_quality.get('error_count', 0))],
        ]
        quality_table = Table(quality_data, colWidths=[180, 270])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(quality_table)
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

    # Prompt template - only for classification tasks
    if task_type in ["assign", "extract_and_assign"] and prompt_template:
        story.append(PageBreak())
        story.append(Paragraph("Prompt Template Used", title_style))
        story.append(Paragraph("The following prompt template was sent to the LLM for each classification:", normal_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("Template with Placeholders:", heading_style))
        story.append(Spacer(1, 8))

        for line in prompt_template.split('\n'):
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if escaped_line.strip():
                story.append(Paragraph(escaped_line, code_style))
            else:
                story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))

        if categories:
            story.append(Paragraph("Example with Your Categories:", heading_style))
            story.append(Spacer(1, 8))

            categories_list = "\n".join([f"  {i}. {cat}" for i, cat in enumerate(categories, 1)])
            example_prompt = f'''Categorize this survey response "[YOUR TEXT HERE]" into the following categories:
{categories_list}
Provide your work in JSON format where the number belonging to each category
is the key and a 1 if the category is present and a 0 if not.'''

            for line in example_prompt.split('\n'):
                escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                if escaped_line.strip():
                    story.append(Paragraph(escaped_line, code_style))
                else:
                    story.append(Spacer(1, 5))

    # Reproducibility code section
    story.append(PageBreak())
    story.append(Paragraph("Reproducibility Code", title_style))

    if task_type == "extract_and_assign":
        story.append(Paragraph("Use the following Python code to reproduce this extraction and classification:", normal_style))
        story.append(Spacer(1, 15))

        categories_str = ", ".join([f'"{cat}"' for cat in categories]) if categories else ""

        code_text = f'''import catllm

# Step 1: Extract categories from your data
extract_result = catllm.extract(
    input_data="path/to/your/data",
    api_key="YOUR_API_KEY",
    input_type="{input_type}",
    description="{description or column_name}",
    user_model="{model}",
    model_source="{model_source}",
    max_categories={max_categories or 12}
)

categories = extract_result["top_categories"]
print("Extracted categories:", categories)

# Step 2: Classify data using extracted categories
result = catllm.classify(
    input_data="path/to/your/data",
    categories=categories,
    api_key="YOUR_API_KEY",
    input_type="{input_type}",
    description="{description or column_name}",
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)
result.to_csv("classified_results.csv", index=False)'''

    else:  # assign
        story.append(Paragraph("Use the following Python code to reproduce this classification:", normal_style))
        story.append(Spacer(1, 15))

        categories_str = ", ".join([f'"{cat}"' for cat in categories]) if categories else ""

        code_text = f'''import catllm
import pandas as pd

# Load your survey data
df = pd.read_csv("{filename}")

# Define your categories
categories = [{categories_str}]

# Classify the responses
result = catllm.classify(
    input_data=df["{column_name}"].tolist(),
    categories=categories,
    api_key="YOUR_API_KEY",
    input_type="{input_type}",
    description="{description or column_name}",
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)

# Save to CSV
result.to_csv("classified_results.csv", index=False)'''

    for line in code_text.split('\n'):
        if line.strip() == '':
            story.append(Spacer(1, 5))
        else:
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(escaped_line, code_style))

    doc.build(story)
    return pdf_file.name


def load_example_dataset():
    """Load the example dataset for users to try the app."""
    example_path = "example_data.csv"
    try:
        df = pd.read_csv(example_path)
        columns = df.columns.tolist()
        return (
            example_path,
            gr.update(choices=columns, value=columns[0] if columns else None),
            f"Loaded example dataset ({len(df)} rows). Select column and choose a task."
        )
    except Exception as e:
        return None, gr.update(choices=[], value=None), f"**Error loading example:** {str(e)}"


def load_columns(file):
    if file is None:
        return gr.update(choices=[], value=None), "Please upload a file first"

    try:
        file_path = file if isinstance(file, str) else file.name
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        columns = df.columns.tolist()
        num_rows = len(df)

        if num_rows > 1000:
            est_minutes = round(num_rows * 1.5 / 60)
            status_msg = f"Loaded {num_rows:,} rows. Processing may take ~{est_minutes} minutes."
        else:
            status_msg = f"Loaded {num_rows:,} rows. Choose a task to proceed."

        return (
            gr.update(choices=columns, value=columns[0] if columns else None),
            status_msg
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"**Error:** {str(e)}"


def update_task_visibility(task):
    """Update visibility of components based on selected task."""
    if task == "manual":
        return (
            gr.update(visible=False),  # auto_extract_group
            gr.update(visible=True),   # categories_group
            gr.update(visible=True),   # model_group
            gr.update(visible=True, value="Classify Data"),  # run_btn
            gr.update(visible=True),   # classify_output_group
            "Enter your categories and click 'Classify Data'."
        )
    elif task == "auto_extract":
        return (
            gr.update(visible=True),   # auto_extract_group
            gr.update(visible=False),  # categories_group (will show after extraction)
            gr.update(visible=False),  # model_group (will show after extraction)
            gr.update(visible=False),  # run_btn (will show after extraction)
            gr.update(visible=True),   # classify_output_group
            "Select number of categories and click 'Extract Categories'."
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Upload data and select a task to continue."
        )


def run_auto_extract(input_type, spreadsheet_file, spreadsheet_column,
                     pdf_file, pdf_folder, pdf_description, pdf_mode,
                     image_file, image_folder, image_description,
                     max_categories_val,
                     model_tier, model, model_source_input, api_key_input,
                     progress=gr.Progress(track_tqdm=True)):
    """Extract categories from data and fill the category textboxes."""
    if not CATLLM_AVAILABLE:
        # Return empty updates for all category inputs + status
        return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** catllm package not available"]

    actual_api_key, provider = get_api_key(model, model_tier, api_key_input)
    if not actual_api_key:
        return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, f"**Error:** {provider} API key not configured"]

    if model_source_input == "auto":
        model_source = get_model_source(model)
    else:
        model_source = model_source_input

    try:
        if input_type == "Survey Responses":
            if not spreadsheet_file:
                return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** Please upload a CSV/Excel file first"]
            if not spreadsheet_column:
                return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** Please select a column first"]

            file_path = spreadsheet_file if isinstance(spreadsheet_file, str) else spreadsheet_file.name
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            input_data = df[spreadsheet_column].tolist()
            description = spreadsheet_column
            input_type_param = "text"
            mode_param = None

        elif input_type == "PDF Documents":
            if pdf_folder:
                if isinstance(pdf_folder, list):
                    input_data = [f if isinstance(f, str) else f.name for f in pdf_folder if str(f.name if hasattr(f, 'name') else f).lower().endswith('.pdf')]
                else:
                    input_data = pdf_folder if isinstance(pdf_folder, str) else pdf_folder.name
            elif pdf_file:
                if isinstance(pdf_file, list):
                    input_data = [f if isinstance(f, str) else f.name for f in pdf_file]
                else:
                    input_data = pdf_file if isinstance(pdf_file, str) else pdf_file.name
            else:
                return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** Please upload PDF file(s) first"]

            description = pdf_description or "document"
            input_type_param = "pdf"
            mode_mapping = {"Image (visual documents)": "image", "Text (text-heavy)": "text", "Both (comprehensive)": "both"}
            mode_param = mode_mapping.get(pdf_mode, "image")

        elif input_type == "Images":
            if image_folder:
                if isinstance(image_folder, list):
                    input_data = [f if isinstance(f, str) else f.name for f in image_folder]
                else:
                    input_data = image_folder if isinstance(image_folder, str) else image_folder.name
            elif image_file:
                if isinstance(image_file, list):
                    input_data = [f if isinstance(f, str) else f.name for f in image_file]
                else:
                    input_data = image_file if isinstance(image_file, str) else image_file.name
            else:
                return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** Please upload image file(s) first"]

            description = image_description or "images"
            input_type_param = "image"
            mode_param = None

        else:
            return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, f"**Error:** Unknown input type: {input_type}"]

        # Calculate divisions based on input size
        if isinstance(input_data, list):
            num_items = len(input_data)
        else:
            num_items = 1

        if input_type_param == "image":
            divisions = min(3, max(1, num_items // 5))
            categories_per_chunk = 12
        else:
            # Ensure each chunk has enough items for extraction
            # We need at least categories_per_chunk items per chunk
            divisions = max(1, num_items // 15)  # Aim for ~15 items per chunk
            divisions = min(divisions, 5)  # Cap at 5 divisions
            chunk_size = num_items // max(1, divisions)
            categories_per_chunk = min(10, chunk_size - 1)  # Leave room for extraction

        # Extract categories
        extract_kwargs = {
            'input_data': input_data,
            'api_key': actual_api_key,
            'input_type': input_type_param,
            'description': description,
            'user_model': model,
            'model_source': model_source,
            'divisions': divisions,
            'categories_per_chunk': categories_per_chunk,
            'max_categories': int(max_categories_val)
        }
        if mode_param:
            extract_kwargs['mode'] = mode_param

        extract_result = catllm.extract(**extract_kwargs)
        categories = extract_result.get('top_categories', [])

        if not categories:
            return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, "**Error:** No categories were extracted"]

        # Fill the category textboxes
        updates = []
        num_categories = min(len(categories), MAX_CATEGORIES)
        for i in range(MAX_CATEGORIES):
            if i < num_categories:
                updates.append(gr.update(value=categories[i], visible=True))
            elif i < INITIAL_CATEGORIES:
                updates.append(gr.update(value="", visible=True))
            else:
                updates.append(gr.update(value="", visible=False))

        # Return updates for category inputs + new category count + status
        return updates + [num_categories, f"Extracted {len(categories)} categories. Review and edit as needed, then click 'Classify Data'."]

    except Exception as e:
        return [gr.update()] * MAX_CATEGORIES + [MAX_CATEGORIES, f"**Error:** {str(e)}"]


def run_classify_data(input_type, spreadsheet_file, spreadsheet_column,
                      pdf_file, pdf_folder, pdf_description, pdf_mode,
                      image_file, image_folder, image_description,
                      cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                      model_tier, model, model_source_input, api_key_input,
                      progress=gr.Progress(track_tqdm=True)):
    """Classify data with user-provided categories."""
    if not CATLLM_AVAILABLE:
        yield None, None, None, None, None, "**Error:** catllm package not available"
        return

    all_cats = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    categories = [c.strip() for c in all_cats if c and c.strip()]

    if not categories:
        yield None, None, None, None, None, "**Error:** Please enter at least one category"
        return

    actual_api_key, provider = get_api_key(model, model_tier, api_key_input)
    if not actual_api_key:
        yield None, None, None, None, None, f"**Error:** {provider} API key not configured"
        return

    if model_source_input == "auto":
        model_source = get_model_source(model)
    else:
        model_source = model_source_input

    # Check file size for images and PDFs
    files_to_check = None
    if input_type == "Images":
        files_to_check = image_folder if image_folder else image_file
    elif input_type == "PDF Documents":
        files_to_check = pdf_folder if pdf_folder else pdf_file

    if files_to_check:
        total_size_mb = calculate_total_file_size(files_to_check)
        if total_size_mb > MAX_FILE_SIZE_MB:
            # Generate the code for the user
            if input_type == "Images":
                code = generate_classify_code("image", image_description or "images", categories, model, model_source)
            else:
                mode_mapping = {"Image (visual documents)": "image", "Text (text-heavy)": "text", "Both (comprehensive)": "both"}
                actual_mode = mode_mapping.get(pdf_mode, "image")
                code = generate_classify_code("pdf", pdf_description or "document", categories, model, model_source, actual_mode)

            warning_msg = f"""**⚠️ Large Upload Detected ({total_size_mb:.1f} MB)**

Uploads over {MAX_FILE_SIZE_MB} MB may experience performance issues or timeouts on this web app.

**Recommended:** Run the code locally using the Python package instead. See the code below, or click "See the Code" after this message.

```
pip install cat-llm
```
"""
            yield None, None, None, code, None, warning_msg
            return

    try:
        yield None, None, None, None, None, "Classifying your data..."

        start_time = time.time()

        if input_type == "Survey Responses":
            if not spreadsheet_file:
                yield None, None, None, None, None, "**Error:** Please upload a CSV/Excel file"
                return
            if not spreadsheet_column:
                yield None, None, None, None, None, "**Error:** Please select a column"
                return

            file_path = spreadsheet_file if isinstance(spreadsheet_file, str) else spreadsheet_file.name
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            input_data = df[spreadsheet_column].tolist()
            original_filename = file_path.split("/")[-1]
            column_name = spreadsheet_column

            result = catllm.classify(
                input_data=input_data,
                categories=categories,
                api_key=actual_api_key,
                input_type="text",
                description=spreadsheet_column,
                user_model=model,
                model_source=model_source
            )

        elif input_type == "PDF Documents":
            # Use folder if provided, otherwise use uploaded files
            if pdf_folder:
                if isinstance(pdf_folder, list):
                    pdf_input = [f if isinstance(f, str) else f.name for f in pdf_folder if str(f.name if hasattr(f, 'name') else f).lower().endswith('.pdf')]
                    original_filename = "pdf_folder"
                else:
                    pdf_input = pdf_folder if isinstance(pdf_folder, str) else pdf_folder.name
                    original_filename = pdf_input.split("/")[-1]
            elif pdf_file:
                if isinstance(pdf_file, list):
                    pdf_input = [f if isinstance(f, str) else f.name for f in pdf_file]
                    original_filename = "multiple_pdfs"
                else:
                    pdf_input = pdf_file if isinstance(pdf_file, str) else pdf_file.name
                    original_filename = pdf_input.split("/")[-1]
            else:
                yield None, None, None, None, None, "**Error:** Please upload PDF file(s) or a folder"
                return

            column_name = "PDF Pages"

            mode_mapping = {
                "Image (visual documents)": "image",
                "Text (text-heavy)": "text",
                "Both (comprehensive)": "both"
            }
            actual_mode = mode_mapping.get(pdf_mode, "image")

            result = catllm.classify(
                input_data=pdf_input,
                categories=categories,
                api_key=actual_api_key,
                input_type="pdf",
                description=pdf_description or "document",
                mode=actual_mode,
                user_model=model,
                model_source=model_source
            )

        elif input_type == "Images":
            # Use folder if provided, otherwise use uploaded files
            if image_folder:
                if isinstance(image_folder, list):
                    image_input = [f if isinstance(f, str) else f.name for f in image_folder]
                    original_filename = "image_folder"
                else:
                    image_input = image_folder if isinstance(image_folder, str) else image_folder.name
                    original_filename = image_input.split("/")[-1]
            elif image_file:
                if isinstance(image_file, list):
                    image_input = [f if isinstance(f, str) else f.name for f in image_file]
                    original_filename = "multiple_images"
                else:
                    image_input = image_file if isinstance(image_file, str) else image_file.name
                    original_filename = image_input.split("/")[-1]
            else:
                yield None, None, None, None, None, "**Error:** Please upload image file(s) or a folder"
                return

            column_name = "Image Files"

            result = catllm.classify(
                input_data=image_input,
                categories=categories,
                api_key=actual_api_key,
                input_type="image",
                description=image_description or "images",
                user_model=model,
                model_source=model_source
            )

        else:
            yield None, None, None, None, None, f"**Error:** Unknown input type: {input_type}"
            return

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
        prompt_template = '''Categorize this survey response "{response}" into the following categories that apply:
{categories}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values.'''

        # Determine input_type_param for the report
        if input_type == "Survey Responses":
            input_type_param = "text"
            description_param = spreadsheet_column
        elif input_type == "PDF Documents":
            input_type_param = "pdf"
            description_param = pdf_description or "document"
        else:
            input_type_param = "image"
            description_param = image_description or "images"

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
            prompt_template=prompt_template,
            data_quality={'null_count': 0, 'avg_length': 0, 'min_length': 0, 'max_length': 0, 'error_count': 0},
            catllm_version=catllm_version,
            python_version=python_version,
            task_type="assign",
            input_type=input_type_param,
            description=description_param
        )

        # Create distribution plot
        dist_data = []
        total_rows = len(result)
        for i, cat in enumerate(categories, 1):
            col_name = f"category_{i}"
            if col_name in result.columns:
                count = int(result[col_name].sum())
                pct = (count / total_rows) * 100 if total_rows > 0 else 0
                dist_data.append({"Category": cat, "Percentage": round(pct, 1)})

        fig, ax = plt.subplots(figsize=(12, max(6, len(dist_data) * 1.0)))
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

        # Generate reproducibility code
        if input_type == "Survey Responses":
            code = generate_classify_code("text", spreadsheet_column, categories, model, model_source)
        elif input_type == "PDF Documents":
            mode_mapping = {"Image (visual documents)": "image", "Text (text-heavy)": "text", "Both (comprehensive)": "both"}
            actual_mode = mode_mapping.get(pdf_mode, "image")
            code = generate_classify_code("pdf", pdf_description or "document", categories, model, model_source, actual_mode)
        else:  # Images
            code = generate_classify_code("image", image_description or "images", categories, model, model_source)

        yield (
            gr.update(value=fig, visible=True),
            gr.update(value=result, visible=True),
            [csv_path, report_pdf_path],
            code,
            None,
            f"Classified {num_items} items in {processing_time:.1f}s"
        )

    except Exception as e:
        yield None, None, None, None, None, f"**Error:** {str(e)}"



def add_category_field(current_count):
    new_count = min(current_count + 1, MAX_CATEGORIES)
    updates = []
    for i in range(MAX_CATEGORIES):
        updates.append(gr.update(visible=(i < new_count)))
    updates.append(gr.update(visible=(new_count < MAX_CATEGORIES)))
    updates.append(new_count)
    return updates


def reset_all():
    """Reset all inputs and outputs to initial state."""
    updates = [
        "Survey Responses",  # input_type
        gr.update(visible=True),  # text_input_group
        gr.update(visible=False),  # pdf_input_group
        gr.update(visible=False),  # image_input_group
        None,  # spreadsheet_file
        gr.update(choices=[], value=None),  # spreadsheet_column
        "Upload File(s)",  # pdf_upload_type
        None,  # pdf_file
        None,  # pdf_folder
        "",  # pdf_description
        "Image (visual documents)",  # pdf_mode
        "Upload File(s)",  # image_upload_type
        None,  # image_file
        None,  # image_folder
        "",  # image_description
        None,  # task_mode
        gr.update(variant="secondary"),  # manual_btn - reset to unselected
        gr.update(variant="secondary"),  # auto_extract_btn - reset to unselected
    ]
    # Reset category inputs
    for i in range(MAX_CATEGORIES):
        updates.append(gr.update(value="", visible=(i < INITIAL_CATEGORIES)))
    updates.extend([
        gr.update(visible=True),  # add_category_btn
        INITIAL_CATEGORIES,  # category_count
        gr.update(visible=False),  # auto_extract_group
        12,  # max_categories (reset to default)
        "",  # auto_extract_status
        gr.update(visible=False),  # categories_group
        gr.update(visible=False),  # model_group
        gr.update(visible=False, value="Classify Data"),  # run_btn
        "Free Models",  # model_tier
        FREE_MODEL_CHOICES[0],  # model
        "auto",  # model_source
        "",  # api_key
        gr.update(visible=False),  # api_key
        "**Free tier** - no API key required!",  # api_key_status
        "Ready. Upload data and select a task.",  # status
        gr.update(visible=False),  # classify_output_group
        gr.update(value=None, visible=False),  # distribution_plot
        gr.update(value=None, visible=False),  # results
        None,  # download_file
        "# Code will be generated after classification",  # classify_code_display
    ])
    return updates


custom_css = """
* {
    font-family: Helvetica, Arial, sans-serif !important;
}
.task-btn {
    min-width: 150px !important;
}

/* Expandable plot styles */
.expandable-plot {
    min-height: 400px !important;
    cursor: pointer;
    transition: transform 0.3s ease;
}

.expandable-plot:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.expandable-plot img {
    min-height: 350px !important;
    object-fit: contain;
}

/* Fullscreen modal for plot */
.expandable-plot.fullscreen {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 9999 !important;
    background: rgba(255, 255, 255, 0.98) !important;
    padding: 20px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.expandable-plot.fullscreen img {
    max-width: 95vw !important;
    max-height: 90vh !important;
    min-height: auto !important;
}

/* Mobile-responsive styles */
@media screen and (max-width: 768px) {
    /* Reduce overall padding */
    .gradio-container {
        padding: 8px !important;
    }

    /* Stack columns vertically on mobile */
    .gradio-row {
        flex-direction: column !important;
    }

    /* Full width columns on mobile */
    .gradio-column {
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Smaller task buttons on mobile */
    .task-btn {
        min-width: 100px !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
    }

    /* Compact file upload areas */
    .file-upload, .upload-button, [data-testid="file"], .gr-file {
        min-height: 80px !important;
        padding: 10px !important;
    }

    /* Smaller upload button text */
    .upload-button span, .gr-file span, [data-testid="file"] span {
        font-size: 13px !important;
    }

    /* Compact buttons */
    button {
        padding: 8px 14px !important;
        font-size: 14px !important;
    }

    /* Smaller primary action button */
    button.primary, button.lg {
        padding: 10px 16px !important;
        font-size: 15px !important;
    }

    /* Compact form inputs */
    input, textarea, select {
        padding: 8px !important;
        font-size: 14px !important;
    }

    /* Reduce textbox heights */
    textarea {
        min-height: 60px !important;
    }

    /* Smaller dropdown */
    .gr-dropdown {
        font-size: 14px !important;
    }

    /* Compact radio buttons */
    .gr-radio, .gr-checkbox {
        gap: 6px !important;
    }

    .gr-radio label, .gr-checkbox label {
        font-size: 13px !important;
        padding: 6px 10px !important;
    }

    /* Smaller headings */
    h1 {
        font-size: 1.4rem !important;
    }

    h2, h3 {
        font-size: 1.1rem !important;
    }

    /* Compact markdown text */
    .prose, .gr-markdown {
        font-size: 14px !important;
    }

    /* Compact accordion */
    .gr-accordion {
        padding: 8px !important;
    }

    /* Smaller logo on mobile */
    img[alt="logo"], .gr-image img {
        height: 80px !important;
        max-height: 80px !important;
    }

    /* Compact group containers */
    .gr-group {
        padding: 10px !important;
        gap: 8px !important;
    }

    /* Reduce spacing between form elements */
    .gr-form > *, .gr-block > * {
        margin-bottom: 8px !important;
    }

    /* Compact dataframe/table on mobile */
    .gr-dataframe, .dataframe {
        font-size: 12px !important;
    }

    .gr-dataframe th, .gr-dataframe td {
        padding: 4px 6px !important;
    }

    /* Make plot responsive */
    .gr-plot {
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Compact download buttons */
    .gr-file-download, [data-testid="download"] {
        padding: 8px !important;
    }
}

/* Extra small devices (phones in portrait) */
@media screen and (max-width: 480px) {
    .gradio-container {
        padding: 4px !important;
    }

    .task-btn {
        min-width: 80px !important;
        padding: 6px 10px !important;
        font-size: 12px !important;
    }

    button {
        padding: 6px 12px !important;
        font-size: 13px !important;
    }

    h1 {
        font-size: 1.2rem !important;
    }

    h2, h3 {
        font-size: 1rem !important;
    }

    .prose, .gr-markdown {
        font-size: 13px !important;
    }

    img[alt="logo"], .gr-image img {
        height: 60px !important;
        max-height: 60px !important;
    }

    /* Stack task buttons vertically on very small screens */
    .task-btn {
        width: 100% !important;
        min-width: unset !important;
    }
}
"""

custom_js = """
function() {
    // Add click-to-expand functionality for the plot
    document.addEventListener('click', function(e) {
        const plot = e.target.closest('.expandable-plot');
        if (plot) {
            plot.classList.toggle('fullscreen');
        }
    });

    // Close fullscreen on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const fullscreenPlot = document.querySelector('.expandable-plot.fullscreen');
            if (fullscreenPlot) {
                fullscreenPlot.classList.remove('fullscreen');
            }
        }
    });
}
"""

with gr.Blocks(title="CatLLM - Research Data Classifier", theme=gr.themes.Soft(), css=custom_css, js=custom_js) as demo:
    gr.Image("logo.png", show_label=False, show_download_button=False, height=115, container=False)
    gr.Markdown("# CatLLM - Research Data Classifier")
    gr.Markdown("Extract categories from or classify text data, PDFs, and images using LLMs.")

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
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

    # State variables
    category_count = gr.State(value=INITIAL_CATEGORIES)
    task_mode = gr.State(value=None)

    with gr.Row():
        with gr.Column():
            # Input type selector
            input_type = gr.Radio(
                choices=["Survey Responses", "PDF Documents", "Images"],
                value="Survey Responses",
                label="Input Type"
            )

            # Survey Responses input group
            with gr.Group(visible=True) as text_input_group:
                spreadsheet_file = gr.File(
                    label="Upload Data (CSV or Excel)",
                    file_types=[".csv", ".xlsx", ".xls"]
                )
                example_btn = gr.Button("Try Example Dataset", variant="secondary", size="sm")
                spreadsheet_column = gr.Dropdown(
                    label="Column to Process",
                    choices=[],
                    info="Select the column containing text"
                )

            # PDF input group
            with gr.Group(visible=False) as pdf_input_group:
                pdf_upload_type = gr.Radio(
                    choices=["Upload File(s)", "Upload Folder"],
                    value="Upload File(s)",
                    label="Upload Type"
                )
                pdf_file = gr.File(
                    label="Upload PDF Document(s)",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                pdf_folder = gr.File(
                    label="Upload PDF Folder",
                    file_count="directory",
                    visible=False
                )
                pdf_description = gr.Textbox(
                    label="Document Description",
                    placeholder="e.g., 'research papers', 'interview transcripts'",
                    info="Helps the LLM understand context"
                )
                pdf_mode = gr.Radio(
                    choices=["Image (visual documents)", "Text (text-heavy)", "Both (comprehensive)"],
                    value="Image (visual documents)",
                    label="Processing Mode"
                )

            # Image input group
            with gr.Group(visible=False) as image_input_group:
                image_upload_type = gr.Radio(
                    choices=["Upload File(s)", "Upload Folder"],
                    value="Upload File(s)",
                    label="Upload Type"
                )
                image_file = gr.File(
                    label="Upload Images",
                    file_types=["image"],
                    file_count="multiple"
                )
                image_folder = gr.File(
                    label="Upload Image Folder",
                    file_count="directory",
                    visible=False
                )
                image_description = gr.Textbox(
                    label="Image Description",
                    placeholder="e.g., 'product photos', 'social media posts'",
                    info="Helps the LLM understand context"
                )

            # Task selection buttons
            gr.Markdown("### What would you like to do?")
            with gr.Row():
                manual_btn = gr.Button("Enter Categories Manually", variant="secondary", elem_classes="task-btn")
                auto_extract_btn = gr.Button("Auto-extract Categories", variant="secondary", elem_classes="task-btn")

            # Auto-extract settings group (visible when auto-extract is selected)
            with gr.Group(visible=False) as auto_extract_group:
                gr.Markdown("### Auto-extract Categories")
                gr.Markdown("We'll analyze your data to discover the main categories.")
                max_categories = gr.Slider(
                    minimum=3,
                    maximum=25,
                    value=12,
                    step=1,
                    label="Number of Categories to Extract",
                    info="How many categories should be identified in your data"
                )
                run_auto_extract_btn = gr.Button("Extract Categories", variant="primary")
                auto_extract_status = gr.Markdown("")

            # Categories group (visible for manual entry or after auto-extract)
            with gr.Group(visible=False) as categories_group:
                gr.Markdown("### Categories")
                categories_instruction = gr.Markdown("Enter your classification categories below.")
                category_inputs = []
                placeholder_examples = [
                    "e.g., Positive sentiment",
                    "e.g., Negative sentiment",
                    "e.g., Product feedback",
                    "e.g., Service complaint",
                    "e.g., Feature request",
                    "e.g., Custom category"
                ]
                for i in range(MAX_CATEGORIES):
                    visible = i < INITIAL_CATEGORIES
                    placeholder = placeholder_examples[i] if i < len(placeholder_examples) else "e.g., Custom category"
                    cat_input = gr.Textbox(
                        label=f"Category {i+1}",
                        placeholder=placeholder,
                        visible=visible
                    )
                    category_inputs.append(cat_input)
                add_category_btn = gr.Button("+ Add More", variant="secondary", size="sm")

            # Model selection group
            with gr.Group(visible=False) as model_group:
                gr.Markdown("### Model")
                model_tier = gr.Radio(
                    choices=["Free Models", "Bring Your Own Key"],
                    value="Free Models",
                    label="Model Tier"
                )
                model = gr.Dropdown(
                    choices=FREE_MODEL_CHOICES,
                    value="Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
                    label="Model",
                    allow_custom_value=False,  # Only allow custom for "Bring Your Own Key"
                    interactive=True
                )
                model_source = gr.Dropdown(
                    choices=["auto", "openai", "anthropic", "google", "mistral", "xai", "huggingface", "perplexity"],
                    value="auto",
                    label="Model Source",
                    visible=False  # Hide for free tier, show for BYOK
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key",
                    visible=False
                )
                api_key_status = gr.Markdown("**Free tier** - no API key required!")

            # Run button
            run_btn = gr.Button("Run", variant="primary", size="lg", visible=False)
            reset_btn = gr.Button("Reset", variant="stop")

        with gr.Column():
            status = gr.Markdown("Ready. Upload data and select a task.")

            # Classify output group
            with gr.Group(visible=False) as classify_output_group:
                gr.Markdown("### Classification Results")
                distribution_plot = gr.Plot(label="Category Distribution (%)", visible=False, elem_classes="expandable-plot")
                results = gr.DataFrame(label="Full Results", visible=False)
                download_file = gr.File(label="Download Results (CSV + Methodology Report)", file_count="multiple")
                with gr.Accordion("See the Code", open=False):
                    classify_code_display = gr.Code(
                        label="Python Code",
                        language="python",
                        value="# Code will be generated after classification",
                        interactive=False
                    )

    # Event handlers
    def switch_input_type(input_type_val):
        """Toggle visibility between input groups."""
        return (
            gr.update(visible=(input_type_val == "Survey Responses")),
            gr.update(visible=(input_type_val == "PDF Documents")),
            gr.update(visible=(input_type_val == "Images")),
            f"Ready to process {input_type_val.lower()}."
        )

    input_type.change(
        fn=switch_input_type,
        inputs=[input_type],
        outputs=[text_input_group, pdf_input_group, image_input_group, status]
    )

    def update_model_tier(tier):
        if tier == "Free Models":
            return (
                gr.update(choices=FREE_MODEL_CHOICES, value=FREE_MODEL_CHOICES[0], allow_custom_value=False),
                gr.update(visible=False),  # model_source hidden for free
                gr.update(visible=False),  # api_key hidden for free
                "**Free tier** - no API key required!"
            )
        else:
            return (
                gr.update(choices=PAID_MODEL_CHOICES, value=PAID_MODEL_CHOICES[0], allow_custom_value=True),
                gr.update(visible=True),   # model_source shown for BYOK
                gr.update(visible=True),   # api_key shown for BYOK
                "**Bring Your Own Key** - enter your API key below."
            )

    model_tier.change(
        fn=update_model_tier,
        inputs=[model_tier],
        outputs=[model, model_source, api_key, api_key_status]
    )

    spreadsheet_file.change(
        fn=load_columns,
        inputs=[spreadsheet_file],
        outputs=[spreadsheet_column, status]
    )

    example_btn.click(
        fn=load_example_dataset,
        inputs=[],
        outputs=[spreadsheet_file, spreadsheet_column, status]
    )

    # Toggle between file and folder upload for PDFs
    def toggle_pdf_upload(upload_type):
        if upload_type == "Upload File(s)":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    pdf_upload_type.change(
        fn=toggle_pdf_upload,
        inputs=[pdf_upload_type],
        outputs=[pdf_file, pdf_folder]
    )

    # Toggle between file and folder upload for Images
    def toggle_image_upload(upload_type):
        if upload_type == "Upload File(s)":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    image_upload_type.change(
        fn=toggle_image_upload,
        inputs=[image_upload_type],
        outputs=[image_file, image_folder]
    )

    add_category_btn.click(
        fn=add_category_field,
        inputs=[category_count],
        outputs=category_inputs + [add_category_btn, category_count]
    )

    # Manual entry button handler
    def select_manual():
        visibility = update_task_visibility("manual")
        return (
            "manual",
            gr.update(variant="primary"),    # manual_btn - selected
            gr.update(variant="secondary"),  # auto_extract_btn - not selected
        ) + visibility

    manual_btn.click(
        fn=select_manual,
        inputs=[],
        outputs=[task_mode, manual_btn, auto_extract_btn, auto_extract_group, categories_group, model_group, run_btn, classify_output_group, status]
    )

    # Auto-extract button handler
    def select_auto_extract():
        visibility = update_task_visibility("auto_extract")
        return (
            "auto_extract",
            gr.update(variant="secondary"),  # manual_btn - not selected
            gr.update(variant="primary"),    # auto_extract_btn - selected
        ) + visibility

    auto_extract_btn.click(
        fn=select_auto_extract,
        inputs=[],
        outputs=[task_mode, manual_btn, auto_extract_btn, auto_extract_group, categories_group, model_group, run_btn, classify_output_group, status]
    )

    # Run auto-extract button - extracts categories then shows next steps
    def run_auto_extract_and_show(input_type, spreadsheet_file, spreadsheet_column,
                                   pdf_file, pdf_folder, pdf_description, pdf_mode,
                                   image_file, image_folder, image_description,
                                   max_categories_val, model_tier, model, model_source, api_key,
                                   progress=gr.Progress(track_tqdm=True)):
        """Extract categories and then show the categories/model/run sections."""
        # Run the extraction
        result = run_auto_extract(input_type, spreadsheet_file, spreadsheet_column,
                                  pdf_file, pdf_folder, pdf_description, pdf_mode,
                                  image_file, image_folder, image_description,
                                  max_categories_val, model_tier, model, model_source, api_key,
                                  progress)
        # result is: category_inputs updates + [category_count, status]
        # Add visibility updates for categories_group, model_group, run_btn
        return result + [
            gr.update(visible=True),   # categories_group
            gr.update(visible=True),   # model_group
            gr.update(visible=True, value="Classify Data"),  # run_btn
        ]

    run_auto_extract_btn.click(
        fn=run_auto_extract_and_show,
        inputs=[input_type, spreadsheet_file, spreadsheet_column,
                pdf_file, pdf_folder, pdf_description, pdf_mode,
                image_file, image_folder, image_description,
                max_categories, model_tier, model, model_source, api_key],
        outputs=category_inputs + [category_count, auto_extract_status, categories_group, model_group, run_btn]
    )

    # Main run button handler - dispatches based on task_mode
    def dispatch_run(task, input_type, spreadsheet_file, spreadsheet_column,
                     pdf_file, pdf_folder_val, pdf_description, pdf_mode,
                     image_file, image_folder_val, image_description,
                     cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                     model_tier, model, model_source, api_key,
                     progress=gr.Progress(track_tqdm=True)):
        """Run classification with user-provided categories."""
        if task in ["manual", "auto_extract"]:
            # run_classify_data yields: (plot, df, files, code, unused, status)
            for update in run_classify_data(
                input_type, spreadsheet_file, spreadsheet_column,
                pdf_file, pdf_folder_val, pdf_description, pdf_mode,
                image_file, image_folder_val, image_description,
                cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                model_tier, model, model_source, api_key,
                progress
            ):
                yield (
                    update[0],  # distribution_plot
                    update[1],  # results
                    update[2],  # download_file
                    update[3],  # classify_code_display
                    update[5]   # status
                )
        else:
            yield (None, None, None, None, "Please select a task first.")

    run_btn.click(
        fn=dispatch_run,
        inputs=[task_mode, input_type, spreadsheet_file, spreadsheet_column,
                pdf_file, pdf_folder, pdf_description, pdf_mode,
                image_file, image_folder, image_description] + category_inputs + [model_tier, model, model_source, api_key],
        outputs=[distribution_plot, results, download_file, classify_code_display, status]
    )

    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            input_type, text_input_group, pdf_input_group, image_input_group,
            spreadsheet_file, spreadsheet_column,
            pdf_upload_type, pdf_file, pdf_folder, pdf_description, pdf_mode,
            image_upload_type, image_file, image_folder, image_description,
            task_mode, manual_btn, auto_extract_btn
        ] + category_inputs + [
            add_category_btn, category_count,
            auto_extract_group, max_categories, auto_extract_status,
            categories_group, model_group, run_btn,
            model_tier, model, model_source, api_key, api_key, api_key_status,
            status,
            classify_output_group, distribution_plot, results, download_file, classify_code_display
        ]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
