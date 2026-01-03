"""
Gradio app - Step 5g: Add actual catllm classification
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

# Free models (uses Space secrets - no user API key needed)
FREE_MODEL_CHOICES = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-4.1-8B-Instruct:novita",
    "gemini-2.5-flash",
    "gpt-4o",
    "mistral-medium-2505",
    "claude-3-haiku-20240307",
    "sonar",
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
    "meta-llama/Llama-4.1-8B-Instruct:novita",
]


def is_free_model(model, model_tier):
    """Check if using free tier (Space pays for API)."""
    return model_tier == "Free Models"


def generate_methodology_report_pdf(categories, model, column_name, num_rows, model_source, filename, success_rate,
                          result_df=None, processing_time=None, prompt_template=None,
                          data_quality=None, catllm_version=None, python_version=None):
    """Generate a PDF methodology report for reproducibility and transparency."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    # Create temp file for PDF
    pdf_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_methodology_report.pdf', delete=False)
    doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles['Normal']
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=9, leftIndent=20, spaceAfter=3)

    story = []

    # === PAGE 1: Title, About, Category Mapping ===
    story.append(Paragraph("CatLLM Methodology Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 15))

    # About CatLLM - addressing prompt hacking
    story.append(Paragraph("About This Report", heading_style))
    about_text = """This methodology report documents the classification process for reproducibility and transparency. \
CatLLM addresses an issue identified by researchers in "Prompt-Hacking: The New p-Hacking?" (Kosch &amp; Feger, 2025; \
arXiv:2504.14571): researchers could keep modifying prompts to obtain outputs that support desired conclusions, and \
this variability in pseudo-natural language poses a challenge for reproducibility since each prompt, even if only \
slightly altered, can yield different outputs, making it impossible to replicate findings reliably. CatLLM restricts \
the prompt to a standard template that is impartial to the researcher's inclinations, ensuring \
consistent and reproducible results."""
    story.append(Paragraph(about_text, normal_style))
    story.append(Spacer(1, 15))

    # Category mapping
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

    # Other columns
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

    # Citation at end of page 1
    story.append(Spacer(1, 30))
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316", normal_style))

    # === PAGE 2: Sample Results ===
    if result_df is not None and len(result_df) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Sample Results (First 5 Rows)", title_style))
        story.append(Paragraph("Example classifications showing original text and assigned categories:", normal_style))
        story.append(Spacer(1, 15))

        sample_data = [["Original Text (truncated)", "Assigned Categories"]]
        sample_df = result_df.head(5)

        for _, row in sample_df.iterrows():
            # Get original text, truncate to 80 chars
            original_text = str(row.get('survey_input', ''))[:80]
            if len(str(row.get('survey_input', ''))) > 80:
                original_text += "..."

            # Get assigned categories
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

    # === PAGE 3: Category Distribution ===
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

    # === PAGE 4: Classification Summary (Expanded) ===
    story.append(PageBreak())
    story.append(Paragraph("Classification Summary", title_style))
    story.append(Spacer(1, 15))

    # Basic summary
    story.append(Paragraph("Classification Details", heading_style))
    summary_data = [
        ["Source File", filename],
        ["Source Column", column_name],
        ["Model Used", model],
        ["Model Source", model_source],
        ["Temperature", "default"],
        ["Rows Classified", str(num_rows)],
        ["Number of Categories", str(len(categories))],
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

    # Processing Time
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

    # Data Quality Notes
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

    # Version Information
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

    # === PAGE 5: Prompt Template ===
    story.append(PageBreak())
    story.append(Paragraph("Prompt Template Used", title_style))
    story.append(Paragraph("The following prompt template was sent to the LLM for each classification:", normal_style))
    story.append(Spacer(1, 15))

    if prompt_template:
        # Show the template with placeholders
        story.append(Paragraph("Template with Placeholders:", heading_style))
        story.append(Spacer(1, 8))

        for line in prompt_template.split('\n'):
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if escaped_line.strip():
                story.append(Paragraph(escaped_line, code_style))
            else:
                story.append(Spacer(1, 5))

        story.append(Spacer(1, 20))

        # Show example with actual categories
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

    # === PAGE 6: Reproducibility Code ===
    story.append(PageBreak())
    story.append(Paragraph("Reproducibility Code", title_style))
    story.append(Paragraph("Use the following Python code to reproduce this classification:", normal_style))
    story.append(Spacer(1, 15))

    # Build categories list string
    categories_str = ", ".join([f'"{cat}"' for cat in categories])

    code_text = f'''import catllm
import pandas as pd

# Load your survey data
df = pd.read_csv("{filename}")

# Define your categories
categories = [{categories_str}]

# Classify the responses
result = catllm.multi_class(
    survey_input=df["{column_name}"].tolist(),
    categories=categories,
    api_key="YOUR_API_KEY",
    user_model="{model}",
    model_source="{model_source}"
)

# View results
print(result)

# Save to CSV
result.to_csv("classified_results.csv", index=False)'''

    # Split code into lines and add each as a paragraph
    for line in code_text.split('\n'):
        if line.strip() == '':
            story.append(Spacer(1, 5))
        else:
            # Escape special characters for PDF
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(escaped_line, code_style))

    doc.build(story)
    return pdf_file.name


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
    # All models routed through HuggingFace (including novita, groq variants)
    elif any(x in model_lower for x in [":novita", ":groq", "qwen", "llama", "deepseek"]):
        return "huggingface"
    return "huggingface"


def load_example_dataset():
    """Load the example dataset for users to try the app."""
    example_path = "example_data.csv"
    try:
        df = pd.read_csv(example_path)
        columns = df.columns.tolist()
        return (
            example_path,  # file path
            gr.update(choices=columns, value=columns[0] if columns else None),  # column dropdown
            f"Loaded example dataset ({len(df)} rows). Select column and click Classify."  # status
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

        # Warning for large datasets
        if num_rows > 1000:
            est_minutes = round(num_rows * 1.5 / 60)  # ~1.5 seconds per row estimate
            status_msg = f"⚠️ **Large dataset** ({num_rows:,} rows). Classification may take ~{est_minutes} minutes. Select column and click Classify."
        else:
            status_msg = f"Loaded {num_rows:,} rows. Select column and click Classify."

        return (
            gr.update(choices=columns, value=columns[0] if columns else None),
            status_msg
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"**Error:** {str(e)}"


def classify_data(input_type, spreadsheet_file, spreadsheet_column,
                  pdf_file, pdf_description, pdf_mode,
                  cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                  model_tier, model, model_source_input, api_key_input):
    """Main classification function with progress updates. Yields status updates then final results."""
    if not CATLLM_AVAILABLE:
        yield None, None, None, None, "**Error:** catllm package not available"
        return

    all_cats = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    categories = [c.strip() for c in all_cats if c and c.strip()]

    if not categories:
        yield None, None, None, None, "**Error:** Please enter at least one category"
        return

    actual_model = model

    # Get API key based on tier
    if is_free_model(model, model_tier):
        # Free tier - use Space secrets
        if model in HF_ROUTED_MODELS:
            actual_api_key = os.environ.get("HF_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** HuggingFace API key not configured in Space secrets"
                return
        elif "gpt" in model.lower():
            actual_api_key = os.environ.get("OPENAI_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** OpenAI API key not configured in Space secrets"
                return
        elif "gemini" in model.lower():
            actual_api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** Google API key not configured in Space secrets"
                return
        elif "mistral" in model.lower():
            actual_api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** Mistral API key not configured in Space secrets"
                return
        elif "claude" in model.lower():
            actual_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** Anthropic API key not configured in Space secrets"
                return
        elif "sonar" in model.lower():
            actual_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** Perplexity API key not configured in Space secrets"
                return
        elif "grok" in model.lower():
            actual_api_key = os.environ.get("XAI_API_KEY", "")
            if not actual_api_key:
                yield None, None, None, None, "**Error:** xAI API key not configured in Space secrets"
                return
        else:
            actual_api_key = os.environ.get("HF_API_KEY", "")
    else:
        # Paid tier - user provides their own API key
        if api_key_input and api_key_input.strip():
            actual_api_key = api_key_input.strip()
        else:
            yield None, None, None, None, f"**Error:** Please provide your API key for {model}"
            return

    # Use user-selected model_source, or auto-detect if "auto"
    if model_source_input == "auto":
        model_source = get_model_source(actual_model)
    else:
        model_source = model_source_input

    try:
        # Determine if we're processing text or PDF
        is_pdf_mode = input_type == "PDF Documents"

        if is_pdf_mode:
            # PDF validation
            if not pdf_file:
                yield None, None, None, None, "**Error:** Please upload a PDF file"
                return

            pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

            # Map UI mode to function parameter
            mode_mapping = {
                "Image (visual documents)": "image",
                "Text (text-heavy)": "text",
                "Both (comprehensive)": "both"
            }
            actual_pdf_mode = mode_mapping.get(pdf_mode, "image")

            # Progress update
            yield None, None, None, None, f"⏳ **Loading PDF...** Processing document."

            # Data quality placeholder for PDFs
            data_quality = {
                'null_count': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'error_count': 0
            }

            # Progress update: starting classification
            yield None, None, None, None, f"🔄 **Classifying PDF pages...** This may take a moment."

            # Capture timing
            start_time = time.time()

            result = catllm.pdf_multi_class(
                pdf_description=pdf_description or "document",
                pdf_input=pdf_path,
                categories=categories,
                api_key=actual_api_key,
                user_model=actual_model,
                model_source=model_source,
                mode=actual_pdf_mode
            )

            processing_time = time.time() - start_time
            num_items = len(result)
            original_filename = pdf_path.split("/")[-1]
            column_name = "PDF Pages"

            # Build prompt template for PDF
            prompt_template = f'''Categorize this PDF page from "{pdf_description or 'document'}" into the following categories that apply:
{{categories}}

Let's think step by step:
1. First, identify the main themes present in this page
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values.'''

        else:
            # Text data validation
            if not spreadsheet_file:
                yield None, None, None, None, "**Error:** Please upload a file"
                return
            if not spreadsheet_column:
                yield None, None, None, None, "**Error:** Please select a column to classify"
                return

            file_path = spreadsheet_file if isinstance(spreadsheet_file, str) else spreadsheet_file.name
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            if spreadsheet_column not in df.columns:
                yield None, None, None, None, f"**Error:** Column '{spreadsheet_column}' not found"
                return

            input_data = df[spreadsheet_column].tolist()

            # Progress update: data loaded
            yield None, None, None, None, f"⏳ **Loading data...** Found {len(input_data)} responses to classify."

            # Calculate data quality metrics before classification
            text_series = df[spreadsheet_column].dropna().astype(str)
            data_quality = {
                'null_count': int(df[spreadsheet_column].isna().sum()),
                'avg_length': round(text_series.str.len().mean(), 1) if len(text_series) > 0 else 0,
                'min_length': int(text_series.str.len().min()) if len(text_series) > 0 else 0,
                'max_length': int(text_series.str.len().max()) if len(text_series) > 0 else 0,
                'error_count': 0  # Will be updated after classification
            }

            # Progress update: starting classification
            yield None, None, None, None, f"🔄 **Classifying {len(input_data)} responses...** This may take a moment."

            # Capture timing
            start_time = time.time()

            result = catllm.multi_class(
                survey_input=input_data,
                categories=categories,
                api_key=actual_api_key,
                user_model=actual_model,
                model_source=model_source
            )

            processing_time = time.time() - start_time
            num_items = len(input_data)
            original_filename = file_path.split("/")[-1]
            column_name = spreadsheet_column

            # Build prompt template for documentation (chain of thought - default)
            prompt_template = '''Categorize this survey response "{response}" into the following categories that apply:
{categories}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Provide your work in JSON format where the number belonging to each category is the key and a 1 if the category is present and a 0 if it is not present as key values.'''

        # Update error count from results
        if 'processing_status' in result.columns:
            data_quality['error_count'] = int((result['processing_status'] == 'error').sum())

        # Save CSV for download
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

        # Progress update: generating report
        yield None, None, None, None, f"📄 **Generating methodology report...** Classification complete in {processing_time:.1f}s."

        # Generate PDF methodology report with all new data
        report_pdf_path = generate_methodology_report_pdf(
            categories=categories,
            model=actual_model,
            column_name=column_name,
            num_rows=num_items,
            model_source=model_source,
            filename=original_filename,
            success_rate=success_rate,
            result_df=result,
            processing_time=processing_time,
            prompt_template=prompt_template,
            data_quality=data_quality,
            catllm_version=catllm_version,
            python_version=python_version
        )

        # Build distribution data and create matplotlib plot
        dist_data = []
        total_rows = len(result)
        for i, cat in enumerate(categories, 1):
            col_name = f"category_{i}"
            if col_name in result.columns:
                count = int(result[col_name].sum())
                pct = (count / total_rows) * 100 if total_rows > 0 else 0
                dist_data.append({
                    "Category": cat,
                    "Percentage": round(pct, 1)
                })

        # Create matplotlib horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(4, len(dist_data) * 0.8)))
        categories_list = [d["Category"] for d in dist_data]
        percentages = [d["Percentage"] for d in dist_data]

        # Reverse order so first category is at top
        categories_list = categories_list[::-1]
        percentages = percentages[::-1]

        bars = ax.barh(categories_list, percentages, color='#2563eb')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_title('Category Distribution (%)', fontsize=14, fontweight='bold')

        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=10)

        plt.tight_layout()
        distribution_fig = fig

        # Build sample results DataFrame (first 5 rows)
        sample_data = []
        # Determine the input column name based on mode
        input_col = 'pdf_input' if is_pdf_mode else 'survey_input'
        input_label = "PDF Page" if is_pdf_mode else "Original Text"

        for _, row in result.head(5).iterrows():
            original_text = str(row.get(input_col, ''))[:100]
            if len(str(row.get(input_col, ''))) > 100:
                original_text += "..."
            assigned = row.get('categories_id', '')
            if pd.isna(assigned) or assigned == '':
                assigned = "None"
            sample_data.append({
                input_label: original_text,
                "Assigned Categories": str(assigned)
            })
        sample_df = pd.DataFrame(sample_data)

        # Determine success message based on mode
        item_type = "pages" if is_pdf_mode else "responses"

        # Final yield: distribution plot (visible), samples (visible), full results (visible), files, status
        yield (
            gr.update(value=distribution_fig, visible=True),
            gr.update(value=sample_df, visible=True),
            gr.update(value=result, visible=True),
            [csv_path, report_pdf_path],
            f"✅ **Success!** Classified {num_items} {item_type} in {processing_time:.1f}s"
        )

    except Exception as e:
        yield None, None, None, None, f"**Error:** {str(e)}"


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
        "Text Data (CSV/Excel)",  # input_type
        gr.update(visible=True),  # text_input_group
        gr.update(visible=False),  # pdf_input_group
        None,  # spreadsheet_file
        gr.update(choices=[], value=None),  # spreadsheet_column
        None,  # pdf_file
        "",  # pdf_description
        "Image (visual documents)",  # pdf_mode
    ]
    # Reset category inputs (first 3 visible, rest hidden, all empty)
    for i in range(MAX_CATEGORIES):
        updates.append(gr.update(value="", visible=(i < INITIAL_CATEGORIES)))
    updates.extend([
        gr.update(visible=True),  # add_category_btn
        INITIAL_CATEGORIES,  # category_count
        "Free Models",  # model_tier
        FREE_MODEL_CHOICES[0],  # model
        "auto",  # model_source
        "",  # api_key
        "**Free tier** - no API key required! This app is made free possible by Bashir Ahmed's generous fellowship support.",  # api_key_status
        "Ready to classify",  # status
        gr.update(value=None, visible=False),  # distribution_plot
        gr.update(value=None, visible=False),  # sample_results
        gr.update(value=None, visible=False),  # results
        None,  # download_file
        gr.update(value="", visible=False),  # code_output
    ])
    return updates


def generate_code(spreadsheet_file, spreadsheet_column,
                  cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                  model_tier, model, model_source_input):
    """Generate Python code snippet based on user inputs."""
    all_cats = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    categories = [c.strip() for c in all_cats if c and c.strip()]

    actual_model = model

    # Determine model source
    if model_source_input == "auto":
        model_source = get_model_source(actual_model)
    else:
        model_source = model_source_input

    # Get filename from uploaded file
    if spreadsheet_file:
        file_path = spreadsheet_file if isinstance(spreadsheet_file, str) else spreadsheet_file.name
        filename = file_path.split("/")[-1]
    else:
        filename = "your_survey_data.csv"

    # Build categories list string
    categories_str = ", ".join([f'"{cat}"' for cat in categories]) if categories else '"Category1", "Category2"'

    # Build the code snippet
    code = f'''import catllm
import pandas as pd

# Load your survey data
df = pd.read_csv("{filename}")

# Define your categories
categories = [{categories_str}]

# Classify the responses
result = catllm.multi_class(
    survey_input=df["{spreadsheet_column or 'your_column'}"].tolist(),
    categories=categories,
    api_key="YOUR_API_KEY",  # Replace with your actual API key
    user_model="{actual_model}",
    model_source="{model_source}"
)

# View results
print(result)

# Save to CSV
result.to_csv("classified_results.csv", index=False)
'''

    return gr.update(value=code, visible=True)


custom_css = """
* {
    font-family: Helvetica, Arial, sans-serif !important;
}
"""

with gr.Blocks(title="CatLLM - Research Data Classifier", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Image("logo.png", show_label=False, show_download_button=False, height=115, container=False)
    gr.Markdown("# CatLLM - Research Data Classifier")
    gr.Markdown("Classify text data (CSV/Excel) and PDF documents into custom categories using LLMs.")

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
⚠️ **Privacy Notice:** Your data is sent to third-party LLM APIs for classification. Do not upload sensitive, confidential, or personally identifiable information (PII).

---

**CatLLM** is an open-source Python package for classifying text and document data using Large Language Models.

### What It Does
- Classifies survey responses, open-ended text, PDF documents, and other unstructured data into custom categories
- Supports multiple LLM providers: OpenAI, Anthropic, Google, HuggingFace, and more
- Returns structured results with category assignments for each response or PDF page
- Tested on over 40,000 rows of data with a 100% structured output rate (actual output rate ~99.98% due to occasional server errors)

### Beta Test - We Want Your Feedback!
This app is currently in **beta** and **free to use** while CatLLM is under review for publication, made possible by **Bashir Ahmed's generous fellowship support**. We're actively accepting feedback to improve the tool for researchers.

- Found a bug? Have a feature request? Please open an issue or submit a PR on [GitHub](https://github.com/chrissoria/cat-llm) so other researchers can benefit!
- **Open to collaboration** on research projects involving LLM-based classification
- Reach out directly: [chrissoria@berkeley.edu](mailto:chrissoria@berkeley.edu)
- Connect on [LinkedIn](https://www.linkedin.com/in/chris-soria-9340931a/)

### Links
- 📦 **PyPI**: [pip install cat-llm](https://pypi.org/project/cat-llm/)
- 💻 **GitHub**: [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)
- 🌐 **Author**: [christophersoria.com](https://christophersoria.com)

### Citation
If you use CatLLM in your research, please cite:
```
Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316
```
""")

    category_count = gr.State(value=INITIAL_CATEGORIES)

    with gr.Row():
        with gr.Column():
            # Input type toggle
            input_type = gr.Radio(
                choices=["Text Data (CSV/Excel)", "PDF Documents"],
                value="Text Data (CSV/Excel)",
                label="Input Type"
            )

            # Text data input group
            with gr.Group(visible=True) as text_input_group:
                spreadsheet_file = gr.File(
                    label="Upload Data (CSV or Excel)",
                    file_types=[".csv", ".xlsx", ".xls"]
                )
                example_btn = gr.Button("📋 Try Example Dataset", variant="secondary", size="sm")

                spreadsheet_column = gr.Dropdown(
                    label="Column to Classify",
                    choices=[],
                    info="Select the column containing text to classify"
                )

            # PDF input group
            with gr.Group(visible=False) as pdf_input_group:
                pdf_file = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"]
                )
                pdf_description = gr.Textbox(
                    label="Document Description",
                    placeholder="e.g., 'research papers', 'interview transcripts', 'policy documents'",
                    info="Helps the LLM understand the context of your PDF"
                )
                pdf_mode = gr.Radio(
                    choices=["Image (visual documents)", "Text (text-heavy)", "Both (comprehensive)"],
                    value="Image (visual documents)",
                    label="Processing Mode",
                    info="Image mode is best for scans/charts; Text mode is faster for text-heavy docs"
                )

            gr.Markdown("### Categories")
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

            add_category_btn = gr.Button("+ Add More Categories", variant="secondary", size="sm")

            gr.Markdown("### Model")
            model_tier = gr.Radio(
                choices=["Free Models", "Bring Your Own Key"],
                value="Free Models",
                label="Model Tier",
                info="Free models use our API keys. 'Bring Your Own Key' lets you use your own API key."
            )

            model = gr.Dropdown(
                choices=FREE_MODEL_CHOICES,
                value="Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
                label="Model",
                allow_custom_value=True
            )

            model_source = gr.Dropdown(
                choices=["auto", "openai", "anthropic", "google", "mistral", "xai", "huggingface", "perplexity"],
                value="auto",
                label="Model Source",
                info="Auto-detects from model name, or select manually."
            )

            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Enter your API key",
                info="Required for 'Bring Your Own Key' tier",
                visible=False
            )

            api_key_status = gr.Markdown("**Free tier** - no API key required! This app is made free possible by Bashir Ahmed's generous fellowship support.")

            classify_btn = gr.Button("Classify", variant="primary", size="lg")
            with gr.Row():
                see_code_btn = gr.Button("See the Code", variant="secondary")
                reset_btn = gr.Button("Reset", variant="stop")

        with gr.Column():
            status = gr.Markdown("Ready to classify")
            distribution_plot = gr.Plot(
                label="Category Distribution (%)",
                visible=False
            )
            sample_results = gr.DataFrame(label="Sample Results (First 5 Rows)", visible=False)
            results = gr.DataFrame(label="Full Classification Results", visible=False)
            download_file = gr.File(label="Download Results (CSV + Methodology Report)", file_count="multiple")
            code_output = gr.Code(
                label="Python Code",
                language="python",
                visible=False
            )

    # Event handlers
    def switch_input_type(input_type_val):
        """Toggle visibility between text and PDF input groups."""
        if input_type_val == "Text Data (CSV/Excel)":
            return gr.update(visible=True), gr.update(visible=False), "Ready to classify text data"
        else:
            return gr.update(visible=False), gr.update(visible=True), "Ready to classify PDF document"

    input_type.change(
        fn=switch_input_type,
        inputs=[input_type],
        outputs=[text_input_group, pdf_input_group, status]
    )

    def update_model_tier(tier):
        """Update model choices and API key visibility based on tier."""
        if tier == "Free Models":
            return (
                gr.update(choices=FREE_MODEL_CHOICES, value=FREE_MODEL_CHOICES[0]),
                gr.update(visible=False),
                "**Free tier** - no API key required! This app is made free possible by Bashir Ahmed's generous fellowship support."
            )
        else:
            return (
                gr.update(choices=PAID_MODEL_CHOICES, value=PAID_MODEL_CHOICES[0]),
                gr.update(visible=True),
                "**Bring Your Own Key** - enter your API key below."
            )

    model_tier.change(
        fn=update_model_tier,
        inputs=[model_tier],
        outputs=[model, api_key, api_key_status]
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

    add_category_btn.click(
        fn=add_category_field,
        inputs=[category_count],
        outputs=category_inputs + [add_category_btn, category_count]
    )

    classify_btn.click(
        fn=classify_data,
        inputs=[input_type, spreadsheet_file, spreadsheet_column, pdf_file, pdf_description, pdf_mode] + category_inputs + [model_tier, model, model_source, api_key],
        outputs=[distribution_plot, sample_results, results, download_file, status]
    )

    see_code_btn.click(
        fn=generate_code,
        inputs=[spreadsheet_file, spreadsheet_column] + category_inputs + [model_tier, model, model_source],
        outputs=[code_output]
    )

    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[input_type, text_input_group, pdf_input_group, spreadsheet_file, spreadsheet_column, pdf_file, pdf_description, pdf_mode] + category_inputs + [add_category_btn, category_count, model_tier, model, model_source, api_key, api_key_status, status, distribution_plot, sample_results, results, download_file, code_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
