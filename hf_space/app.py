"""
Gradio app - Step 5g: Add actual catllm classification
"""

import gradio as gr
import pandas as pd
import tempfile
import os
from datetime import datetime

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


def generate_codebook_pdf(categories, model, column_name, num_rows):
    """Generate a PDF codebook explaining the output columns."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    # Create temp file for PDF
    pdf_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_codebook.pdf', delete=False)
    doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles['Normal']

    story = []

    # Title
    story.append(Paragraph("CatLLM Classification Codebook", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))

    # Classification summary
    story.append(Paragraph("Classification Summary", heading_style))
    summary_data = [
        ["Source Column", column_name],
        ["Model Used", model],
        ["Rows Classified", str(num_rows)],
        ["Number of Categories", str(len(categories))],
    ]
    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Category mapping
    story.append(Paragraph("Category Mapping", heading_style))
    story.append(Paragraph("Each category column contains binary values: 1 = present, 0 = not present", normal_style))
    story.append(Spacer(1, 10))

    category_data = [["Column Name", "Category Description"]]
    for i, cat in enumerate(categories, 1):
        category_data.append([f"category_{i}", cat])

    cat_table = Table(category_data, colWidths=[120, 330])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 20))

    # Other columns
    story.append(Paragraph("Other Output Columns", heading_style))
    other_cols = [
        ["Column Name", "Description"],
        ["survey_input", "The original text that was classified"],
        ["model_response", "Raw response from the LLM"],
        ["json", "Extracted JSON with category assignments"],
        ["processing_status", "'success' if classification worked, 'error' if it failed"],
        ["categories_id", "Comma-separated list of category numbers that were assigned"],
    ]
    other_table = Table(other_cols, colWidths=[120, 330])
    other_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
    ]))
    story.append(other_table)
    story.append(Spacer(1, 20))

    # Citation
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. https://github.com/chrissoria/cat-llm", normal_style))

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
        return (
            gr.update(choices=columns, value=columns[0] if columns else None),
            f"Loaded {len(df)} rows. Select column and click Classify."
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"**Error:** {str(e)}"


def classify_data(spreadsheet_file, spreadsheet_column,
                  cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                  model_tier, model, model_source_input, api_key_input):
    """Main classification function."""
    if not CATLLM_AVAILABLE:
        return None, None, "**Error:** catllm package not available"

    all_cats = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    categories = [c.strip() for c in all_cats if c and c.strip()]

    if not categories:
        return None, None, "**Error:** Please enter at least one category"

    actual_model = model

    # Get API key based on tier
    if is_free_model(model, model_tier):
        # Free tier - use Space secrets
        if model in HF_ROUTED_MODELS:
            actual_api_key = os.environ.get("HF_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** HuggingFace API key not configured in Space secrets"
        elif "gpt" in model.lower():
            actual_api_key = os.environ.get("OPENAI_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** OpenAI API key not configured in Space secrets"
        elif "gemini" in model.lower():
            actual_api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** Google API key not configured in Space secrets"
        elif "mistral" in model.lower():
            actual_api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** Mistral API key not configured in Space secrets"
        elif "claude" in model.lower():
            actual_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** Anthropic API key not configured in Space secrets"
        elif "sonar" in model.lower():
            actual_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** Perplexity API key not configured in Space secrets"
        elif "grok" in model.lower():
            actual_api_key = os.environ.get("XAI_API_KEY", "")
            if not actual_api_key:
                return None, None, "**Error:** xAI API key not configured in Space secrets"
        else:
            actual_api_key = os.environ.get("HF_API_KEY", "")
    else:
        # Paid tier - user provides their own API key
        if api_key_input and api_key_input.strip():
            actual_api_key = api_key_input.strip()
        else:
            return None, None, f"**Error:** Please provide your API key for {model}"

    # Use user-selected model_source, or auto-detect if "auto"
    if model_source_input == "auto":
        model_source = get_model_source(actual_model)
    else:
        model_source = model_source_input

    try:
        if not spreadsheet_file:
            return None, None, "**Error:** Please upload a file"
        if not spreadsheet_column:
            return None, None, "**Error:** Please select a column to classify"

        file_path = spreadsheet_file if isinstance(spreadsheet_file, str) else spreadsheet_file.name
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if spreadsheet_column not in df.columns:
            return None, None, f"**Error:** Column '{spreadsheet_column}' not found"

        input_data = df[spreadsheet_column].tolist()

        result = catllm.multi_class(
            survey_input=input_data,
            categories=categories,
            api_key=actual_api_key,
            user_model=actual_model,
            model_source=model_source
        )

        # Save CSV for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='_classified.csv', delete=False) as f:
            result.to_csv(f.name, index=False)
            csv_path = f.name

        # Generate PDF codebook
        pdf_path = generate_codebook_pdf(categories, actual_model, spreadsheet_column, len(input_data))

        return result, [csv_path, pdf_path], f"**Success!** Classified {len(input_data)} responses"

    except Exception as e:
        return None, None, f"**Error:** {str(e)}"


def add_category_field(current_count):
    new_count = min(current_count + 1, MAX_CATEGORIES)
    updates = []
    for i in range(MAX_CATEGORIES):
        updates.append(gr.update(visible=(i < new_count)))
    updates.append(gr.update(visible=(new_count < MAX_CATEGORIES)))
    updates.append(new_count)
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


with gr.Blocks(title="CatLLM - Survey Response Classifier") as demo:
    gr.Image("logo.png", show_label=False, show_download_button=False, height=100, container=False)
    gr.Markdown("# CatLLM - Survey Response Classifier")
    gr.Markdown("Classify survey responses into custom categories using LLMs.")

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
**CatLLM** is an open-source Python package for classifying text data using Large Language Models.

### What It Does
- Classifies survey responses, open-ended text, and other unstructured data into custom categories
- Supports multiple LLM providers: OpenAI, Anthropic, Google, HuggingFace, and more
- Returns structured results with category assignments for each response
- Tested on over 40,000 rows of data with a 100% structured output rate (actual output rate ~99.98% due to occasional server errors)

### Beta Test - We Want Your Feedback!
This app is currently in **beta** and **free to use** while CatLLM is under review for publication. We're actively accepting feedback to improve the tool for researchers.

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
Soria, C. (2025). CatLLM: A Python package for LLM-based text classification.
https://github.com/chrissoria/cat-llm
```
""")

    category_count = gr.State(value=INITIAL_CATEGORIES)

    with gr.Row():
        with gr.Column():
            spreadsheet_file = gr.File(
                label="Upload Survey Data (CSV or Excel)",
                file_types=[".csv", ".xlsx", ".xls"]
            )

            spreadsheet_column = gr.Dropdown(
                label="Column to Classify",
                choices=[],
                info="Select the column containing text to classify"
            )

            gr.Markdown("### Categories")
            category_inputs = []
            for i in range(MAX_CATEGORIES):
                visible = i < INITIAL_CATEGORIES
                cat_input = gr.Textbox(
                    label=f"Category {i+1}",
                    placeholder=f"e.g., {'Positive' if i==0 else 'Negative' if i==1 else 'Neutral'}",
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

            api_key_status = gr.Markdown("**Free tier** - no API key required! We cover the cost while CatLLM is in review.")

            with gr.Row():
                classify_btn = gr.Button("Classify", variant="primary")
                see_code_btn = gr.Button("See the Code", variant="secondary")

        with gr.Column():
            status = gr.Markdown("Ready to classify")
            results = gr.DataFrame(label="Classification Results")
            download_file = gr.File(label="Download Results (CSV + Codebook PDF)", file_count="multiple")
            code_output = gr.Code(
                label="Python Code",
                language="python",
                visible=False
            )

    # Event handlers
    def update_model_tier(tier):
        """Update model choices and API key visibility based on tier."""
        if tier == "Free Models":
            return (
                gr.update(choices=FREE_MODEL_CHOICES, value=FREE_MODEL_CHOICES[0]),
                gr.update(visible=False),
                "**Free tier** - no API key required! We cover the cost while CatLLM is in review."
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

    add_category_btn.click(
        fn=add_category_field,
        inputs=[category_count],
        outputs=category_inputs + [add_category_btn, category_count]
    )

    classify_btn.click(
        fn=classify_data,
        inputs=[spreadsheet_file, spreadsheet_column] + category_inputs + [model_tier, model, model_source, api_key],
        outputs=[results, download_file, status]
    )

    see_code_btn.click(
        fn=generate_code,
        inputs=[spreadsheet_file, spreadsheet_column] + category_inputs + [model_tier, model, model_source],
        outputs=[code_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
