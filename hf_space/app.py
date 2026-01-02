"""
Gradio app - Step 5g: Add actual catllm classification
"""

import gradio as gr
import pandas as pd
import tempfile
import os

# Import catllm
try:
    import catllm
    CATLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import catllm: {e}")
    CATLLM_AVAILABLE = False

MAX_CATEGORIES = 10
INITIAL_CATEGORIES = 3

MODEL_CHOICES = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita (Free)",
    "deepseek-ai/DeepSeek-V3.1:novita (Free)",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq (Free)",
    "gpt-4o",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash",
]

HF_FREE_MODELS = {
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita (Free)": "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita (Free)": "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq (Free)": "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq",
}


def is_free_model(model):
    return "(Free)" in model


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


def classify_data(input_type, spreadsheet_file, spreadsheet_column,
                  cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10,
                  model, model_source_input, api_key_input):
    """Main classification function."""
    if not CATLLM_AVAILABLE:
        return None, None, "**Error:** catllm package not available"

    all_cats = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10]
    categories = [c.strip() for c in all_cats if c and c.strip()]

    if not categories:
        return None, None, "**Error:** Please enter at least one category"

    # Get API key - priority: user input > environment variable
    if is_free_model(model):
        actual_api_key = os.environ.get("HF_API_KEY", "")
        actual_model = HF_FREE_MODELS.get(model, model.replace(" (Free)", ""))
        if not actual_api_key:
            return None, None, "**Error:** HuggingFace API key not configured in Space secrets"
    else:
        # For paid models, check user input first, then environment
        actual_model = model
        if api_key_input and api_key_input.strip():
            actual_api_key = api_key_input.strip()
        else:
            # Try to get from environment based on model
            if "gpt" in model.lower():
                actual_api_key = os.environ.get("OPENAI_API_KEY", "")
            elif "claude" in model.lower():
                actual_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            elif "gemini" in model.lower():
                actual_api_key = os.environ.get("GOOGLE_API_KEY", "")
            else:
                actual_api_key = ""

            if not actual_api_key:
                return None, None, f"**Error:** Please provide an API key for {model}"

    # Use user-selected model_source, or auto-detect if "auto"
    if model_source_input == "auto":
        model_source = get_model_source(actual_model)
    else:
        model_source = model_source_input

    try:
        if input_type == "Spreadsheet":
            if not spreadsheet_file:
                return None, None, "**Error:** Please upload a spreadsheet"
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

            # Return the result directly from catllm (already properly formatted)
            # Save for download
            with tempfile.NamedTemporaryFile(mode='w', suffix='_classified.csv', delete=False) as f:
                result.to_csv(f.name, index=False)
                download_path = f.name

            return result, download_path, f"**Success!** Classified {len(input_data)} items"

        else:
            return None, None, f"**Error:** {input_type} not yet supported in this version"

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


with gr.Blocks(title="catllm Classifier") as demo:
    gr.Markdown("# catllm Classifier")
    gr.Markdown("Classify spreadsheets into custom categories using LLMs.")

    category_count = gr.State(value=INITIAL_CATEGORIES)

    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(
                choices=["Spreadsheet", "Image", "PDF"],
                value="Spreadsheet",
                label="Input Type"
            )

            spreadsheet_file = gr.File(
                label="Upload CSV or Excel File",
                file_types=[".csv", ".xlsx", ".xls"]
            )

            with gr.Row():
                spreadsheet_column = gr.Dropdown(
                    label="Column to Classify",
                    choices=[],
                    info="Select the column containing text to classify"
                )
                load_cols_btn = gr.Button("Load Columns", size="sm")

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
            model = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="Qwen/Qwen3-VL-235B-A22B-Instruct:novita (Free)",
                label="Model",
                allow_custom_value=True
            )

            model_source = gr.Dropdown(
                choices=["auto", "openai", "anthropic", "google", "mistral", "xai", "huggingface", "perplexity"],
                value="auto",
                label="Model Source",
                info="Auto-detects from model name, or select manually. Use 'huggingface' for Qwen/Llama/DeepSeek models."
            )

            api_key = gr.Textbox(
                label="API Key (optional)",
                type="password",
                placeholder="Enter your API key, or leave blank to use Space secrets",
                info="For paid models, enter your key or configure in Space secrets"
            )

            api_key_status = gr.Markdown("**Free model selected** - no API key required!")

            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            status = gr.Markdown("Ready to classify")
            results = gr.DataFrame(label="Classification Results")
            download_file = gr.File(label="Download Results")

    # Event handlers
    def update_api_key_status(selected_model):
        if is_free_model(selected_model):
            return "**Free model selected** - no API key required!"
        elif "gpt" in selected_model.lower():
            return "**OpenAI model** - using OPENAI_API_KEY from secrets (or enter your own)"
        elif "claude" in selected_model.lower():
            return "**Anthropic model** - using ANTHROPIC_API_KEY from secrets (or enter your own)"
        elif "gemini" in selected_model.lower():
            return "**Google model** - using GOOGLE_API_KEY from secrets (or enter your own)"
        else:
            return "**Paid model** - enter your API key or configure in Space secrets"

    model.change(
        fn=update_api_key_status,
        inputs=[model],
        outputs=[api_key_status]
    )

    load_cols_btn.click(
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
        inputs=[input_type, spreadsheet_file, spreadsheet_column] + category_inputs + [model, model_source, api_key],
        outputs=[results, download_file, status]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
