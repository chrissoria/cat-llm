#!/usr/bin/env python
"""Test PDF report generation with ensemble output."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hf_space'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd

# Import the app's PDF + chart functions
from app import (
    generate_methodology_report_pdf,
    create_distribution_chart,
    create_classification_heatmap,
    _find_model_column_suffix,
    sanitize_model_name,
)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_output')

categories = ["Employment", "Family", "Financial", "Housing",
              "Education", "Safety", "Lifestyle", "Life transition"]

# --- Test 1: Ensemble PDF (from haiku temp ensemble) ---
print("Test 1: Ensemble PDF from haiku_temp_ensemble.csv")
df_ensemble = pd.read_csv(os.path.join(output_dir, 'haiku_temp_ensemble.csv'))
models_list = ["claude-haiku-4-5-20251001"] * 5

# Verify column suffix detection works
for m in models_list:
    suffix = _find_model_column_suffix(df_ensemble, m)
    print(f"  Model: {m} -> detected suffix: {suffix}")

# Check that we find the right columns
suffix = _find_model_column_suffix(df_ensemble, models_list[0])
col = f"category_1_{suffix}"
print(f"  Looking for column '{col}' -> {'FOUND' if col in df_ensemble.columns else 'MISSING'}")

pdf_path = generate_methodology_report_pdf(
    categories=categories,
    model="claude-haiku-4-5-20251001 (5x temperatures)",
    column_name="survey_input",
    num_rows=len(df_ensemble),
    model_source="Ensemble (5 models)",
    filename="haiku_temp_ensemble.csv",
    success_rate=100.0,
    result_df=df_ensemble,
    processing_time=53.0,
    catllm_version="2.1.0",
    python_version="3.11",
    classify_mode="Ensemble",
    models_list=models_list,
    code="# example code\nimport catllm\nresult = catllm.classify(...)",
    consensus_threshold=0.5,
)
# Copy to test_output
import shutil
dest = os.path.join(output_dir, 'test_ensemble_report.pdf')
shutil.copy(pdf_path, dest)
print(f"  PDF saved to: {dest}")
print(f"  PASS")
print()

# --- Test 2: Multi-model ensemble (from threshold_two-thirds.csv) ---
print("Test 2: Model Comparison PDF from threshold_two-thirds.csv")
df_comparison = pd.read_csv(os.path.join(output_dir, 'threshold_two-thirds.csv'))
comp_models = ["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini-2.0-flash"]

for m in comp_models:
    suffix = _find_model_column_suffix(df_comparison, m)
    col = f"category_1_{suffix}"
    print(f"  Model: {m} -> suffix: {suffix} -> col '{col}' {'FOUND' if col in df_comparison.columns else 'MISSING'}")

pdf_path2 = generate_methodology_report_pdf(
    categories=categories,
    model=", ".join(comp_models),
    column_name="survey_input",
    num_rows=len(df_comparison),
    model_source="Model Comparison (3 models)",
    filename="threshold_two-thirds.csv",
    success_rate=100.0,
    result_df=df_comparison,
    processing_time=7.8,
    catllm_version="2.1.0",
    python_version="3.11",
    classify_mode="Model Comparison",
    models_list=comp_models,
    code="# example code\nimport catllm\nresult = catllm.classify(...)",
)
dest2 = os.path.join(output_dir, 'test_comparison_report.pdf')
shutil.copy(pdf_path2, dest2)
print(f"  PDF saved to: {dest2}")
print(f"  PASS")
print()

# --- Test 3: Single model PDF ---
print("Test 3: Single model PDF from no_cot_openai.csv")
df_single = pd.read_csv(os.path.join(output_dir, 'no_cot_openai.csv'))
single_cats = ["Positive sentiment", "Negative sentiment", "Neutral sentiment"]

pdf_path3 = generate_methodology_report_pdf(
    categories=single_cats,
    model="gpt-4o-mini",
    column_name="survey_input",
    num_rows=len(df_single),
    model_source="openai",
    filename="no_cot_openai.csv",
    success_rate=100.0,
    result_df=df_single,
    processing_time=1.7,
    catllm_version="2.1.0",
    python_version="3.11",
    classify_mode="Single Model",
    code="# example code\nimport catllm\nresult = catllm.classify(...)",
)
dest3 = os.path.join(output_dir, 'test_single_report.pdf')
shutil.copy(pdf_path3, dest3)
print(f"  PDF saved to: {dest3}")
print(f"  PASS")

print("\nAll 3 PDFs generated successfully!")
