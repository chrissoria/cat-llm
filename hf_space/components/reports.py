"""
Unified methodology PDF report generation (ReportLab).
"""

import io
import tempfile
from datetime import datetime

from components.visualizations import (
    _find_all_model_suffixes,
    create_distribution_chart,
    create_classification_heatmap,
)


def generate_methodology_report_pdf(
    task_type="classify",
    categories=None,
    model="",
    column_name="",
    num_rows=0,
    model_source="",
    filename="",
    success_rate=100.0,
    result_df=None,
    processing_time=None,
    catllm_version=None,
    python_version=None,
    input_type="text",
    description=None,
    classify_mode="Single Model",
    models_list=None,
    code=None,
    consensus_threshold=None,
    domain="general",
    # Summarize-specific
    focus=None,
    max_length=None,
):
    """Generate a PDF methodology report for any task type."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    import matplotlib.pyplot as plt

    pdf_file = tempfile.NamedTemporaryFile(mode="wb", suffix="_methodology_report.pdf", delete=False)
    doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles["Normal"]
    code_style = ParagraphStyle("Code", parent=styles["Normal"], fontName="Courier", fontSize=9, leftIndent=20, spaceAfter=3)

    story = []

    # Title
    titles = {
        "classify": "CatLLM Classification Report",
        "extract_and_classify": "CatLLM Extraction &amp; Classification Report",
        "summarize": "CatLLM Summarization Report",
        "extract": "CatLLM Category Extraction Report",
        "explore": "CatLLM Category Exploration Report",
    }
    report_title = titles.get(task_type, "CatLLM Report")
    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    if domain != "general":
        story.append(Paragraph(f"Domain: {domain.replace('_', ' ').title()}", normal_style))
    story.append(Spacer(1, 15))

    # About
    story.append(Paragraph("About This Report", heading_style))
    about_texts = {
        "classify": "This report documents the classification process for reproducibility and transparency.",
        "extract_and_classify": "This report documents the automated category extraction and classification process.",
        "summarize": "This report documents the automated summarization process.",
        "extract": "This report documents the automated category extraction process.",
        "explore": "This report documents the category exploration and saturation analysis.",
    }
    story.append(Paragraph(about_texts.get(task_type, ""), normal_style))
    story.append(Spacer(1, 15))

    # Category mapping (classify tasks)
    if categories and task_type in ("classify", "extract_and_classify"):
        story.append(Paragraph("Category Mapping", heading_style))
        if classify_mode in ("Ensemble", "Model Comparison") and result_df is not None:
            all_suffixes = _find_all_model_suffixes(result_df)
            category_data = [["Column Name", "Category Description"]]
            for i, cat in enumerate(categories, 1):
                for suffix in all_suffixes:
                    category_data.append([f"category_{i}_{suffix}", f"{cat} ({suffix})"])
                category_data.append([f"category_{i}_consensus", f"{cat} (consensus)"])
                category_data.append([f"category_{i}_agreement", f"{cat} (agreement score)"])
            cat_table = Table(category_data, colWidths=[200, 250])
        else:
            story.append(Paragraph("Each category column: 1 = present, 0 = not present", normal_style))
            story.append(Spacer(1, 8))
            category_data = [["Column Name", "Category Description"]]
            for i, cat in enumerate(categories, 1):
                category_data.append([f"category_{i}", cat])
            cat_table = Table(category_data, colWidths=[120, 330])

        cat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 1), (0, -1), colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(cat_table)
        story.append(Spacer(1, 15))

    # Citation
    story.append(Spacer(1, 30))
    story.append(Paragraph("Citation", heading_style))
    story.append(Paragraph("If you use CatLLM in your research, please cite:", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316",
        normal_style,
    ))

    # Summary page
    story.append(PageBreak())
    summary_title = {
        "classify": "Classification Summary",
        "extract_and_classify": "Classification Summary",
        "summarize": "Summarization Summary",
        "extract": "Extraction Summary",
        "explore": "Exploration Summary",
    }
    story.append(Paragraph(summary_title.get(task_type, "Summary"), title_style))
    story.append(Spacer(1, 15))

    summary_data = [
        ["Source File", filename],
        ["Source Column", column_name],
        ["Model(s) Used", model],
        ["Model Source", model_source],
        ["Items Processed", str(num_rows)],
        ["Success Rate", f"{success_rate:.2f}%"],
    ]
    if classify_mode != "Single Model" and task_type in ("classify", "extract_and_classify"):
        summary_data.insert(2, ["Classification Mode", classify_mode])
    if consensus_threshold is not None and classify_mode == "Ensemble":
        labels = {0.5: "Majority (50%+)", 0.67: "Two-Thirds (67%+)", 1.0: "Unanimous (100%)"}
        summary_data.append(["Consensus Threshold", labels.get(consensus_threshold, f"{consensus_threshold:.0%}")])
    if focus:
        summary_data.append(["Focus", focus])
    if max_length:
        summary_data.append(["Max Length", f"{max_length} words"])
    if categories:
        summary_data.append(["Number of Categories", str(len(categories))])

    summary_table = Table(summary_data, colWidths=[150, 300])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Ensemble agreement scores
    if classify_mode == "Ensemble" and result_df is not None and categories:
        agreement_cols = [f"category_{i}_agreement" for i in range(1, len(categories) + 1)]
        if all(col in result_df.columns for col in agreement_cols):
            story.append(Paragraph("Ensemble Agreement Scores", heading_style))
            agree_data = [["Category", "Mean Agreement", "Min Agreement"]]
            for i, cat in enumerate(categories, 1):
                col = f"category_{i}_agreement"
                agree_data.append([cat, f"{result_df[col].mean():.1%}", f"{result_df[col].min():.1%}"])
            agree_table = Table(agree_data, colWidths=[200, 125, 125])
            agree_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(agree_table)
            story.append(Spacer(1, 15))

    # Processing time
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
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(time_table)

    # Version info
    story.append(Spacer(1, 15))
    story.append(Paragraph("Version Information", heading_style))
    version_data = [
        ["CatLLM Version", catllm_version or "unknown"],
        ["Python Version", python_version or "unknown"],
        ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    version_table = Table(version_data, colWidths=[180, 270])
    version_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(version_table)

    # Reproducibility code
    if code:
        story.append(PageBreak())
        story.append(Paragraph("Reproducibility Code", title_style))
        story.append(Paragraph("Use this Python code to reproduce the results:", normal_style))
        story.append(Spacer(1, 10))
        for line in code.strip().split("\n"):
            escaped_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if escaped_line.strip():
                story.append(Paragraph(escaped_line, code_style))
            else:
                story.append(Spacer(1, 6))

    # Visualizations (classify tasks only)
    if result_df is not None and categories and task_type in ("classify", "extract_and_classify"):
        from reportlab.platypus import Image as RLImage

        story.append(PageBreak())
        story.append(Paragraph("Category Distribution", title_style))
        try:
            fig1 = create_distribution_chart(result_df, categories, classify_mode, models_list)
            img_buffer1 = io.BytesIO()
            fig1.savefig(img_buffer1, format="png", dpi=150, bbox_inches="tight")
            img_buffer1.seek(0)
            plt.close(fig1)
            img_temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_temp1.write(img_buffer1.read())
            img_temp1.close()
            story.append(RLImage(img_temp1.name, width=450, height=250))
        except Exception as e:
            story.append(Paragraph(f"Could not generate chart: {e}", normal_style))

        story.append(PageBreak())
        story.append(Paragraph("Classification Matrix", title_style))
        try:
            fig2 = create_classification_heatmap(result_df, categories, classify_mode, models_list)
            img_buffer2 = io.BytesIO()
            fig2.savefig(img_buffer2, format="png", dpi=150, bbox_inches="tight")
            img_buffer2.seek(0)
            plt.close(fig2)
            img_temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_temp2.write(img_buffer2.read())
            img_temp2.close()
            story.append(RLImage(img_temp2.name, width=450, height=300))
        except Exception as e:
            story.append(Paragraph(f"Could not generate matrix: {e}", normal_style))

    doc.build(story)
    return pdf_file.name
