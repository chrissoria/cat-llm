"""
Results display: DataFrame, downloads, code expander.
"""

import io
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from components.visualizations import create_distribution_chart, create_classification_heatmap


COLS_TO_HIDE = ["model_response", "json", "raw_response", "raw_json"]


def render_classify_results(results):
    """Render classification results: visualizations, table, downloads, code."""
    # Visualization selector
    viz_type = st.selectbox(
        "Visualization",
        options=["Category Distribution", "Classification Matrix"],
        key="viz_type",
        help="Distribution shows category percentages. Matrix shows each response's classifications.",
    )

    if viz_type == "Category Distribution":
        fig = create_distribution_chart(
            results["df"],
            results["categories"],
            classify_mode=results.get("classify_mode", "Single Model"),
            models_list=results.get("models_list", []),
        )
        st.pyplot(fig)
        st.caption("Note: Categories are not mutually exclusive -- each item can belong to multiple categories.")
    else:
        fig = create_classification_heatmap(
            results["df"],
            results["categories"],
            classify_mode=results.get("classify_mode", "Single Model"),
            models_list=results.get("models_list", []),
        )
        st.pyplot(fig)
        st.caption("Orange = category present, Black = not present. Each row is one response.")

    # Results dataframe
    display_df = results["df"].copy()
    display_df = display_df.drop(columns=[c for c in COLS_TO_HIDE if c in display_df.columns])
    st.dataframe(display_df, use_container_width=True)

    # Downloads
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        with open(results["csv_path"], "rb") as f:
            st.download_button("Download CSV", data=f, file_name="classified_results.csv", mime="text/csv")
    with col_dl2:
        with open(results["pdf_path"], "rb") as f:
            st.download_button("Download Report", data=f, file_name="methodology_report.pdf", mime="application/pdf")
    with col_dl3:
        plot_buffer = io.BytesIO()
        with PdfPages(plot_buffer) as pdf:
            fig1 = create_distribution_chart(
                results["df"], results["categories"],
                classify_mode=results.get("classify_mode", "Single Model"),
                models_list=results.get("models_list", []),
            )
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)
            fig2 = create_classification_heatmap(
                results["df"], results["categories"],
                classify_mode=results.get("classify_mode", "Single Model"),
                models_list=results.get("models_list", []),
            )
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)
        plot_buffer.seek(0)
        st.download_button("Download Plots", data=plot_buffer, file_name="classification_plots.pdf", mime="application/pdf")

    # Code
    with st.expander("See the Code"):
        st.code(results["code"], language="python")


def render_summarize_results(results):
    """Render summarization results: table, downloads, code."""
    st.info("Summary visualization coming soon!")

    display_df = results["df"].copy()
    display_df = display_df.drop(columns=[c for c in COLS_TO_HIDE if c in display_df.columns])
    st.dataframe(display_df, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        with open(results["csv_path"], "rb") as f:
            st.download_button("Download Results (CSV)", data=f, file_name="summarized_results.csv", mime="text/csv")
    with col_dl2:
        if results.get("pdf_path"):
            with open(results["pdf_path"], "rb") as f:
                st.download_button("Download Report (PDF)", data=f, file_name="methodology_report.pdf", mime="application/pdf")

    with st.expander("See the Code"):
        st.code(results["code"], language="python")


def render_extract_results(results):
    """Render extraction results: category list, counts, code."""
    if results.get("categories"):
        st.success(f"Extracted {len(results['categories'])} categories")
        for i, cat in enumerate(results["categories"], 1):
            st.markdown(f"**{i}.** {cat}")

    if results.get("counts_df") is not None:
        st.dataframe(results["counts_df"], use_container_width=True)

    if results.get("code"):
        with st.expander("See the Code"):
            st.code(results["code"], language="python")


def render_explore_results(results):
    """Render exploration results."""
    if results.get("categories"):
        st.success(f"Discovered {len(results['categories'])} categories")
        for i, cat in enumerate(results["categories"], 1):
            st.markdown(f"**{i}.** {cat}")

    if results.get("code"):
        with st.expander("See the Code"):
            st.code(results["code"], language="python")
