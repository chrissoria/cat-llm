"""
Visualization helpers: distribution chart, classification heatmap.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def sanitize_model_name(model: str) -> str:
    """Convert model name to column-safe suffix (matches catllm logic)."""
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", model)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_").lower()
    return sanitized[:40]


def _find_model_column_suffix(result_df, model_name):
    """Find the actual column suffix used for a model in the DataFrame."""
    sanitized = sanitize_model_name(model_name)
    prefix = f"category_1_{sanitized}"
    for col in result_df.columns:
        if col.startswith(prefix):
            return col[len("category_1_"):]
    return sanitized


def _find_all_model_suffixes(result_df):
    """Discover all distinct per-model column suffixes from the DataFrame."""
    suffixes = []
    for col in result_df.columns:
        m = re.match(r"^category_1_(.+)$", col)
        if m:
            suffix = m.group(1)
            if suffix not in ("consensus", "agreement"):
                suffixes.append(suffix)
    return suffixes


MODEL_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
    "#9333ea", "#0891b2", "#be185d", "#65a30d",
]


def create_distribution_chart(result_df, categories, classify_mode="Single Model", models_list=None):
    """Create a horizontal bar chart showing category distribution."""
    total_rows = len(result_df)
    if total_rows == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    if classify_mode == "Single Model":
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

        bars = ax.barh(categories_list, percentages, color="#2563eb")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage (%)", fontsize=11)
        ax.set_title("Category Distribution (%)", fontsize=14, fontweight="bold")
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=10)

    elif classify_mode == "Ensemble":
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

        bars = ax.barh(categories_list, percentages, color="#16a34a")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage (%)", fontsize=11)
        ax.set_title("Ensemble Consensus Distribution (%)", fontsize=14, fontweight="bold")
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=10)

    else:  # Model Comparison
        if not models_list:
            models_list = []
        model_suffixes = [_find_model_column_suffix(result_df, m) for m in models_list]
        n_models = len(model_suffixes)
        n_categories = len(categories)

        fig, ax = plt.subplots(figsize=(12, max(5, n_categories * 1.2)))
        bar_height = 0.8 / max(n_models, 1)
        y_positions = np.arange(n_categories)

        for model_idx, (model_name, suffix) in enumerate(zip(models_list, model_suffixes)):
            model_pcts = []
            for i in range(1, n_categories + 1):
                col_name = f"category_{i}_{suffix}"
                if col_name in result_df.columns:
                    pct = (int(result_df[col_name].sum()) / total_rows) * 100
                else:
                    pct = 0
                model_pcts.append(pct)

            model_pcts = model_pcts[::-1]
            offset = (model_idx - n_models / 2 + 0.5) * bar_height
            color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
            display_name = model_name.split("/")[-1].split(":")[0][:20]
            ax.barh(y_positions + offset, model_pcts, bar_height * 0.9,
                    label=display_name, color=color, alpha=0.85)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories[::-1])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage (%)", fontsize=11)
        ax.set_title("Category Distribution by Model (%)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig


def create_classification_heatmap(result_df, categories, classify_mode="Single Model", models_list=None):
    """Create a binary heatmap showing classification for each row."""
    total_rows = len(result_df)
    if total_rows == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    if classify_mode == "Ensemble":
        col_names = [f"category_{i}_consensus" for i in range(1, len(categories) + 1)]
    elif classify_mode == "Model Comparison" and models_list:
        suffix = _find_model_column_suffix(result_df, models_list[0])
        col_names = [f"category_{i}_{suffix}" for i in range(1, len(categories) + 1)]
    else:
        col_names = [f"category_{i}" for i in range(1, len(categories) + 1)]

    matrix_data = []
    for col in col_names:
        if col in result_df.columns:
            matrix_data.append(result_df[col].astype(int).values)
        else:
            matrix_data.append(np.zeros(total_rows, dtype=int))

    matrix = np.array(matrix_data).T

    fig_height = max(4, min(20, total_rows * 0.15))
    fig_width = max(8, len(categories) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = ListedColormap(["#1a1a1a", "#E8A33C"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Categories", fontsize=11)
    ax.set_ylabel(f"Responses (n={total_rows})", fontsize=11)
    ax.set_yticks([])

    title = "Classification Matrix"
    if classify_mode == "Ensemble":
        title += " (Ensemble Consensus)"
    elif classify_mode == "Model Comparison" and models_list:
        title += f' ({models_list[0].split("/")[-1].split(":")[0][:20]})'
    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#1a1a1a", edgecolor="white", label="Not Present"),
        Patch(facecolor="#E8A33C", edgecolor="white", label="Present"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    return fig
