"""Generate the poster figures for the CatLLM Stata Conference poster.

Every number plotted here traces to a source in this repository:

  * Accuracy bands (fig2)    -- ../../closed_vs_open_llm/plots_tables/
                                leaderboard.csv, question == "all" rows
                                (macro_f1 there is already the mean of the
                                3 per-question macro-F1s -- verified equal
                                to hand-averaging a19i/a19f/e1b for all 18
                                models, since each question has the same
                                category count). Tier ranges = min/max
                                macro_f1 across the models in that tier.
                                Claude Fable 5's point from
                                fable5_reference.csv, question == "all".
                                Local/laptop models excluded from fig2 --
                                shown separately in fig3 instead.
  * Local models (fig3)      -- ../../closed_vs_open_llm/plots_tables/
                                leaderboard.csv, tier == "local", question
                                == "all". Same macro-F1 metric and gold
                                standard as fig2.

fig1 and fig4 are schematics and encode no data.

Usage:  python make_figures.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BERK_BLUE = "#003262"
BERK_BLUE_2 = "#0a4d7a"
GOLD = "#fdb515"
GOLD_DK = "#c98a00"
INK = "#111111"
MUTED = "#45606a"
PANEL = "#f1f5f6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)


def _box(ax, x, y, w, h, label, sub=None, fc=BERK_BLUE, tc="white", fs=13):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0,
            facecolor=fc,
        )
    )
    ty = y + h / 2 + (0.035 if sub else 0)
    ax.text(x + w / 2, ty, label, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.055, sub, ha="center", va="center",
                color=tc, fontsize=fs - 3.5, zorder=3, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color=GOLD_DK):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=22,
            linewidth=2.6, color=color, zorder=2,
        )
    )


# ---------------------------------------------------------------- fig 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.00, 0.30, 0.203, 0.42, "Stata dataset",
         "one open-ended\nstring variable", fc=BERK_BLUE, fs=12.5)
    _box(ax, 0.243, 0.30, 0.218, 0.42, "catllm extract",
         "discover categories\nfrom the data", fc=BERK_BLUE_2, fs=12.5)
    _box(ax, 0.501, 0.30, 0.249, 0.42, "catllm classify",
         "categories( ) as\none-sentence definitions", fc=GOLD, tc=INK, fs=12.5)
    _box(ax, 0.790, 0.30, 0.210, 0.42, "K indicators",
         "prefix_cat = 0/1\nready for tab / regress", fc=BERK_BLUE, fs=12.5)

    for x0, x1 in ((0.203, 0.243), (0.461, 0.501), (0.750, 0.790)):
        _arrow(ax, x0, 0.51, x1, 0.51)

    ax.text(0.5, 0.135,
            "Categories come from extract or your own list; classify "
            "returns multi-label indicator variables either way.",
            ha="center", va="center", fontsize=12.5, color=MUTED, style="italic")
    ax.text(0.5, 0.90, "Text goes in as a variable — categories come back as variables",
            ha="center", va="center", fontsize=14, fontweight="bold", color=BERK_BLUE)
    fig.savefig(OUT / "fig1_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_accuracy():
    """Macro F1 against human consensus, survey-question-level (mean of the
    three per-question F1s, so no question dominates by category count or
    base rate). Three cloud tiers as ranges, plus Claude Fable 5 as a single
    highlighted model -- four things to track, not six. Local/laptop-scale
    models excluded (separate regime, covered by "Choosing a Route")."""
    fig, ax = plt.subplots(figsize=(9.6, 3.0))

    # Precise values (2026-09-22 pull from leaderboard.csv / fable5_reference.csv,
    # question == "all"), F1 on its native 0-1 scale. Plotted at full precision --
    # only the printed labels round (floor/ceil to hundredths for ranges, nearest
    # hundredth for the single point) -- so relative positions on the axis stay
    # honest even where rounded labels coincide (Frontier open's true ceiling,
    # 0.806, is visibly short of Fable's 0.813 despite both rounding to "0.81").
    ROWS = [
        ("Frontier closed", 0.7722, 0.8279, BERK_BLUE, False),
        ("Economy closed", 0.7795, 0.8179, BERK_BLUE_2, False),
        ("Frontier open", 0.7583, 0.8060, MUTED, False),
        ("Claude Fable 5", 0.8132, 0.8132, GOLD_DK, True),
    ]
    ypos = [3, 2, 1, 0]

    def floor2(v):
        return math.floor(v * 100) / 100

    def ceil2(v):
        return math.ceil(v * 100) / 100

    for y, (label, lo, hi, color, is_point) in zip(ypos, ROWS):
        if is_point:
            ax.scatter([lo], [y], s=200, color=color, zorder=3,
                       marker="D", edgecolor="white", linewidth=1.2)
            ax.text(lo + 0.01, y, f"{round(lo, 2):.2f}", va="center", fontsize=13.5,
                    fontweight="bold", color=color)
        else:
            ax.plot([lo, hi], [y, y], color=color, linewidth=11,
                    solid_capstyle="round", zorder=2)
            ax.scatter([lo, hi], [y, y], s=130, color=color, zorder=3)
            ax.text(hi + 0.01, y, f"{floor2(lo):.2f}–{ceil2(hi):.2f}",
                    va="center", fontsize=13.5, fontweight="bold", color=color)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=12.5)
    ax.set_ylim(-0.8, 3.8)
    ax.set_xlim(0.72, 0.89)
    ax.set_xticks([0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88])
    ax.xaxis.set_major_formatter(lambda v, pos: f"{v:.2f}")
    ax.set_xlabel("Macro F1 vs. human consensus", fontsize=13)

    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dfe7ea", linewidth=1)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig2_accuracy.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_local_models():
    """Macro F1 against human consensus for the four laptop-scale local
    models (Ollama), same metric and gold standard as fig2's cloud tiers.
    Individual bars, not ranges -- there are only four models, each worth
    naming, and the spread between them is the point."""
    fig, ax = plt.subplots(figsize=(9.6, 3.0))

    # Precise values (2026-09-22 pull from leaderboard.csv, tier == "local",
    # question == "all"), sorted ascending so the bars read as a ladder.
    MODELS = [
        ("Mistral 7B", 0.6164),
        ("Llama 3.1 8B", 0.6365),
        ("Gemma 3 12B", 0.7432),
        ("Qwen3 14B", 0.7680),
    ]
    ypos = list(range(len(MODELS)))[::-1]

    for y, (name, val) in zip(ypos, MODELS):
        ax.barh(y, val, color=MUTED, height=0.55, zorder=2)
        ax.text(val + 0.012, y, f"{val:.2f}", va="center", fontsize=13.5,
                fontweight="bold", color=MUTED)

    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in MODELS], fontsize=12.5)
    ax.set_xlim(0, 0.88)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(lambda v, pos: f"{v:.1f}")
    ax.set_xlabel("Macro F1 vs. human consensus", fontsize=13)

    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dfe7ea", linewidth=1)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig3_local.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_ensemble():
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.0, 0.40, 0.155, 0.30, "response",
         "one row of text", fc=BERK_BLUE, fs=12.5)

    models = [
        ("gpt-4o-mini", "openai", 0.715),
        ("claude-haiku", "anthropic", 0.435),
        ("qwen2.5:14b", "ollama (local)", 0.155),
    ]
    for name, prov, y in models:
        _box(ax, 0.255, y, 0.245, 0.185, name, prov, fc=BERK_BLUE_2, fs=12.5)
        _arrow(ax, 0.155, 0.55, 0.255, y + 0.0925)

    _box(ax, 0.575, 0.34, 0.185, 0.42, "consensus\nvote",
         None, fc=GOLD, tc=INK, fs=13.5)
    for _, _, y in models:
        _arrow(ax, 0.500, y + 0.0925, 0.575, 0.55)

    _box(ax, 0.815, 0.40, 0.185, 0.30, "one label",
         "+ agreement score", fc=BERK_BLUE, fs=12.5)
    _arrow(ax, 0.760, 0.55, 0.815, 0.55)

    ax.text(0.5, 0.045,
            "Thresholds:  unanimous  ·  two-thirds  ·  majority  ·  any value in [0, 1]",
            ha="center", va="center", fontsize=12.5, color=MUTED, style="italic")
    ax.text(0.5, 0.965,
            "Disagreement between models is recorded, not hidden",
            ha="center", va="center", fontsize=14, fontweight="bold", color=BERK_BLUE)
    fig.savefig(OUT / "fig4_ensemble.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_accuracy()
    fig_local_models()
    fig_ensemble()
    print(f"Wrote 4 figures to {OUT}")
