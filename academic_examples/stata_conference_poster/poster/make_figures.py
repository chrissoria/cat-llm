"""Generate the poster figures for the CatLLM Stata Conference poster.

Every number plotted here traces to a source in this repository:

  * Accuracy bands (fig2)    -- an unpublished companion benchmark (18
                                models, 4 access tiers, not included in
                                this repository): survey-question-level
                                macro F1 (mean of the 3 per-question
                                macro-F1s -- verified equal to hand-
                                averaging a19i/a19f/e1b for every model,
                                since each question has the same category
                                count) against the same UCNets gold
                                standard used elsewhere on this poster.
                                Tier ranges = min/max macro_f1 across the
                                models in that tier. Claude Fable 5 scored
                                the same way, same gold standard. Local/
                                laptop models excluded from fig2 -- shown
                                separately in fig3 instead.
  * Local models (fig3)      -- same companion benchmark, local tier only.
                                Same macro-F1 metric and gold standard as
                                fig2.
  * Unanimous ensembles      -- same companion benchmark's row-level
    (fig4)                     per-model predictions (not aggregate
                                stats -- unanimous-AND requires knowing
                                where models actually overlap), scored
                                against the same UCNets gold standard,
                                unanimous-AND combined per response x
                                category, then macro F1 per question and
                                averaged across the 3 questions. See
                                unanimous_ensemble.R (this directory) for
                                the exact computation -- run
                                `Rscript unanimous_ensemble.R` to
                                reproduce (needs the source data, not
                                included here).

fig1 is a schematic and encodes no data.

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
def fig_unanimous_ensembles():
    """Unanimous-vote ensemble macro F1 (survey-question-level, mean of the
    3 per-question F1s), frontier-open tier (6 models, AND-combined) and
    local tier (4 models, AND-combined), against Claude Fable 5 as a single
    model. Computed from row-level per-model predictions and the same gold
    standard as fig2/fig3 (unanimous_ensemble.R, this directory):
    frontier-open unanimous = 0.8044 (per-q 0.8786/0.9238/0.6107), local
    unanimous = 0.6693 (per-q 0.7647/0.7728/0.4702), all 175/175 gold rows
    matched exactly for every question/tier, no fuzzy fallback needed."""
    fig, ax = plt.subplots(figsize=(9.2, 4.0))

    BARS = [
        ("Frontier open\n(6-model unanimous)", 0.8044, MUTED),
        ("Local\n(4-model unanimous)", 0.6693, GOLD),
        ("Claude Fable 5\n(single model)", 0.8132, GOLD_DK),
    ]
    xpos = [0, 1, 2]

    for x, (label, val, color) in zip(xpos, BARS):
        ax.bar(x, val, color=color, width=0.5, zorder=2)
        ax.text(x, val + 0.02, f"{val:.2f}", ha="center", fontsize=16,
                fontweight="bold", color=color)

    ax.set_xticks(xpos)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=12.5)
    ax.set_ylim(0, 0.94)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.1f}")
    ax.set_ylabel("Macro F1 vs. human consensus", fontsize=13)

    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#dfe7ea", linewidth=1)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig4_unanimous.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_accuracy()
    fig_local_models()
    fig_unanimous_ensembles()
    print(f"Wrote 4 figures to {OUT}")
