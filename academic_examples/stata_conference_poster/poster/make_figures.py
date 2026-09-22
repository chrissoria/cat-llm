"""Generate the poster figures for the CatLLM Stata Conference poster.

Every number plotted here traces to a source in this repository:

  * Accuracy bands (fig2)    -- academic_examples/README.md (UCNETS validation:
                                GPT-4o, Claude Sonnet 3.7, Llama 3.1 Sonar Large,
                                Mistral Large vs. human annotators) and README.md
                                (98% vs. human consensus with GPT-5 / Gemini /
                                Qwen 3).
  * Cost + runtime (fig3)    -- academic_examples/paper.md (8 models x 3,208
                                responses = 25,664 classifications).

fig1 and fig4 are schematics and encode no data.

Usage:  python make_figures.py
"""

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
    """Accuracy bands vs. human coders. Ranges, not point estimates, because
    the underlying validation reports per-family ranges across four models."""
    fig, ax = plt.subplots(figsize=(9.6, 4.4))

    rows = [
        ("Straightforward items", 97, 97, BERK_BLUE),
        ("Complex interpretive items", 88, 91, BERK_BLUE),
        ("Straightforward items", 95, 96, GOLD_DK),
        ("Complex interpretive items", 87, 87, GOLD_DK),
    ]
    ypos = [3, 2, 1, 0]

    for y, (task, lo, hi, color) in zip(ypos, rows):
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=color, linewidth=11,
                    solid_capstyle="round", zorder=2)
            label = f"{lo}–{hi}%"
        else:
            label = f"{lo}%"
        ax.scatter([lo, hi], [y, y], s=130, color=color, zorder=3)
        ax.text(hi + 0.7, y, label, va="center", fontsize=13.5,
                fontweight="bold", color=color)

    ax.axvline(98, color="#b3261e", linestyle="--", linewidth=2, zorder=1)
    ax.text(98.3, 4.05, "98%  ensemble vs.\nhuman consensus\n(GPT-5 · Gemini · Qwen 3)",
            fontsize=11, color="#b3261e", va="top", fontweight="bold",
            linespacing=1.3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(-0.7, 4.3)
    ax.set_xlim(85, 103.5)
    ax.set_xticks([86, 88, 90, 92, 94, 96, 98, 100])
    ax.set_xlabel("Agreement with human coders (%)", fontsize=13)

    ax.axhspan(-0.7, 1.55, color=PANEL, zorder=0)
    ax.text(85.35, 3.95, "PROPRIETARY  (GPT-4o, Claude Sonnet 3.7)", fontsize=12,
            fontweight="bold", color=BERK_BLUE, va="center")
    ax.text(85.35, 1.30, "OPEN-WEIGHT  (Llama 3.1, Mistral Large)", fontsize=12,
            fontweight="bold", color=GOLD_DK, va="center")

    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dfe7ea", linewidth=1)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig2_accuracy.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_providers():
    """Cost and wall-clock spread across the 8 models benchmarked on the same
    3,208 responses. Only the endpoints are published, so only the endpoints
    are drawn."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # -- cost
    ax1.plot([0.38, 27.85], [0, 0], color=BERK_BLUE, linewidth=12,
             solid_capstyle="round", zorder=2)
    ax1.scatter([0.38, 27.85], [0, 0], s=170, color=GOLD, zorder=3,
                edgecolor=BERK_BLUE, linewidth=2)
    ax1.text(0.38, 0.30, "$0.38\nMistral Medium", ha="left", va="bottom",
             fontsize=12, fontweight="bold", color=BERK_BLUE)
    ax1.text(27.85, -0.32, "$27.85\nGPT-5", ha="right", va="top",
             fontsize=12, fontweight="bold", color=BERK_BLUE)
    ax1.set_xscale("log")
    ax1.set_xlim(0.2, 60)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_yticks([])
    ax1.set_xticks([0.5, 1, 5, 10, 30])
    ax1.set_xticklabels(["$0.50", "$1", "$5", "$10", "$30"], fontsize=12)
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("Cost to classify 3,208 responses (log scale)", fontsize=12)
    ax1.set_title("73× spread in price", fontsize=13.5, fontweight="bold",
                  color=BERK_BLUE, pad=10)

    # -- runtime
    ax2.plot([23, 420], [0, 0], color=GOLD_DK, linewidth=12,
             solid_capstyle="round", zorder=2)
    ax2.scatter([23, 420], [0, 0], s=170, color=GOLD, zorder=3,
                edgecolor=GOLD_DK, linewidth=2)
    ax2.text(23, 0.30, "23 min", ha="left", va="bottom", fontsize=12,
             fontweight="bold", color=GOLD_DK)
    ax2.text(420, -0.32, "7+ hr", ha="right", va="top", fontsize=12,
             fontweight="bold", color=GOLD_DK)
    ax2.set_xscale("log")
    ax2.set_xlim(12, 900)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_yticks([])
    ax2.set_xticks([15, 30, 60, 120, 240, 480])
    ax2.set_xticklabels(["15m", "30m", "1h", "2h", "4h", "8h"], fontsize=12)
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("Wall-clock time, same job (log scale)", fontsize=12)
    ax2.set_title("Rate limits, not model size, drive runtime", fontsize=13.5,
                  fontweight="bold", color=BERK_BLUE, pad=10)

    for ax in (ax1, ax2):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#dfe7ea", linewidth=1)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_providers.png")
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
    fig_providers()
    fig_ensemble()
    print(f"Wrote 4 figures to {OUT}")
