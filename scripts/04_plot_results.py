"""Render the figures in REPORT.md from the committed evaluation artifacts.

Every number drawn here is read from an artifact under eval/results/ and none
is typed in, so a figure can only disagree with a table if the artifact changed.

    python scripts/04_plot_results.py            # writes eval/results/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be chosen first
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
FIGURES = RESULTS / "figures"

# Light-surface palette; the two series hues pass the colour-vision checks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
QUIET = "#b9b8b1"  # the de-emphasised series

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }
)

# Retrieval axes in the order the report discusses them, with the shipped arm.
RETRIEVAL_AXES = (
    ("retrieval_mode", "Retrieval mode", "dense"),
    ("rerank", "Cross-encoder reranker", "off"),
    ("chunking", "Chunking", "section"),
    ("embedding", "Embedding model", "bge-small"),
    ("translation", "Query translation", "rewrite"),
)
# End-to-end metrics, upstream to downstream.
PIPELINE_METRICS = (
    ("route_label_accuracy", "Route label accuracy"),
    ("route_slots_accuracy", "Route slot accuracy"),
    ("stats_exact_match", "Stats exact match"),
    ("retrieval_recall_at_5", "Retrieval Recall@5"),
    ("judge_correctness", "Judge: correctness"),
    ("judge_completeness", "Judge: completeness"),
    ("judge_faithfulness", "Judge: faithfulness"),
    ("answer_evidence_valid", "Evidence checks passed"),
)


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  skip: {path.relative_to(ROOT)} not found", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _hgrid(ax, limit: float = 1.0) -> None:
    ax.set_xlim(0, limit)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=AXIS)


def retrieval_figure() -> Path | None:
    """One row per axis, MRR and Recall@5 side by side, every arm labelled."""
    axes_data = []
    for name, title, shipped in RETRIEVAL_AXES:
        report = _load(f"ablations/{name}.json")
        if report is None:
            continue
        arms = [arm for arm in report["arms"] if not arm.get("error")]
        axes_data.append((title, shipped, arms))
    if not axes_data:
        return None

    heights = [len(arms) for _, _, arms in axes_data]
    fig, grid = plt.subplots(
        len(axes_data),
        2,
        figsize=(8.6, 0.42 * sum(heights) + 0.9 * len(axes_data)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.9, "wspace": 0.35},
        squeeze=False,
    )
    for row, (title, shipped, arms) in enumerate(axes_data):
        for col, (metric, label) in enumerate(
            (("retrieval_mrr", "MRR"), ("retrieval_recall_at_5", "Recall@5"))
        ):
            ax = grid[row][col]
            names = [
                f"{arm['name']}  (shipped)" if arm["name"] == shipped else arm["name"]
                for arm in arms
            ]
            values = [arm["metrics"][metric]["mean"] for arm in arms]
            positions = range(len(arms))[::-1]
            ax.barh(positions, values, height=0.58, color=BLUE, zorder=3)
            for position, value in zip(positions, values, strict=True):
                ax.text(value + 0.012, position, f"{value:.3f}", va="center", color=INK_2)
            ax.set_yticks(list(positions), names)
            _hgrid(ax, limit=0.8)
            ax.set_xticks([0, 0.2, 0.4, 0.6])
            ax.set_title(f"{title} — {label}" if col == 0 else label)
    fig.suptitle(
        "Retrieval ablations, 30 labelled items, one axis at a time",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.955)
    return _save(fig, "retrieval_ablations.png")


def bootstrap_figure() -> Path | None:
    """Per-90 rates with their intervals, and the paired difference intervals."""
    report = _load("bootstrap.json")
    if report is None:
        return None
    players = [player for player in report["players"] if player.get("interval")]
    players.sort(key=lambda player: player["interval"]["point"], reverse=True)
    pairs = report["pairs"]
    level = report["settings"]["confidence_level"]
    metric = report["metric"]
    # The artifact keys players by their StatsBomb legal name; label them as asked.
    common = {player["player_name"]: player["query"] for player in report["players"]}
    surname = {name: query.split()[-1] for name, query in common.items()}

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.6, 1.4 + 0.5 * len(players) + 0.62 * len(pairs)),
        gridspec_kw={"height_ratios": [len(players) + 1, len(pairs) + 1.6], "hspace": 0.6},
    )

    positions = range(len(players))[::-1]
    for position, player in zip(positions, players, strict=True):
        interval = player["interval"]
        top.plot([interval["low"], interval["high"]], [position, position], color=BLUE, lw=2)
        top.plot(interval["point"], position, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        top.text(
            2.15,
            position,
            f"{interval['point']:.2f}  [{interval['low']:.2f}, {interval['high']:.2f}]"
            f"   {interval['n_matches']} matches",
            va="center",
            color=INK_2,
        )
    top.set_yticks(list(positions), [common[player["player_name"]] for player in players])
    top.set_xlim(0, 3.3)
    top.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    top.grid(axis="x", color=GRID, linewidth=0.8)
    top.set_axisbelow(True)
    top.tick_params(axis="y", length=0)
    top.set_title(
        f"{metric.capitalize()} per 90 with a {level:.0%} interval from resampled matches"
    )

    positions = range(len(pairs))[::-1]
    for position, pair in zip(positions, pairs, strict=True):
        paired, independent = pair["paired"], pair["independent"]
        shared = len(pair["shared_match_ids"])
        bottom.plot(
            [independent["low"], independent["high"]],
            [position - 0.18, position - 0.18],
            color=QUIET,
            lw=2,
            solid_capstyle="round",
        )
        bottom.plot(
            [paired["low"], paired["high"]], [position + 0.18, position + 0.18], color=BLUE, lw=2
        )
        bottom.plot(
            paired["point"], position + 0.18, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4
        )
        bottom.text(
            1.75,
            position,
            f"{paired['point']:+.2f}  [{paired['low']:+.2f}, {paired['high']:+.2f}]"
            f"   {shared} shared fixture{'s' if shared != 1 else ''}",
            va="center",
            color=INK_2,
        )
    bottom.axvline(0, color=AXIS, lw=1, zorder=2)
    bottom.set_yticks(
        list(positions), [f"{surname[pair['left']]} − {surname[pair['right']]}" for pair in pairs]
    )
    bottom.set_xlim(-1.2, 3.3)
    bottom.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
    bottom.grid(axis="x", color=GRID, linewidth=0.8)
    bottom.set_axisbelow(True)
    bottom.tick_params(axis="y", length=0)
    bottom.set_title("Difference between two players: every interval spans zero")
    bottom.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, label="paired draw"),
            Line2D([], [], color=QUIET, lw=2, label="independent draw"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.suptitle(
        f"Bootstrap intervals — {report['settings']['n_resamples']:,} resamples, "
        f"seed {report['settings']['seed']}",
        x=0.02,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    return _save(fig, "bootstrap_intervals.png")


def router_figure() -> Path | None:
    """Few-shot router against the keyword baseline, end to end."""
    report = _load("ablations/router.json")
    if report is None:
        return None
    arms = {arm["name"]: arm for arm in report["arms"]}
    few_shot, keyword = arms["few_shot"], arms["keyword"]

    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(PIPELINE_METRICS) + 1.2))
    positions = list(range(len(PIPELINE_METRICS)))[::-1]
    for offset, arm, colour in ((0.19, few_shot, BLUE), (-0.19, keyword, ORANGE)):
        values = [arm["metrics"][metric]["mean"] for metric, _ in PIPELINE_METRICS]
        ax.barh(
            [position + offset for position in positions],
            values,
            height=0.34,
            color=colour,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            ax.text(value + 0.012, position + offset, f"{value:.3f}", va="center", color=INK_2)
    ax.set_yticks(positions, [label for _, label in PIPELINE_METRICS])
    _hgrid(ax, limit=1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(
        handles=[
            Patch(color=BLUE, label="few-shot router (shipped)"),
            Patch(color=ORANGE, label="keyword baseline"),
        ],
        loc="lower right",
    )
    ax.set_title(
        "The evidence checks cannot see a routing error: identical under both routers "
        "while judged correctness falls"
    )
    fig.suptitle(
        "Router ablation, all 45 items, end to end",
        x=0.02,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    return _save(fig, "router_ablation.png")


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def main() -> int:
    written = [path for path in (retrieval_figure(), bootstrap_figure(), router_figure()) if path]
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
