"""Derived presentation: same-state Fourier intervention and frozen E012 counts."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from spectral_diffusion_playground.table_analysis import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    rows = read_csv(root / "results/experiment_12/experiment_12_condition_summary.csv")
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(14, 8), facecolor="#f7f9fc")
    ax = fig.add_axes([0.035, 0.53, 0.93, 0.38])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, label, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012",
                facecolor=color,
                edgecolor="#ccd4df",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="#182737",
        )

    def arrow(a, b):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={"arrowstyle": "->", "color": "#53667d", "lw": 1.8},
        )

    box(0.01, 0.36, 0.13, 0.25, "Same noisy\nstate x, sigma", "#e3eaf3")
    box(0.21, 0.60, 0.18, 0.26, "Recipient G\nprediction D_G", "#dce9ff")
    box(0.21, 0.09, 0.18, 0.26, "Donor M\nprediction D_M", "#ffecd7")
    box(
        0.48,
        0.36,
        0.24,
        0.25,
        "Fourier mask S\n(1-S) FFT(D_G)\n+ S FFT(D_M)",
        "#e3f1e6",
    )
    box(
        0.81,
        0.36,
        0.17,
        0.25,
        "Inverse FFT\nHybrid prediction\n→ sampler update",
        "#e3f1e6",
    )
    arrow((0.14, 0.50), (0.21, 0.72))
    arrow((0.14, 0.46), (0.21, 0.22))
    arrow((0.39, 0.72), (0.48, 0.55))
    arrow((0.39, 0.22), (0.48, 0.40))
    arrow((0.72, 0.48), (0.81, 0.48))
    ax.text(
        0.5,
        0.98,
        "E011 / E012: replace selected Fourier coefficients of predictions at the same state",
        ha="center",
        weight="bold",
        fontsize=14,
    )
    ax.text(
        0.5,
        -0.02,
        "Calls 6–13 of the frozen 18-call Euler sampler; coefficients are not swapped in completed images.",
        ha="center",
        fontsize=11,
    )
    chart = fig.add_axes([0.075, 0.17, 0.87, 0.29], facecolor="white")
    labels = [
        "G",
        "B1",
        "B2",
        "B3",
        "B1+B2",
        "B1+B3",
        "B2+B3",
        "B1+B2+B3",
        "B4 control",
    ]
    counts = [int(row["memorized_count"]) for row in rows]
    bars = chart.bar(
        range(len(rows)),
        [v / 10 for v in counts],
        color=["#9cacc0"] + ["#4979b5"] * 7 + ["#9cacc0"],
        width=0.65,
    )
    chart.set_xticks(range(len(rows)), labels, fontsize=10)
    chart.set_ylim(0, 90)
    chart.set_ylabel("Flagged samples (%)")
    chart.spines[["top", "right"]].set_visible(False)
    chart.set_title(
        "E012 · G receives M coefficients · 1,000 generation seeds per condition",
        fontsize=13,
        loc="left",
        pad=12,
    )
    for bar, count in zip(bars, counts):
        chart.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{count}/1000",
            ha="center",
            fontsize=10,
        )
    fig.text(
        0.075,
        0.085,
        "Metric: d1 < d2/3 against the same frozen 1,000-image training reference bank.",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.055,
        "B1: r ≤ 6     B2: 6 < r ≤ 9     B3: 9 < r ≤ 11     B4: 11 < r ≤ 13",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.025,
        "One fixed checkpoint pair. Reverse-direction suppression in E011 coincides with worse descriptive FID-1K.",
        fontsize=10,
        color="#6c3d2c",
    )
    fig.savefig(args.output_dir / "fourier_intervention.png", dpi=180)
    fig.savefig(args.output_dir / "fourier_intervention.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
