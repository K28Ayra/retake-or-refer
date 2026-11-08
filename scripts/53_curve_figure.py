"""The accuracy against file size figure. Each line keeps only the settings no other setting beats."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LINES = [
    ("plain", "jpeg", "plain model, JPEG", "#2a78d6"),
    ("plain", "jpeg444_retina", "plain model, retina crop, full-colour JPEG", "#eb6834"),
    ("tunelight", "jpeg", "fine-tuned with light compression, JPEG", "#1baf7a"),
    ("compress", "jpeg", "trained with strong compression, JPEG", "#eda100"),
]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def frontier(rows):
    """Smallest file first, keeping a setting only if it beats every smaller one."""
    rows = rows.sort_values("kb")
    kept = []
    best = -1.0
    for _, row in rows.iterrows():
        if row["auroc"] > best:
            kept.append(row)
            best = row["auroc"]
    return pd.DataFrame(kept)


def style(axis):
    axis.set_xscale("log")
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xticks([2, 5, 10, 20, 50, 100])
    axis.set_xticklabels(["2", "5", "10", "20", "50", "100"])


def main():
    means = pd.read_csv("results/phase5_curve_means.csv")
    means = means[means["target"] == 0.85]
    clean = pd.read_csv("results/phase3_runs.csv")
    plain_runs = clean[clean["run"].str.match(r"joint_384_s\d_nodeepdridquality_grade$")]
    plain_auroc = plain_runs["auroc_good_usable"].mean()

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for model, encoder, label, colour in LINES:
        rows = means[(means["model"] == model) & (means["encoder"] == encoder)]
        best = frontier(rows)
        left.plot(best["kb"], best["auroc"], color=colour, linewidth=2, marker="o", markersize=4,
                  label=label)
        merged = rows.merge(best[["side", "file_quality"]], on=["side", "file_quality"])
        merged = merged.sort_values("kb")
        right.plot(merged["kb"], merged["grade_change"] * 100, color=colour, linewidth=2, marker="o",
                   markersize=4, label=label)

    left.axhline(plain_auroc, color=MUTED, linewidth=1, linestyle="--")
    left.text(1.6, plain_auroc + 0.004, "plain model on original files", color=MUTED, fontsize=8)
    left.axhline(plain_auroc - 0.01, color=MUTED, linewidth=1, linestyle=":")
    left.text(1.6, plain_auroc - 0.022, "0.01 below", color=MUTED, fontsize=8)
    left.set_ylim(0.6, 0.93)
    left.set_ylabel("Referable AUROC, EyeQ Good + Usable", color=INK, fontsize=9)
    left.set_xlabel("Mean file size, KB (log scale)", color=INK, fontsize=9)
    left.set_title("Disease", color=INK, fontsize=10, loc="left")

    right.axhline(5, color=MUTED, linewidth=1, linestyle=":")
    right.text(1.6, 5.6, "5% limit", color=MUTED, fontsize=8)
    right.set_ylim(0, 40)
    right.set_ylabel("Photos whose quality grade changed, %", color=INK, fontsize=9)
    right.set_xlabel("Mean file size, KB (log scale)", color=INK, fontsize=9)
    right.set_title("Quality grade, same settings as the left", color=INK, fontsize=10, loc="left")

    for axis in [left, right]:
        style(axis)
    left.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    figure.text(0.01, 0.01, "Means over 3 seeds, 1,784 EyeQ validation photos. Each line keeps a setting only "
                "if no smaller setting of that line scores higher.", color=MUTED, fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig("docs/figures/phase5_curve.png", dpi=150)


if __name__ == "__main__":
    main()
