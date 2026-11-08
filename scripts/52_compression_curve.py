"""Turn the sweeps into one table per setting, and find the smallest file that passes the rule.

    python scripts/52_compression_curve.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from retake.compress import sensitivity, specificity, threshold_for_sensitivity
from retake.metrics import macro_f1, paired_gap

SIDES = [1024, 768, 512, 384, 256]
QUALITIES = [95, 85, 75, 60, 50, 40, 30, 20, 10]
ENCODERS = ["jpeg", "webp", "jpeg_retina", "webp_retina", "jpeg444", "jpeg444_retina"]
TARGETS = [0.80, 0.85, 0.90]
MAIN_TARGET = 0.85
MAX_DROP = 0.02
MAX_GRADE_CHANGE = 0.05
REJECT = 2
SEEDS = [0, 1, 2]
MODELS = [
    "plain",
    "tunelight",
    "teachermild",
    "teacherstrong",
    "light",
    "mild",
    "mildconsistency",
    "compress",
    "consistency",
]
AUROC_MARGIN = 0.01
AVERAGED = [
    "sens_drop",
    "spec_drop",
    "auroc",
    "spec_refit",
    "quality_f1",
    "grade_change",
    "new_reject",
    "passes",
]


def load_val():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    return eyeq[eyeq["split"] == "val"].reset_index(drop=True)


def score_setting(sweep, label, val, threshold, target):
    readable = val["quality"].to_numpy() < 2
    referable = val["referable"].to_numpy()[readable]
    true_grade = val["quality"].to_numpy()

    start_scores = sweep["original__referable"][readable]
    start_grade = sweep["original__quality"]
    scores = sweep[f"{label}__referable"][readable]
    grade = sweep[f"{label}__quality"]

    start_sens = sensitivity(referable, start_scores, threshold)
    start_spec = specificity(referable, start_scores, threshold)
    sens = sensitivity(referable, scores, threshold)
    spec = specificity(referable, scores, threshold)

    # only for comparing models. The rule itself never moves the threshold
    refit = threshold_for_sensitivity(referable, scores, target)
    newly_rejected = (grade == REJECT) & (start_grade != REJECT)
    return {
        "kb": sweep[f"{label}__bytes"].mean() / 1024,
        "sensitivity": sens,
        "specificity": spec,
        "sens_drop": start_sens - sens,
        "spec_drop": start_spec - spec,
        "auroc": roc_auc_score(referable, scores),
        "spec_refit": specificity(referable, scores, refit),
        "quality_f1": macro_f1(true_grade, grade),
        "grade_change": float((grade != start_grade).mean()),
        "new_reject": float(newly_rejected.mean()),
    }


def add_pass_column(table):
    sens_ok = table["sens_drop"] < MAX_DROP
    spec_ok = table["spec_drop"] < MAX_DROP
    grade_ok = table["grade_change"] <= MAX_GRADE_CHANGE
    table["passes"] = sens_ok & spec_ok & grade_ok
    return table


def sweep_table(sweep, val, target):
    readable = val["quality"].to_numpy() < 2
    referable = val["referable"].to_numpy()[readable]
    threshold = threshold_for_sensitivity(referable, sweep["original__referable"][readable], target)

    rows = []
    for encoder in ENCODERS:
        if f"{encoder}_{SIDES[0]}_{QUALITIES[0]}__bytes" not in sweep:
            continue
        for side in SIDES:
            for quality in QUALITIES:
                label = f"{encoder}_{side}_{quality}"
                row = {"encoder": encoder, "side": side, "file_quality": quality}
                row["threshold"] = threshold
                row["target"] = target
                row.update(score_setting(sweep, label, val, threshold, target))
                rows.append(row)
    return add_pass_column(pd.DataFrame(rows))


def seed_means(table):
    """One row per setting, averaged over seeds. The rule is then applied to the means."""
    keys = ["model", "target", "encoder", "side", "file_quality"]
    means = table.groupby(keys, sort=False)[["kb"] + AVERAGED].mean().reset_index()
    means = means.rename(columns={"passes": "seeds_passing_share"})
    return add_pass_column(means)


def smallest(table, encoder, auroc_floor=None):
    inside = table[(table["encoder"] == encoder) & table["passes"]]
    if auroc_floor is not None:
        inside = inside[inside["auroc"] >= auroc_floor]
    if len(inside) == 0:
        return None
    return inside.sort_values("kb").iloc[0]


def describe(row):
    return f"{row['kb']:.1f} KB, {int(row['side'])} px q{int(row['file_quality'])}"


def drop_intervals(sweep, val, label, threshold):
    readable = val["quality"].to_numpy() < 2
    referable = val["referable"].to_numpy()[readable]
    patients = val["patient"].to_numpy()[readable]
    start = sweep["original__referable"][readable]
    scores = sweep[f"{label}__referable"][readable]

    def sens(labels, values):
        return sensitivity(labels, values, threshold)

    def spec(labels, values):
        return specificity(labels, values, threshold)

    return {
        "sens_drop_ci": paired_gap(sens, referable, start, scores, patients),
        "spec_drop_ci": paired_gap(spec, referable, start, scores, patients),
    }


def sweep_file(model, seed):
    return Path(f"outputs/phase5/sweeps/{model}_s{seed}.npz")


def clean_scores(sweep, val):
    readable = val["quality"].to_numpy() < 2
    return {
        "auroc": roc_auc_score(val["referable"][readable], sweep["original__referable"][readable]),
        "macro_f1": macro_f1(val["quality"], sweep["original__quality"]),
    }


def main():
    val = load_val()
    tables = []
    sweeps = {}
    clean = []
    for model in MODELS:
        for seed in SEEDS:
            if not sweep_file(model, seed).exists():
                continue
            sweep = dict(np.load(sweep_file(model, seed)))
            sweeps[(model, seed)] = sweep
            scores = clean_scores(sweep, val)
            clean.append({"model": model, "seed": seed, **scores})
            for target in TARGETS:
                table = sweep_table(sweep, val, target)
                table["model"] = model
                table["seed"] = seed
                tables.append(table)

    every = pd.concat(tables)
    every.round(4).to_csv("results/phase5_curve.csv", index=False)
    means = seed_means(every)

    clean = pd.DataFrame(clean)
    print(clean.round(4).to_string(index=False))
    print(clean.groupby("model", sort=False)[["auroc", "macro_f1"]].mean().round(4))

    plain_clean = clean[clean["model"] == "plain"]["auroc"].mean()
    auroc_floor = plain_clean - AUROC_MARGIN
    print("AUROC floor", round(auroc_floor, 4))

    summary = []
    for model in means["model"].unique():
        for target in TARGETS:
            inside = means[(means["model"] == model) & (means["target"] == target)]
            for encoder in ENCODERS:
                if encoder not in set(inside["encoder"]):
                    continue
                best = smallest(inside, encoder)
                if best is None:
                    print(model, target, encoder, "nothing passes")
                    continue
                print(model, target, encoder, describe(best),
                      f"sens {best['sens_drop']:.3f} spec {best['spec_drop']:.3f}",
                      f"grades {best['grade_change']:.3f} reject {best['new_reject']:.3f}",
                      f"auroc {best['auroc']:.3f} refit spec {best['spec_refit']:.3f}",
                      f"seeds {best['seeds_passing_share']:.2f}")
                row = {"model": model, "target": target, "encoder": encoder}
                wanted = ["side", "file_quality", "kb", "seeds_passing_share"]
                wanted += [name for name in AVERAGED if name != "passes"]
                row.update(best[wanted].to_dict())
                with_floor = smallest(inside, encoder, auroc_floor)
                if with_floor is None:
                    print("   with AUROC floor: nothing")
                    row["kb_with_floor"] = None
                else:
                    print("   with AUROC floor:", describe(with_floor),
                          f"auroc {with_floor['auroc']:.3f}",
                          f"seeds {with_floor['seeds_passing_share']:.2f}")
                    row["kb_with_floor"] = with_floor["kb"]
                    row["side_with_floor"] = with_floor["side"]
                    row["quality_with_floor"] = with_floor["file_quality"]

                # the interval is per seed: each one is a different model on the same photos
                label = f"{encoder}_{int(best['side'])}_{int(best['file_quality'])}"
                for seed in SEEDS:
                    if (model, seed) not in sweeps:
                        continue
                    picked = every[(every["model"] == model) & (every["seed"] == seed)]
                    threshold = picked[picked["target"] == target]["threshold"].iloc[0]
                    gaps = drop_intervals(sweeps[(model, seed)], val, label, threshold)
                    row[f"sens_ci_s{seed}"] = gaps["sens_drop_ci"]
                    row[f"spec_ci_s{seed}"] = gaps["spec_drop_ci"]
                    if target == MAIN_TARGET:
                        print("   seed", seed, "sens drop CI", np.round(gaps["sens_drop_ci"], 3),
                              "spec drop CI", np.round(gaps["spec_drop_ci"], 3))
                summary.append(row)

    means.round(4).to_csv("results/phase5_curve_means.csv", index=False)
    with open("results/phase5_smallest.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
