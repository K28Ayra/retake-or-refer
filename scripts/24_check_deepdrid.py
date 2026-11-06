"""Why is DeepDRiD disease AUROC so much higher than EyeQ? Test the possible reasons.

    python scripts/24_check_deepdrid.py disease_mobilenetv3_small_100_320_s0
"""
import json
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NEAR_MATCH = 0.98
SEED = 0


def grade_shares(table):
    counts = table["DR_grade"].astype(int).value_counts(normalize=True)
    return {str(grade): round(float(counts.get(grade, 0)), 3) for grade in range(5)}


def auroc_by_grade(labels, scores, grades):
    """Each referable grade against all non-referable photos, so the case mix cannot help."""
    healthy = labels == 0
    results = {}
    for grade in [2, 3, 4]:
        chosen = healthy | (grades == grade)
        results[str(grade)] = round(float(roc_auc_score(labels[chosen], scores[chosen])), 4)
    return results


def auroc_at_other_mix(labels, scores, grades, target_shares):
    """AUROC after reweighting photos so their grades match another dataset's shares."""
    weights = np.zeros(len(labels))
    for grade in range(5):
        inside = grades == grade
        if inside.sum() > 0:
            weights[inside] = target_shares[str(grade)] / inside.sum()
    return round(float(roc_auc_score(labels, scores, sample_weight=weights)), 4)


def small_gray(paths):
    rows = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = (small - small.mean()) / (small.std() + 1e-6)
        rows.append(small.ravel())
    return np.stack(rows)


def nearest_train(val_rows, train_rows):
    similarity = val_rows @ train_rows.T / val_rows.shape[1]
    return similarity.max(axis=1), similarity.argmax(axis=1)


def main():
    run = sys.argv[1]
    folder = f"outputs/phase2/{run}"

    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    train = deepdrid[deepdrid["split"] == "train"].reset_index(drop=True)
    val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)
    deepdrid_scores = np.load(f"{folder}/deepdrid_val_logits.npy")[:, 0]
    deepdrid_labels = val["referable"].to_numpy()
    deepdrid_grades = val["DR_grade"].astype(int).to_numpy()

    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    eyeq_val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    readable = (eyeq_val["quality"] < 2).to_numpy()
    eyeq_read = eyeq_val[readable]
    eyeq_scores = np.load(f"{folder}/eyeq_val_logits.npy")[readable, 0]
    eyeq_labels = eyeq_read["referable"].to_numpy()
    eyeq_grades = eyeq_read["DR_grade"].astype(int).to_numpy()

    deepdrid_mix = grade_shares(val)
    eyeq_mix = grade_shares(eyeq_read)

    val_rows = small_gray(val["cache_path"])
    train_rows = small_gray(train["cache_path"])
    similarity, nearest = nearest_train(val_rows, train_rows)
    far = similarity <= NEAR_MATCH
    train_labels = train["referable"].to_numpy()

    results = {
        "run": run,
        "grade_shares": {"deepdrid_val": deepdrid_mix, "eyeq_val_readable": eyeq_mix},
        "referable_share": {
            "deepdrid_val": round(float(deepdrid_labels.mean()), 3),
            "eyeq_val_readable": round(float(eyeq_labels.mean()), 3),
        },
        "auroc": {
            "deepdrid": round(float(roc_auc_score(deepdrid_labels, deepdrid_scores)), 4),
            "eyeq": round(float(roc_auc_score(eyeq_labels, eyeq_scores)), 4),
        },
        "auroc_each_grade_against_healthy": {
            "deepdrid": auroc_by_grade(deepdrid_labels, deepdrid_scores, deepdrid_grades),
            "eyeq": auroc_by_grade(eyeq_labels, eyeq_scores, eyeq_grades),
        },
        "auroc_with_grades_swapped": {
            "eyeq_at_deepdrid_mix": auroc_at_other_mix(
                eyeq_labels, eyeq_scores, eyeq_grades, deepdrid_mix
            ),
            "deepdrid_at_eyeq_mix": auroc_at_other_mix(
                deepdrid_labels, deepdrid_scores, deepdrid_grades, eyeq_mix
            ),
        },
        "near_match_in_train": {
            "similarity_above": NEAR_MATCH,
            "photos": int((~far).sum()),
            "share_same_label": round(
                float((train_labels[nearest[~far]] == deepdrid_labels[~far]).mean()), 3
            ),
            "auroc_without_them": round(
                float(roc_auc_score(deepdrid_labels[far], deepdrid_scores[far])), 4
            ),
            "photos_left": int(far.sum()),
        },
    }

    with open(f"results/phase2_deepdrid_check_{run}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
