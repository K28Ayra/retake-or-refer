"""Compare CNNs with the classical baseline on the same validation photos, and by photo size."""
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from retake.metrics import accuracy, macro_f1, patient_bootstrap
from retake.quality_features import feature_names

LESIONS = ["dark_spots", "dark_area", "bright_spots", "bright_area", "dark_blob"]
TEST_COUNTS = np.array([8471, 4558, 3220], dtype=float)
ROUNDS = 1000


def classical_predictions(columns):
    table = pd.read_csv("outputs/phase1/eyeq.csv", dtype={"patient": str})
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    model = HistGradientBoostingClassifier(
        random_state=0,
        class_weight="balanced",
        max_iter=600,
        max_leaf_nodes=63,
        learning_rate=0.03,
    )
    model.fit(train[columns], train["quality"])
    return val["image"].to_numpy(), model.predict(val[columns])


def paired_gap(labels, first, second, patients, metric, seed=0):
    """95% interval for first minus second, resampling whole patients."""
    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())

    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(ROUNDS):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        gaps.append(metric(labels[rows], first[rows]) - metric(labels[rows], second[rows]))
    low, high = np.percentile(gaps, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def at_test_mix(labels, predicted):
    """Estimate what the same recalls would score with EyeQ test's class shares."""
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2]).astype(float)
    rates = matrix / matrix.sum(axis=1, keepdims=True)
    mix = TEST_COUNTS / TEST_COUNTS.sum()
    expected = rates * mix[:, None]

    f1_values = []
    for name in range(3):
        hit = expected[name, name]
        precision = hit / expected[:, name].sum()
        recall = hit / expected[name, :].sum()
        f1_values.append(2 * precision * recall / (precision + recall))
    return round(float(np.trace(expected)), 4), round(float(np.mean(f1_values)), 4)


def photo_sizes(names):
    sizes = []
    for name in names:
        image = cv2.imread(f"data/eyepacs/original/{name}")
        sizes.append(f"{image.shape[1]}x{image.shape[0]}")
    return np.array(sizes)


def quality_by_size(labels, predicted, sizes):
    results = {}
    for size, count in zip(*np.unique(sizes, return_counts=True)):
        if count < 60:
            continue
        inside = sizes == size
        results[size] = {
            "photos": int(count),
            "accuracy": round(accuracy(labels[inside], predicted[inside]), 4),
            "macro_f1": round(macro_f1(labels[inside], predicted[inside]), 4),
        }
    return results


def main():
    runs = pd.read_csv("results/phase2_runs.csv")
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    labels = val["quality"].to_numpy()
    patients = val["patient"].to_numpy()
    sizes = photo_sizes(val["image"])

    all_columns = feature_names() + LESIONS
    names, classical_best = classical_predictions(all_columns)
    assert (names == val["image"].to_numpy()).all()
    _, classical_phone = classical_predictions(feature_names())

    results = {
        "classical": {
            "quality + lesion": {
                "accuracy": round(accuracy(labels, classical_best), 4),
                "macro_f1": round(macro_f1(labels, classical_best), 4),
                "at_test_mix": at_test_mix(labels, classical_best),
                "by_photo_size": quality_by_size(labels, classical_best, sizes),
            },
            "quality only": {
                "accuracy": round(accuracy(labels, classical_phone), 4),
                "macro_f1": round(macro_f1(labels, classical_phone), 4),
            },
        },
        "quality": {},
        "disease": {},
    }

    quality_runs = runs[runs["task"] == "quality"]
    for _, run in quality_runs.iterrows():
        logits = np.load(f"outputs/phase2/{run['run']}/eyeq_val_logits.npy")
        predicted = logits.argmax(axis=1)
        results["quality"][run["run"]] = {
            "macro_f1": run["macro_f1"],
            "macro_f1_ci": patient_bootstrap(macro_f1, labels, predicted, patients),
            "minus_classical_ci": paired_gap(labels, predicted, classical_best, patients, macro_f1),
            "at_test_mix": at_test_mix(labels, predicted),
            "by_photo_size": quality_by_size(labels, predicted, sizes),
        }

    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)
    readable = labels < 2

    disease_runs = runs[runs["task"] == "disease"]
    for _, run in disease_runs.iterrows():
        folder = f"outputs/phase2/{run['run']}"
        scores = np.load(f"{folder}/eyeq_val_logits.npy")[:, 0]
        results["disease"][run["run"]] = {
            "auroc_good_usable": run["auroc_good_usable"],
            "auroc_good_usable_ci": patient_bootstrap(
                roc_auc_score,
                val["referable"].to_numpy()[readable],
                scores[readable],
                patients[readable],
            ),
            "deepdrid_auroc": run["deepdrid_auroc"],
            "deepdrid_auroc_ci": patient_bootstrap(
                roc_auc_score,
                deepdrid_val["referable"].to_numpy(),
                np.load(f"{folder}/deepdrid_val_logits.npy")[:, 0],
                deepdrid_val["patient"].to_numpy(),
            ),
        }

    with open("results/phase2_compare.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("written", len(results["quality"]), len(results["disease"]))


if __name__ == "__main__":
    main()
