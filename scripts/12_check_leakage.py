"""The first Phase 1 scores looked too good, so this looks for shortcuts behind them."""
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from retake.metrics import accuracy, macro_f1
from retake.quality_features import feature_names

LESIONS = ["dark_spots", "dark_area", "bright_spots", "bright_area", "dark_blob"]
GROUPS = {
    "cover": ["cover", "corner_light"],
    "colour": ["red", "green", "blue", "blue_to_red", "green_to_red"],
    "brightness": ["brightness", "dark_share", "bright_share"],
    "contrast": ["contrast", "green_contrast", "green_range", "entropy"],
    "sharpness": ["laplacian", "tenengrad", "focus", "high_freq", "vessel_mean", "vessel_p99"],
    "light evenness": ["light_spread", "light_range", "centre_to_edge"],
    "lesion": LESIONS,
}


def boost():
    return HistGradientBoostingClassifier(
        random_state=0, max_iter=300, learning_rate=0.05, max_leaf_nodes=15
    )


def photo_size(path):
    image = cv2.imread(path)
    return f"{image.shape[1]}x{image.shape[0]}"


def drop_groups_quality(eyeq, results):
    train = eyeq[eyeq["split"] == "train"]
    val = eyeq[eyeq["split"] == "val"]
    everything = feature_names() + LESIONS

    rows = {}
    for group, columns in GROUPS.items():
        kept = [name for name in everything if name not in columns]
        model = boost().fit(train[kept], train["quality"])
        predicted = model.predict(val[kept])
        rows[f"without {group}"] = round(macro_f1(val["quality"], predicted), 4)
        print("eyeq quality without", group, rows[f"without {group}"], flush=True)
    results["eyeq_quality_macro_f1_drop_one_group"] = rows


def reweight_to_test_mix(eyeq, results):
    """EyeQ test has far more Usable photos than train, so val flatters any model."""
    train = eyeq[eyeq["split"] == "train"]
    val = eyeq[eyeq["split"] == "val"]
    everything = feature_names() + LESIONS
    model = boost().fit(train[everything], train["quality"])
    predicted = model.predict(val[everything])

    test_counts = np.array([8471, 4558, 3220], dtype=float)
    test_mix = test_counts / test_counts.sum()
    matrix = confusion_matrix(val["quality"], predicted).astype(float)
    rates = matrix / matrix.sum(axis=1, keepdims=True)
    expected = rates * test_mix[:, None]

    f1_values = []
    for name in range(3):
        true_positive = expected[name, name]
        precision = true_positive / expected[:, name].sum()
        recall = true_positive / expected[name, :].sum()
        f1_values.append(2 * precision * recall / (precision + recall))

    val_mix = val["quality"].value_counts(normalize=True).sort_index()
    results["eyeq_val_mix"] = val_mix.round(3).to_dict()
    results["eyeq_test_mix"] = dict(enumerate(test_mix.round(3).tolist()))
    results["reweighted_to_test_mix"] = {
        "accuracy": round(float(np.trace(expected)), 4),
        "macro_f1": round(float(np.mean(f1_values)), 4),
    }
    results["per_class_recall"] = np.diag(rates).round(3).tolist()


def by_camera_eyeq(eyeq, results):
    val = eyeq[eyeq["split"] == "val"].copy()
    paths = "data/eyepacs/original/" + val["image"]
    val["size"] = [photo_size(path) for path in paths]
    table = val.groupby("size")["quality"].agg(["count", "mean"])
    table = table[table["count"] >= 30].sort_values("count", ascending=False)
    shares = pd.crosstab(val["size"], val["quality"], normalize="index").round(3)
    results["eyeq_val_quality_share_by_photo_size"] = shares.loc[table.index].to_dict("index")


def deepdrid_camera_check(results):
    table = pd.read_csv("outputs/phase1/deepdrid.csv", dtype={"patient": str})
    splits = pd.read_csv("data/splits/deepdrid.csv", dtype={"patient": str})
    table["size"] = [photo_size(path) for path in splits["image_path"]]
    table["big"] = (table["size"] == "1976x1984").astype(int)

    rates = table.groupby("size")["referable"].agg(["count", "mean"])
    results["deepdrid_referable_rate_by_photo_size"] = rates.round(3).to_dict("index")

    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    columns = feature_names() + LESIONS

    model = boost().fit(train[columns], train["referable"])
    scores = model.predict_proba(val[columns])[:, 1]
    overall = roc_auc_score(val["referable"], scores)

    # how much of this can be told from the camera alone, and what is left inside one camera
    camera_only = roc_auc_score(val["referable"], val["big"])
    inside = {}
    for size, group in val.assign(score=scores).groupby("size"):
        if len(group) >= 30 and group["referable"].nunique() == 2:
            inside[size] = round(roc_auc_score(group["referable"], group["score"]), 4)

    can_guess = HistGradientBoostingClassifier(random_state=0, max_leaf_nodes=15)
    can_guess.fit(train[columns], train["big"])
    guessed = accuracy(val["big"], can_guess.predict(val[columns]))

    results["deepdrid_referable"] = {
        "overall_auroc": round(overall, 4),
        "photo_size_alone_auroc": round(camera_only, 4),
        "auroc_inside_each_photo_size": inside,
        "features_guess_photo_size_accuracy": round(guessed, 4),
    }


def main():
    eyeq = pd.read_csv("outputs/phase1/eyeq.csv", dtype={"patient": str})
    results = {}
    drop_groups_quality(eyeq, results)
    reweight_to_test_mix(eyeq, results)
    by_camera_eyeq(eyeq, results)
    deepdrid_camera_check(results)

    with open("results/phase1_leakage_check.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
