"""Classical models for quality and disease, tuned on validation. Every attempt goes in the csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retake.metrics import accuracy, macro_f1, patient_bootstrap
from retake.quality_features import feature_names

LESIONS = ["dark_spots", "dark_area", "bright_spots", "bright_area", "dark_blob"]
QUALITY = feature_names()
SHARPNESS = ["laplacian", "tenengrad", "focus", "high_freq", "vessel_mean", "vessel_p99"]


def log_skewed(table, columns):
    values = table[columns].copy()
    for name in ["laplacian", "tenengrad", "focus"]:
        if name in values:
            values[name] = np.log1p(values[name])
    return values


def boost(**options):
    settings = {"random_state": 0, "max_iter": 300, "learning_rate": 0.05}
    settings.update(options)
    return HistGradientBoostingClassifier(**settings)


def logistic(**options):
    model = LogisticRegression(max_iter=2000, **options)
    return make_pipeline(StandardScaler(), model)


def forest():
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=12,
        random_state=0,
    )


def quality_attempts():
    all_columns = QUALITY + LESIONS
    return [
        ("logistic, sharpness only", logistic(), SHARPNESS),
        ("logistic, all quality", logistic(), QUALITY),
        ("logistic balanced, all quality", logistic(class_weight="balanced"), QUALITY),
        ("forest balanced, all quality", forest(), QUALITY),
        ("boosting, all quality", boost(), QUALITY),
        ("boosting balanced, all quality", boost(class_weight="balanced"), QUALITY),
        ("boosting, quality + lesion", boost(), all_columns),
        ("boosting balanced, quality + lesion", boost(class_weight="balanced"), all_columns),
        (
            "boosting balanced deeper, quality + lesion",
            boost(class_weight="balanced", max_iter=600, max_leaf_nodes=63, learning_rate=0.03),
            all_columns,
        ),
        (
            "boosting balanced smaller, quality + lesion",
            boost(class_weight="balanced", max_iter=300, max_leaf_nodes=15, min_samples_leaf=40),
            all_columns,
        ),
    ]


def run_eyeq_quality(eyeq, log):
    train = eyeq[eyeq["split"] == "train"]
    val = eyeq[eyeq["split"] == "val"]

    best = None
    for name, model, columns in quality_attempts():
        model.fit(log_skewed(train, columns), train["quality"])
        predicted = model.predict(log_skewed(val, columns))
        acc = accuracy(val["quality"], predicted)
        f1 = macro_f1(val["quality"], predicted)
        log.append({"task": "eyeq_quality", "attempt": name, "accuracy": acc, "macro_f1": f1})
        print(f"{name:48s} acc {acc:.4f}  macro-F1 {f1:.4f}", flush=True)
        if best is None or f1 > best["macro_f1"]:
            best = {"name": name, "macro_f1": f1, "accuracy": acc, "predicted": predicted}

    labels = val["quality"].to_numpy()
    patients = val["patient"].to_numpy()
    f1_ci = patient_bootstrap(macro_f1, labels, best["predicted"], patients)
    acc_ci = patient_bootstrap(accuracy, labels, best["predicted"], patients)
    matrix = confusion_matrix(labels, best["predicted"]).tolist()
    return {
        "best": best["name"],
        "accuracy": best["accuracy"],
        "accuracy_ci": acc_ci,
        "macro_f1": best["macro_f1"],
        "macro_f1_ci": f1_ci,
        "confusion_good_usable_reject": matrix,
    }


def run_deepdrid_quality(deepdrid, log):
    train = deepdrid[deepdrid["split"] == "train"]
    val = deepdrid[deepdrid["split"] == "val"]
    target = "Overall quality"

    best = None
    attempts = [
        ("logistic, all quality", logistic(), QUALITY),
        ("forest, all quality", forest(), QUALITY),
        ("boosting small, all quality", boost(max_leaf_nodes=7, min_samples_leaf=20), QUALITY),
    ]
    for name, model, columns in attempts:
        model.fit(log_skewed(train, columns), train[target])
        scores = model.predict_proba(log_skewed(val, columns))[:, 1]
        predicted = (scores >= 0.5).astype(int)
        result = {
            "task": "deepdrid_quality",
            "attempt": name,
            "accuracy": accuracy(val[target], predicted),
            "balanced_accuracy": balanced_accuracy_score(val[target], predicted),
            "auroc": roc_auc_score(val[target], scores),
        }
        log.append(result)
        print(name, {k: round(v, 4) for k, v in result.items() if k not in ("task", "attempt")})
        if best is None or result["auroc"] > best["auroc"]:
            best = result
    return best


def run_disease(table, name, extra_attempts, log):
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]

    best = None
    for attempt, model, columns in extra_attempts:
        model.fit(log_skewed(train, columns), train["referable"])
        scores = model.predict_proba(log_skewed(val, columns))[:, 1]
        auroc = roc_auc_score(val["referable"], scores)
        log.append({"task": f"{name}_referable", "attempt": attempt, "auroc": auroc})
        print(f"{name} {attempt:40s} AUROC {auroc:.4f}", flush=True)
        if best is None or auroc > best["auroc"]:
            ci = patient_bootstrap(roc_auc_score, val["referable"], scores, val["patient"])
            best = {"attempt": attempt, "auroc": auroc, "auroc_ci": ci}
    return best


def main():
    eyeq = pd.read_csv("outputs/phase1/eyeq.csv", dtype={"patient": str})
    deepdrid = pd.read_csv("outputs/phase1/deepdrid.csv", dtype={"patient": str})

    log = []
    summary = {}
    summary["eyeq_quality"] = run_eyeq_quality(eyeq, log)
    summary["deepdrid_quality"] = run_deepdrid_quality(deepdrid, log)

    disease_attempts = [
        ("logistic, lesion", logistic(class_weight="balanced"), LESIONS),
        ("boosting, lesion", boost(max_leaf_nodes=15), LESIONS),
        ("boosting, quality + lesion", boost(max_leaf_nodes=15), QUALITY + LESIONS),
    ]
    summary["eyeq_referable_all"] = run_disease(eyeq, "eyeq", disease_attempts, log)

    usable = eyeq[eyeq["quality"] < 2]
    summary["eyeq_referable_good_usable"] = run_disease(
        usable, "eyeq_good_usable", disease_attempts, log
    )
    summary["deepdrid_referable"] = run_disease(deepdrid, "deepdrid", disease_attempts, log)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv("results/phase1_attempts.csv", index=False)
    with open("results/phase1_quality.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
