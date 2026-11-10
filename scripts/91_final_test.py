"""The one run on test data, with everything frozen by 90_freeze.py.

    python scripts/91_final_test.py --dry-run   # same code on validation photos, to check the script
    python scripts/91_final_test.py             # the real run, once

Writes results/final_*.csv and results/final_summary.json (dry run: results/dryrun_*).
"""
import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from retake.decide import (
    CLEAR,
    COSTS,
    DECISIONS,
    REFER,
    REJECT,
    RETAKE,
    apply_platt,
    asymmetric_rule,
    photo_costs,
    reason_answers,
    softmax,
    standard_gate,
)
from retake.degrade import REASONS
from retake.enhance import clahe, routed
from retake.images import crop_to_retina, resize_long_side
from retake.metrics import accuracy, macro_f1, paired_gap, patient_bootstrap
from retake.systems import classical_features, predict
from retake.tune import follow_up

FREEZE = "results/final_freeze.json"
DECIDING = ["full", "full_standard", "full_single", "noreason", "b1", "b2"]
SENDING = {
    "full": ("jpeg444", True, [768, 512, 384], [95, 85, 75, 60]),
    "b2": ("jpeg", False, [1024, 768, 512], [95, 85, 75, 60, 50]),
}
MAX_DROP = 0.02
MAX_GRADE_CHANGE = 0.05
UWF = "data/raw/deepdrid/ultra-widefield_images"


def check_freeze():
    with open(FREEZE) as f:
        frozen = json.load(f)
    for path, expected in frozen["files"].items():
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"{path} changed after the freeze")
    return frozen


def eyeq_set(dry_run):
    if dry_run:
        table = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
        table = table[table["split"] == "val"].reset_index(drop=True)
    else:
        table = pd.read_csv("data/splits/eyeq_test.csv", dtype={"patient": str})
    table["path"] = "data/eyepacs/original/" + table["image"]
    return table


def deepdrid_set(dry_run):
    if dry_run:
        table = pd.read_csv("data/splits/deepdrid.csv", dtype={"patient": str})
        table = table[table["split"] == "val"].reset_index(drop=True)
    else:
        table = pd.read_csv("data/splits/deepdrid_test.csv", dtype={"patient": str})
    table["path"] = table["image_path"]
    return table


def uwf_set():
    """All 256 ultra-widefield photos. None was used for training or tuning."""
    rows = []
    for folder in ["ultra-widefield-training", "ultra-widefield-validation"]:
        labels = pd.read_csv(f"{UWF}/{folder}/{folder}.csv")
        for _, row in labels.iterrows():
            image = row["image_id"]
            patient = image.split("_")[0]
            rows.append({"image": image, "patient": patient, "DR_grade": int(row["DR_level"]),
                         "path": f"{UWF}/{folder}/Images/{patient}/{image}.jpg"})
    labels = pd.read_excel(f"{UWF}/Online-Challenge3-Evaluation/Challenge3_labels.xlsx")
    for _, row in labels.iterrows():
        image = row["image_id"]
        patient = image.split("_")[0]
        rows.append({"image": image, "patient": patient, "DR_grade": int(row["UWF_DR_Levels"]),
                     "path": f"{UWF}/Online-Challenge3-Evaluation/Images/{patient}/{image}.jpg"})
    table = pd.DataFrame(rows)
    table["referable"] = (table["DR_grade"] >= 2).astype(int)
    # patient 121's two training photos are saved as r2.jpg and l3.jpg, without the patient number
    for index, path in table["path"].items():
        if not Path(path).exists():
            folder = Path(path).parent
            short_name = Path(path).name.split("_", 1)[1]
            table.loc[index, "path"] = str(folder / short_name)
    missing = [path for path in table["path"] if not Path(path).exists()]
    if missing:
        sys.exit(f"{len(missing)} ultra-widefield photos not found, first {missing[0]}")
    return table


def classical_quality(paths, model_file):
    import importlib.util

    spec = importlib.util.spec_from_file_location("classical", "scripts/11_classical_quality.py")
    classical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classical)
    saved = joblib.load(model_file)
    features = classical_features(paths)
    return saved["model"].predict_proba(classical.log_skewed(features, saved["columns"]))


def model_files(systems):
    files = {
        "joint": (systems["full"]["files"]["joint"], 384),
        "noreason": (systems["noreason"]["files"]["joint"], 384),
        "b2_quality": (systems["b2"]["files"]["quality"], 384),
        "b2_disease": (systems["b2"]["files"]["disease"], 384),
        "b3_quality": (systems["b3"]["files"]["quality"], 320),
    }
    return files


def calibrated(system, name, raw, classical=None):
    """Calibrated quality probabilities and referable probability for one system."""
    if name in ["full", "full_standard", "full_single"]:
        quality_logits, referable_logits = raw["joint__quality"], raw["joint__referable"][:, 0]
    elif name == "noreason":
        quality_logits, referable_logits = raw["noreason__quality"], raw["noreason__referable"][:, 0]
    else:
        quality_logits, referable_logits = raw["b2_quality__quality"], raw["b2_disease__referable"][:, 0]
    referable = apply_platt(referable_logits, system["referable_platt"])
    if name == "b1":
        return classical, referable
    return softmax(quality_logits / system["quality_temperature"]), referable


def decide(system, quality_probabilities, referable):
    quality_class = quality_probabilities.argmax(axis=1)
    if system["rule"] == "standard":
        return standard_gate(quality_class, referable, system["thresholds"]["refer"])
    return asymmetric_rule(quality_class, referable, system["thresholds"])


def good_threshold(system):
    if system["rule"] == "standard":
        return system["thresholds"]["refer"]
    return system["thresholds"]["good_refer"]


def interval(metric, labels, scores, patients):
    low, high = patient_bootstrap(metric, labels, scores, patients)
    return [round(low, 4), round(high, 4)]


def score_eyeq(systems, raw, classical, table, b3_quality):
    true_quality = table["quality"].to_numpy()
    label = table["referable"].to_numpy()
    patients = table["patient"].to_numpy()
    readable = true_quality < 2
    good = true_quality == 0

    results = {}
    costs = {}
    for name in DECIDING:
        system = systems[name]
        quality, referable = calibrated(system, name, raw, classical)
        decisions = decide(system, quality, referable)
        rates = follow_up(referable, label, good, good_threshold(system))
        cost = photo_costs(decisions, label, rates, COSTS)
        costs[name] = cost
        predicted_quality = quality.argmax(axis=1)
        sick = label == 1
        sick_readable = sick & readable
        well_readable = ~sick & readable
        refer_now = float((decisions[sick_readable] == REFER).mean())
        retaken = float((decisions[sick_readable] == RETAKE).mean())
        results[name] = {
            "quality_accuracy": accuracy(true_quality, predicted_quality),
            "quality_accuracy_ci": interval(accuracy, true_quality, predicted_quality, patients),
            "quality_macro_f1": macro_f1(true_quality, predicted_quality),
            "quality_macro_f1_ci": interval(macro_f1, true_quality, predicted_quality, patients),
            "auroc_good_usable": roc_auc_score(label[readable], referable[readable]),
            "auroc_good_usable_ci": interval(roc_auc_score, label[readable], referable[readable], patients[readable]),
            "auroc_all": roc_auc_score(label, referable),
            "sensitivity_now_good_usable": refer_now,
            "sensitivity_after_retake_good_usable": refer_now + retaken * (1 - rates[0]),
            "specificity_good_usable": float((decisions[well_readable] == CLEAR).mean()),
            "expected_cost": float(cost.mean()),
            "refer_rate": float((decisions == REFER).mean()),
            "retake_rate": float((decisions == RETAKE).mean()),
            "clear_rate": float((decisions == CLEAR).mean()),
            "rejects_cleared": int(((predicted_quality == REJECT) & (decisions == CLEAR)).sum()),
            "follow_up_rates": rates,
        }

    full_kappa = cohen_kappa_score(table["DR_grade"], raw["joint__grade"].argmax(axis=1), weights="quadratic")
    results["full"]["grade_kappa"] = float(full_kappa)
    predicted = b3_quality.argmax(axis=1)
    results["b3"] = {
        "quality_accuracy": accuracy(true_quality, predicted),
        "quality_accuracy_ci": interval(accuracy, true_quality, predicted, patients),
        "quality_macro_f1": macro_f1(true_quality, predicted),
        "quality_macro_f1_ci": interval(macro_f1, true_quality, predicted, patients),
    }
    return results, costs


def main_comparison(systems, raw, classical, table, costs):
    """Full against B2: cost and AUROC, each on the same photos, resampling patients."""
    patients = table["patient"].to_numpy()
    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())
    rng = np.random.default_rng(0)
    changes = []
    for _ in range(2000):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        changes.append(costs["full"][rows].mean() / costs["b2"][rows].mean() - 1)

    readable = table["quality"].to_numpy() < 2
    label = table["referable"].to_numpy()
    _, full_referable = calibrated(systems["full"], "full", raw)
    _, b2_referable = calibrated(systems["b2"], "b2", raw, classical)
    gap = paired_gap(roc_auc_score, label[readable], full_referable[readable], b2_referable[readable],
                     patients[readable])
    return {
        "cost_change": float(costs["full"].mean() / costs["b2"].mean() - 1),
        "cost_change_ci": [float(value) for value in np.percentile(changes, [2.5, 97.5])],
        "auroc_gap_full_minus_b2_ci": [round(gap[0], 4), round(gap[1], 4)],
    }


def score_outside(systems, raw, classical, table, eyeq_rates, with_quality_label):
    label = table["referable"].to_numpy()
    patients = table["patient"].to_numpy()
    results = {}
    for name in DECIDING:
        system = systems[name]
        quality, referable = calibrated(system, name, raw, classical)
        decisions = decide(system, quality, referable)
        sick = label == 1
        row = {
            "photos": int(len(label)),
            "auroc": roc_auc_score(label, referable),
            "auroc_ci": interval(roc_auc_score, label, referable, patients),
            "sensitivity_now": float((decisions[sick] == REFER).mean()),
            "specificity": float((decisions[~sick] == CLEAR).mean()),
            "refer_rate": float((decisions == REFER).mean()),
            "retake_rate": float((decisions == RETAKE).mean()),
            "clear_rate": float((decisions == CLEAR).mean()),
            "predicted_reject_share": float((quality.argmax(axis=1) == REJECT).mean()),
            "expected_cost": float(photo_costs(decisions, label, eyeq_rates[name], COSTS).mean()),
        }
        per_patient = pd.DataFrame({"patient": patients, "label": label, "score": referable})
        per_patient = per_patient.groupby("patient").max()
        if per_patient["label"].nunique() == 2:
            row["auroc_per_patient"] = roc_auc_score(per_patient["label"], per_patient["score"])
        if with_quality_label:
            # DeepDRiD's Overall quality: 1 means good enough to grade
            gradable = table["Overall quality"].to_numpy() == 1
            row["binary_quality_accuracy"] = float(((quality.argmax(axis=1) != REJECT) == gradable).mean())
        results[name] = row
    if "DR_grade" in table:
        grades = raw["joint__grade"].argmax(axis=1)
        results["full"]["grade_kappa"] = float(cohen_kappa_score(table["DR_grade"], grades, weights="quadratic"))
    return results


def sending(systems, files, table, raw_original, dry_run):
    """Smallest file each system can be sent at, judged with its frozen thresholds."""
    readable = table["quality"].to_numpy() < 2
    label = table["referable"].to_numpy()
    rows = []
    for name, (file_format, retina_first, sides, qualities) in SENDING.items():
        system = systems[name]
        if name == "full":
            needed = {"joint": files["joint"]}
        else:
            needed = {"b2_quality": files["b2_quality"], "b2_disease": files["b2_disease"]}
        quality, referable = calibrated(system, name, raw_original)
        start = decide(system, quality, referable)
        start_grade = quality.argmax(axis=1)
        if dry_run:
            sides, qualities = sides[:1], qualities[:2]
        for side in sides:
            for level in qualities:
                raw = predict(needed, table["path"].tolist(), (file_format, level, side, retina_first))
                quality, referable = calibrated(system, name, raw)
                decisions = decide(system, quality, referable)
                sick = (label == 1) & readable
                well = (label == 0) & readable
                row = {
                    "system": name,
                    "format": file_format + ("_retina" if retina_first else ""),
                    "side": side,
                    "file_quality": level,
                    "kb": float(raw["bytes"].mean() / 1024),
                    "sensitivity_drop": float((start[sick] == REFER).mean() - (decisions[sick] == REFER).mean()),
                    "specificity_drop": float((start[well] == CLEAR).mean() - (decisions[well] == CLEAR).mean()),
                    "grade_change": float((quality.argmax(axis=1) != start_grade).mean()),
                    "auroc_good_usable": roc_auc_score(label[readable], referable[readable]),
                }
                row["passes"] = (
                    row["sensitivity_drop"] < MAX_DROP
                    and row["specificity_drop"] < MAX_DROP
                    and row["grade_change"] <= MAX_GRADE_CHANGE
                )
                print(row, flush=True)
                rows.append(row)
    return pd.DataFrame(rows)


def enhanced_photo(method, reasons, settings, index, photo):
    retina = resize_long_side(crop_to_retina(photo), 512)
    if method == "clahe":
        return clahe(retina, settings["clahe_clip"])
    return routed(retina, reasons[index], settings)


def enhancement(systems, files, table, raw_original):
    """What the shipped model does if it is fed an enhanced photo instead of the original."""
    with open("results/phase4_settings.json") as f:
        settings = json.load(f)
    reasons = [REASONS[index] for index in reason_answers(raw_original["joint__reason"])]
    readable = table["quality"].to_numpy() < 2
    label = table["referable"].to_numpy()
    true_quality = table["quality"].to_numpy()
    results = {}
    for method in ["clahe", "routed"]:
        # done in the loader workers: OpenCV work in this process before they fork made them crash
        change = functools.partial(enhanced_photo, method, reasons, settings)
        raw = predict({"joint": files["joint"]}, table["path"].tolist(), change=change)
        quality, referable = calibrated(systems["full"], "full", raw)
        results[method] = {
            "quality_macro_f1": macro_f1(true_quality, quality.argmax(axis=1)),
            "auroc_good_usable": roc_auc_score(label[readable], referable[readable]),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prefix = "results/dryrun_" if args.dry_run else "results/final_"
    if not args.dry_run:
        check_freeze()

    with open("configs/systems.yaml") as f:
        systems = yaml.safe_load(f)["systems"]
    files = model_files(systems)
    summary = {}

    eyeq = eyeq_set(args.dry_run)
    raw = predict(files, eyeq["path"].tolist())
    classical = classical_quality(eyeq["path"].tolist(), systems["b1"]["files"]["quality"])
    b3_quality = softmax(raw["b3_quality__quality"])
    summary["eyeq"], costs = score_eyeq(systems, raw, classical, eyeq, b3_quality)
    summary["full_vs_b2"] = main_comparison(systems, raw, classical, eyeq, costs)
    print(json.dumps(summary["full_vs_b2"]), flush=True)
    eyeq_rates = {name: summary["eyeq"][name]["follow_up_rates"] for name in DECIDING}

    deepdrid = deepdrid_set(args.dry_run)
    raw_deepdrid = predict(files, deepdrid["path"].tolist())
    classical_deepdrid = classical_quality(deepdrid["path"].tolist(), systems["b1"]["files"]["quality"])
    summary["deepdrid"] = score_outside(systems, raw_deepdrid, classical_deepdrid, deepdrid, eyeq_rates, True)

    if not args.dry_run:
        uwf = uwf_set()
        raw_uwf = predict(files, uwf["path"].tolist())
        classical_uwf = classical_quality(uwf["path"].tolist(), systems["b1"]["files"]["quality"])
        summary["ultra_widefield"] = score_outside(systems, raw_uwf, classical_uwf, uwf, eyeq_rates, False)

    summary["enhancement"] = enhancement(systems, files, eyeq, raw)
    sent = sending(systems, files, eyeq, raw, args.dry_run)
    sent.round(4).to_csv(prefix + "sending.csv", index=False)
    smallest = {}
    for name, part in sent.groupby("system"):
        passing = part[part["passes"]]
        smallest[name] = None if passing.empty else passing.sort_values("kb").iloc[0].to_dict()
    summary["smallest_passing_file"] = smallest

    with open(prefix + "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
