"""Get every system the test compares ready, using validation photos only, and write configs/systems.yaml.

    python scripts/85_prepare_systems.py

full           the shipped model with the Phase 7 rule
full_standard  same model, standard gate                 (ablation: no asymmetric rule)
full_single    same model, one threshold for every grade (ablation: no quality-dependent thresholds)
noreason       final model trained without the reason head, Phase 7 rule
b1             classical quality + Phase 2 disease CNN, standard gate
b2             Phase 2 quality CNN + Phase 2 disease CNN, standard gate (the main comparison)
b3             ResNet18 quality, quality only
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from retake.decide import COSTS, apply_platt, fit_platt, fit_temperature, photo_costs, softmax
from retake.export import Named, export_onnx, int8_weights
from retake.models import JointModel, create_model
from retake.systems import classical_features, predict
from retake.tune import apply_asymmetric, apply_standard, build_data, tune_asymmetric, tune_standard

OUT = Path("outputs/phase9")
SMALL = "mobilenetv3_small_100"
SHIPPED = "outputs/phase6/s1/int8_weights.onnx"
PHASE2 = "outputs/phase2"
NO_REASON = "outputs/phase3/joint_384_s1_noreason_nodeepdridquality_grade"
CLASSICAL_COLUMNS_FROM = "scripts/11_classical_quality.py"


class PlainDumper(yaml.SafeDumper):
    """Writes repeated settings out in full, so the file reads without yaml's &id references."""

    def ignore_aliases(self, data):
        return True


def export_pair(model, name, size, outputs):
    """Float file first, then the int8-weight file that ships."""
    float_path = OUT / "models" / f"{name}_fp32.onnx"
    small_path = OUT / "models" / f"{name}.onnx"
    export_onnx(model.eval(), float_path, size, outputs)
    int8_weights(float_path, small_path)
    return str(small_path)


def phase2_model(run, architecture, outputs):
    model = create_model(architecture, outputs, pretrained=False)
    state = torch.load(f"{PHASE2}/{run}/best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def export_all():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    files = {"full": (SHIPPED, 384)}

    quality = Named(phase2_model("quality_mobilenetv3_small_100_384_s0_augfixed", SMALL, 3), "quality")
    files["b2_quality"] = (export_pair(quality, "b2_quality", 384, ["quality"]), 384)
    disease = Named(phase2_model("disease_mobilenetv3_small_100_384_s0_withreject_augfixed", SMALL, 1), "referable")
    files["b2_disease"] = (export_pair(disease, "b2_disease", 384, ["referable"]), 384)

    resnet = phase2_model("quality_resnet18_320_s0", "resnet18", 3)
    files["b3_quality"] = (export_pair(Named(resnet, "quality"), "b3_quality", 320, ["quality"]), 320)

    heads = {"quality": 3, "referable": 1, "grade": 5}
    joint = JointModel(SMALL, heads, pretrained=False)
    joint.load_state_dict(torch.load(f"{NO_REASON}/best.pt", map_location="cpu", weights_only=True))
    files["noreason"] = (export_pair(joint, "noreason", 384, ["quality", "referable", "grade"]), 384)
    return files


def classical_quality(eyeq):
    """Phase 1's best: boosting, balanced, deeper, on quality + lesion features. Refit with the
    feature code as it is now, because the Phase 1 csv was made by an earlier version of it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("classical", CLASSICAL_COLUMNS_FROM)
    classical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classical)
    columns = classical.QUALITY + classical.LESIONS

    train = eyeq[eyeq["split"] == "train"]
    train_features = classical_features("data/eyepacs/cache512/" + train["image"] + ".png")
    model = classical.boost(class_weight="balanced", max_iter=600, max_leaf_nodes=63, learning_rate=0.03)
    model.fit(classical.log_skewed(train_features, columns), train["quality"].to_numpy())
    path = OUT / "models" / "b1_classical_quality.joblib"
    joblib.dump({"model": model, "columns": columns}, path)

    val = eyeq[eyeq["split"] == "val"]
    val_features = classical_features("data/eyepacs/original/" + val["image"])
    return str(path), model.predict_proba(classical.log_skewed(val_features, columns))


def tune(name, quality_probabilities, referable, val, rule):
    data = build_data(quality_probabilities, referable, val)
    if rule == "standard":
        thresholds = tune_standard(data, COSTS)
        decisions, rates = apply_standard(data, thresholds)
    else:
        thresholds = tune_asymmetric(data, COSTS, rule)
        decisions, rates = apply_asymmetric(data, thresholds)
    cost = float(photo_costs(decisions, data["label"], rates, COSTS).mean())
    print(name, rule, {key: round(value, 4) for key, value in thresholds.items()}, "val cost", round(cost, 4))
    return {key: float(value) for key, value in thresholds.items()}, cost


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    paths = ("data/eyepacs/original/" + val["image"]).tolist()

    files = export_all()
    predictions = predict(files, paths)
    np.savez_compressed(OUT / "val_predictions.npz", **predictions)

    quality_label = val["quality"].to_numpy()
    referable_label = val["referable"].to_numpy()
    with open("configs/decision.yaml") as f:
        shipped = yaml.safe_load(f)

    systems = {}
    full_quality = softmax(predictions["full__quality"] / shipped["quality_temperature"])
    full_referable = apply_platt(predictions["full__referable"][:, 0], shipped["referable_platt"])
    base = {
        "files": {"joint": SHIPPED},
        "quality_temperature": shipped["quality_temperature"],
        "referable_platt": shipped["referable_platt"],
    }
    for name, rule in [("full", "shared"), ("full_standard", "standard"), ("full_single", "single")]:
        thresholds, cost = tune(name, full_quality, full_referable, val, rule)
        systems[name] = {**base, "rule": rule, "thresholds": thresholds, "val_cost": cost}

    temperature = fit_temperature(predictions["noreason__quality"], quality_label, binary=False)
    platt = fit_platt(predictions["noreason__referable"][:, 0], referable_label)
    quality = softmax(predictions["noreason__quality"] / temperature)
    referable = apply_platt(predictions["noreason__referable"][:, 0], platt)
    thresholds, cost = tune("noreason", quality, referable, val, "shared")
    systems["noreason"] = {
        "files": {"joint": files["noreason"][0]},
        "quality_temperature": temperature,
        "referable_platt": platt,
        "rule": "shared",
        "thresholds": thresholds,
        "val_cost": cost,
    }

    temperature = fit_temperature(predictions["b2_quality__quality"], quality_label, binary=False)
    platt = fit_platt(predictions["b2_disease__referable"][:, 0], referable_label)
    quality = softmax(predictions["b2_quality__quality"] / temperature)
    referable = apply_platt(predictions["b2_disease__referable"][:, 0], platt)
    thresholds, cost = tune("b2", quality, referable, val, "standard")
    systems["b2"] = {
        "files": {"quality": files["b2_quality"][0], "disease": files["b2_disease"][0]},
        "quality_temperature": temperature,
        "referable_platt": platt,
        "rule": "standard",
        "thresholds": thresholds,
        "val_cost": cost,
    }

    classical_path, classical_val = classical_quality(eyeq)
    thresholds, cost = tune("b1", classical_val, referable, val, "standard")
    systems["b1"] = {
        "files": {"quality": classical_path, "disease": files["b2_disease"][0]},
        "referable_platt": platt,
        "rule": "standard",
        "thresholds": thresholds,
        "val_cost": cost,
    }

    systems["b3"] = {"files": {"quality": files["b3_quality"][0]}, "size": 320, "rule": "quality only"}

    sizes = {}
    for name, (path, _) in files.items():
        sizes[name] = os.path.getsize(path) / 1e6
    with open("configs/systems.yaml", "w") as f:
        yaml.dump({"costs": COSTS, "file_mb": sizes, "systems": systems}, f, Dumper=PlainDumper, sort_keys=False)
    print(yaml.safe_dump(sizes))


if __name__ == "__main__":
    main()
