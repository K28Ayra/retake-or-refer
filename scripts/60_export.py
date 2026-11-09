"""Export the three Phase 3 models to ONNX, shrink them to int8, and score both versions on validation.

What the phone uses, int8 weights with float activations:

    python scripts/60_export.py --weights-only --tag _weights

Full int8 through ONNX Runtime, which ruined this model. Kept so the failure can be repeated:

    python scripts/60_export.py
    python scripts/60_export.py --method percentile --percentile 99.99 --float-bias --seeds 0 --tag _p9999
    python scripts/60_export.py --keep-float /se/,blocks.0/ --seeds 0 --tag _noseb0
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import f1_score, roc_auc_score

from retake.degrade import REASONS, fault_versions
from retake.export import CalibrationPhotos, export_onnx, int8_weights, percentile_ranges, run_onnx
from retake.joint import VAL_FAULT_SEED, VERSIONS_PER_PHOTO, reason_answers
from retake.metrics import macro_f1
from retake.models import JointModel
from retake.train_utils import prepare_image, to_tensor

SMALL = "mobilenetv3_small_100"
SIZE = 384
HEADS = {"quality": 3, "referable": 1, "reason": 3, "grade": 5}
CHECK_PHOTOS = 100
CALIBRATION_PHOTOS = 300
OUT = Path("outputs/phase6")


def read_rgb(path):
    return prepare_image(cv2.imread(path), SIZE)


def load_model(seed):
    folder = f"outputs/phase3/joint_384_s{seed}_nodeepdridquality_grade"
    model = JointModel(SMALL, HEADS, pretrained=False)
    model.load_state_dict(torch.load(f"{folder}/best.pt", map_location="cpu", weights_only=True))
    return model.eval()


def largest_gap(model, session, rgb_images):
    """Biggest difference between PyTorch and ONNX on any output of any photo, before quantising."""
    worst = 0.0
    for rgb in rgb_images:
        with torch.no_grad():
            torch_out = model(to_tensor(rgb)[None])
        onnx_out = run_onnx(session, [rgb])
        for name, value in onnx_out.items():
            gap = np.abs(torch_out[name].numpy() - value).max()
            worst = max(worst, float(gap))
    return worst


def predict_all(session, rgb_images, batch=32):
    parts = {}
    for start in range(0, len(rgb_images), batch):
        answers = run_onnx(session, rgb_images[start : start + batch])
        for name, value in answers.items():
            parts.setdefault(name, []).append(value)
    joined = {}
    for name, value in parts.items():
        joined[name] = np.concatenate(value)
    return joined


def fault_photos(paths):
    images = []
    reasons = []
    for index, path in enumerate(paths):
        image = cv2.imread(path)
        seed = VAL_FAULT_SEED + index
        for damaged, reason, _ in fault_versions(image, seed, VERSIONS_PER_PHOTO):
            images.append(prepare_image(damaged, SIZE))
            reasons.append(REASONS.index(reason))
    return images, np.array(reasons)


def score(session, photos):
    eyeq = predict_all(session, photos["eyeq"])
    val = photos["eyeq_table"]
    readable = val["quality"].to_numpy() < 2
    referable = val["referable"].to_numpy()

    deepdrid = predict_all(session, photos["deepdrid"])
    faults = predict_all(session, photos["faults"])
    found = reason_answers(faults["reason"])
    return {
        "macro_f1": macro_f1(val["quality"], eyeq["quality"].argmax(axis=1)),
        "auroc_good_usable": roc_auc_score(referable[readable], eyeq["referable"][readable, 0]),
        "deepdrid_auroc": roc_auc_score(photos["deepdrid_table"]["referable"], deepdrid["referable"][:, 0]),
        "reason_f1": f1_score(photos["fault_reasons"], found, average="macro"),
    }


def float_nodes(path, parts):
    """Names of the nodes to leave unquantised: any whose name contains one of the parts."""
    if not parts:
        return []
    wanted = parts.split(",")
    names = []
    for node in onnx.load(str(path)).graph.node:
        for part in wanted:
            if part in node.name:
                names.append(node.name)
                break
    return names


def session_for(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def load_photos():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    eyeq["path"] = "data/eyepacs/cache512/" + eyeq["image"] + ".png"
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    train = eyeq[eyeq["split"] == "train"]
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    rng = np.random.default_rng(0)
    calibration_paths = rng.choice(train["path"].to_numpy(), CALIBRATION_PHOTOS, replace=False)
    good = val[val["quality"] == 0]
    fault_images, fault_reasons = fault_photos(good["path"])

    photos = {
        "eyeq_table": val,
        "eyeq": [read_rgb(path) for path in val["path"]],
        "deepdrid_table": deepdrid_val,
        "deepdrid": [read_rgb(path) for path in deepdrid_val["cache_path"]],
        "calibration": [read_rgb(path) for path in calibration_paths],
        "faults": fault_images,
        "fault_reasons": fault_reasons,
    }
    return photos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["minmax", "percentile"], default="minmax")
    parser.add_argument("--tag", default="")
    parser.add_argument("--ops", help="only quantise these op types, comma separated")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--percentile", type=float, default=99.999)
    parser.add_argument("--float-bias", action="store_true")
    parser.add_argument("--weights-only", action="store_true", help="int8 weights, float activations")
    parser.add_argument("--keep-float", default="", help="node name parts to leave in float, comma separated")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    photos = load_photos()
    print("photos loaded", flush=True)
    rows = []
    ops = args.ops.split(",") if args.ops else None
    for seed in [int(seed) for seed in args.seeds.split(",")]:
        folder = OUT / f"s{seed}"
        folder.mkdir(exist_ok=True)
        model = load_model(seed)
        float_path = folder / "fp32.onnx"
        export_onnx(model, float_path, SIZE)

        float_session = session_for(float_path)
        gap = largest_gap(model, float_session, photos["eyeq"][:CHECK_PHOTOS])

        prepared = folder / "prepared.onnx"
        quant_pre_process(str(float_path), str(prepared))
        int8_path = folder / f"int8{args.tag}.onnx"
        keep = float_nodes(prepared, args.keep_float)
        options = {}
        if args.float_bias:
            options["QuantizeBias"] = False
        method = CalibrationMethod.MinMax
        if args.method == "percentile":
            # quantize_static in this version ignores the percentile asked for, so the ranges are set here
            ranges = percentile_ranges(prepared, photos["calibration"], args.percentile, folder)
            options["TensorQuantOverrides"] = ranges
        if args.weights_only:
            int8_weights(float_path, int8_path)
        else:
            quantize_static(
                str(prepared),
                str(int8_path),
                CalibrationPhotos(photos["calibration"]),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=method,
                op_types_to_quantize=ops,
                nodes_to_exclude=keep,
                extra_options=options,
            )

        for kind, path in [("fp32", float_path), ("int8", int8_path)]:
            row = {"seed": seed, "version": kind, "method": args.method, "ops": args.ops or "all"}
            row["keep_float"] = args.keep_float
            row["float_bias"] = args.float_bias
            row["percentile"] = args.percentile if args.method == "percentile" else None
            row["float_nodes"] = len(keep)
            row["mb"] = os.path.getsize(path) / 1e6
            row["onnx_vs_torch"] = gap
            row.update(score(session_for(path), photos))
            print(row, flush=True)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.round(5).to_csv(f"results/phase6_quant{args.tag}.csv", index=False)
    print(table.groupby("version")[["mb", "macro_f1", "auroc_good_usable", "reason_f1"]].mean().round(4))


if __name__ == "__main__":
    main()
