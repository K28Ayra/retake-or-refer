"""Score each fix against the clean photo: added faults first, then Good photos left as they are."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from retake.degrade import REASONS, fault_versions
from retake.enhance import apply_method, psnr, ssim
from retake.joint import VAL_FAULT_SEED, VERSIONS_PER_PHOTO, reason_answers
from retake.metrics import patient_bootstrap
from retake.quality_features import fundus_mask

JOINT = "outputs/phase3/joint_384_s0_nodeepdridquality_grade"
UNET = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase4/unet_deep.pt"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "results/phase4_synthetic.json"
TIMING_PHOTOS = 100

with open("results/phase4_settings.json") as f:
    SETTINGS = json.load(f)
SETTINGS["unet_path"] = UNET

METHODS = ["none", "clahe", "graham", "flat_field", "routed", "routed_flat", "routed_oracle"]
if Path(UNET).exists():
    METHODS.append("unet")


def run_method(method, image, model_reason, true_reason):
    if method == "routed_oracle":
        return apply_method("routed", image, true_reason, SETTINGS)
    return apply_method(method, image, model_reason, SETTINGS)


def one_photo(job):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    path, patient, seed, model_reasons = job
    clean = cv2.imread(path)
    mask = fundus_mask(clean)

    rows = []
    versions = fault_versions(clean, seed, VERSIONS_PER_PHOTO)
    for version, (damaged, reason, strength) in enumerate(versions):
        for method in METHODS:
            output = run_method(method, damaged, model_reasons[version], reason)
            rows.append(
                {
                    "patient": patient,
                    "photo": seed,
                    "version": version,
                    "reason": reason,
                    "strength": strength,
                    "model_reason": model_reasons[version],
                    "method": method,
                    "ssim": ssim(clean, output, mask),
                    "psnr": psnr(clean, output, mask),
                }
            )
    return rows


def one_good_photo(job):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    path, model_reason = job
    clean = cv2.imread(path)
    mask = fundus_mask(clean)
    row = {"model_reason": model_reason}
    for method in METHODS:
        output = run_method(method, clean, model_reason, "none")
        row[method] = ssim(clean, output, mask)
    return row


def gap_ci(table, column, method):
    """Method minus always-CLAHE on the same damaged photos, resampling whole patients."""
    keep = ["photo", "version", column, "patient"]
    ours = table[table["method"] == method][keep]
    theirs = table[table["method"] == "clahe"][keep]
    joined = ours.merge(theirs, on=["photo", "version", "patient"], suffixes=("_ours", "_theirs"))

    def mean_gap(first, second):
        return float((second - first).mean())

    low, high = patient_bootstrap(
        mean_gap,
        joined[f"{column}_theirs"].to_numpy(),
        joined[f"{column}_ours"].to_numpy(),
        joined["patient"].to_numpy(),
    )
    middle = mean_gap(joined[f"{column}_theirs"], joined[f"{column}_ours"])
    return {"gap": round(middle, 4), "low": round(low, 4), "high": round(high, 4)}


def time_methods(paths):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    photos = []
    for path in paths[:TIMING_PHOTOS]:
        photos.append(cv2.imread(path))

    times = {}
    for method in METHODS:
        reasons = [None]
        if method.startswith("routed"):
            reasons = ["blur", "uneven_light", "low_contrast"]
        for reason in reasons:
            spent = []
            for photo in photos:
                start = time.perf_counter()
                apply_method("routed" if method == "routed_oracle" else method, photo, reason, SETTINGS)
                spent.append((time.perf_counter() - start) * 1000)
            name = method if reason is None else f"{method} {reason}"
            times[name] = {
                "median_ms": round(float(np.median(spent)), 1),
                "p95_ms": round(float(np.percentile(spent, 95)), 1),
            }
    return times


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    good = val[val["quality"] == 0].reset_index(drop=True)
    paths = ("data/eyepacs/cache512/" + good["image"] + ".png").tolist()

    saved = np.load(f"{JOINT}/fault_val.npz")
    answers = reason_answers(saved["logits"]).reshape(-1, VERSIONS_PER_PHOTO)

    jobs = []
    for index, path in enumerate(paths):
        names = [REASONS[answer] for answer in answers[index]]
        jobs.append((path, good["patient"][index], VAL_FAULT_SEED + index, names))

    real = np.load(f"{JOINT}/eyeq_val.npz")
    good_rows = (val["quality"] == 0).to_numpy()
    good_answers = reason_answers(real["reason"])[good_rows]
    good_jobs = []
    for index, path in enumerate(paths):
        good_jobs.append((path, REASONS[good_answers[index]]))

    with Pool(12) as pool:
        results = pool.map(one_photo, jobs, chunksize=8)
        good_results = pool.map(one_good_photo, good_jobs, chunksize=8)

    flat = []
    for rows in results:
        flat.extend(rows)
    table = pd.DataFrame(flat)
    faulted = table[table["reason"] != "none"]

    summary = {"settings": SETTINGS, "synthetic": {}, "by_fault": {}, "gap_over_clahe": {}}
    for method in METHODS:
        inside = faulted[faulted["method"] == method]
        summary["synthetic"][method] = {
            "ssim": round(float(inside["ssim"].mean()), 4),
            "psnr": round(float(inside["psnr"].mean()), 2),
            "ssim_mildest": round(float(inside[inside["strength"] == 1]["ssim"].mean()), 4),
            "ssim_worst": round(float(inside[inside["strength"] == 3]["ssim"].mean()), 4),
        }
        by_fault = {}
        for reason in ["blur", "uneven_light", "low_contrast"]:
            part = inside[inside["reason"] == reason]
            by_fault[reason] = {
                "ssim": round(float(part["ssim"].mean()), 4),
                "psnr": round(float(part["psnr"].mean()), 2),
            }
        summary["by_fault"][method] = by_fault

    for method in METHODS:
        if method in ["clahe", "none"]:
            continue
        summary["gap_over_clahe"][method] = {
            "ssim": gap_ci(faulted, "ssim", method),
            "psnr": gap_ci(faulted, "psnr", method),
        }

    good_table = pd.DataFrame(good_results)
    summary["good_photos"] = {"share_changed_by_routed": round(float((good_table["model_reason"] != "none").mean()), 4)}
    for method in METHODS:
        summary["good_photos"][method] = {
            "ssim": round(float(good_table[method].mean()), 4),
            "share_below_098": round(float((good_table[method] < 0.98).mean()), 4),
        }

    summary["cpu_ms_one_thread_512"] = time_methods(paths)

    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
