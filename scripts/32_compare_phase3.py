"""Check the chosen joint model against the two separate Phase 2 models on the same photos."""
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from retake.joint import reason_answers
from retake.metrics import macro_f1, paired_gap, patient_bootstrap
from retake.models import JointModel, create_model

SMALL = "mobilenetv3_small_100"
SIZE = 384
SEEDS = [0, 1, 2]
REPEATS = 60
WARMUP = 10
QUALITY_NAMES = ["Good", "Usable", "Reject"]
REASON_NAMES = ["none", "blur", "uneven_light", "low_contrast"]


def single_logits(run, name):
    return np.load(f"outputs/phase2/{run}/{name}_val_logits.npy")


def spearman_size(scores, labels):
    return spearmanr(scores, labels).statistic


def weighted_kappa(labels, predicted):
    return cohen_kappa_score(labels, predicted, weights="quadratic")


def time_model(model):
    torch.set_num_threads(1)
    model = model.eval()
    image = torch.zeros(1, 3, SIZE, SIZE)
    times = []
    with torch.no_grad():
        for _ in range(REPEATS):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times[WARMUP:])), 1)


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    quality = val["quality"].to_numpy()
    patients = val["patient"].to_numpy()
    readable = quality < 2
    referable = val["referable"].to_numpy()

    good_patients = patients[quality == 0]
    fault_patients = np.repeat(good_patients, 3)

    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    results = {"seeds": {}}
    for seed in SEEDS:
        folder = f"outputs/phase3/joint_384_s{seed}_nodeepdridquality_grade"
        joint = np.load(f"{folder}/eyeq_val.npz")
        joint_deepdrid = np.load(f"{folder}/deepdrid_val.npz")
        faults = np.load(f"{folder}/fault_val.npz")

        quality_run = f"quality_{SMALL}_384_s{seed}_augfixed"
        disease_run = f"disease_{SMALL}_384_s{seed}_withreject_augfixed"
        single_quality = single_logits(quality_run, "eyeq").argmax(axis=1)
        single_disease = single_logits(disease_run, "eyeq")[:, 0]

        joint_quality = joint["quality"].argmax(axis=1)
        joint_disease = joint["referable"][:, 0]

        answers = reason_answers(faults["logits"])
        reasons = faults["reasons"]

        real_flags = {}
        real_answers = reason_answers(joint["reason"])
        for grade, name in enumerate(QUALITY_NAMES):
            inside = real_answers[quality == grade]
            shares = {}
            for index, reason in enumerate(REASON_NAMES):
                shares[reason] = round(float((inside == index).mean()), 3)
            real_flags[name] = shares

        grade_labels = val["DR_grade"].astype(int).to_numpy()
        grade_guess = joint["grade"].argmax(axis=1)
        deepdrid_grades = deepdrid_val["DR_grade"].astype(int).to_numpy()
        deepdrid_guess = joint_deepdrid["grade"].argmax(axis=1)

        blur = joint_deepdrid["reason"][:, 0]
        blur_ci = patient_bootstrap(
            spearman_size,
            deepdrid_val["Clarity"].to_numpy(),
            blur,
            deepdrid_val["patient"].to_numpy(),
        )

        results["seeds"][seed] = {
            "macro_f1_joint_minus_single_ci": paired_gap(
                macro_f1, quality, joint_quality, single_quality, patients
            ),
            "auroc_joint_minus_single_ci": paired_gap(
                roc_auc_score,
                referable[readable],
                joint_disease[readable],
                single_disease[readable],
                patients[readable],
            ),
            "reason_f1": macro_f1(reasons, answers),
            "reason_f1_ci": patient_bootstrap(macro_f1, reasons, answers, fault_patients),
            "blur_clarity_spearman": spearman_size(blur, deepdrid_val["Clarity"]),
            "blur_clarity_spearman_ci": blur_ci,
            "reason_on_real_photos": real_flags,
            "grade_kappa_eyeq_readable": weighted_kappa(grade_labels[readable], grade_guess[readable]),
            "grade_kappa_deepdrid": weighted_kappa(deepdrid_grades, deepdrid_guess),
        }

    head_sizes = {"quality": 3, "referable": 1, "reason": 3, "grade": 5}
    joint_ms = time_model(JointModel(SMALL, head_sizes, pretrained=False))
    quality_ms = time_model(create_model(SMALL, 3, pretrained=False))
    disease_ms = time_model(create_model(SMALL, 1, pretrained=False))
    results["cpu_ms_one_thread"] = {
        "joint": joint_ms,
        "quality": quality_ms,
        "disease": disease_ms,
        "share_of_two": round(joint_ms / (quality_ms + disease_ms), 3),
    }

    with open("results/phase3_compare.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
