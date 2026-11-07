"""Can simple rules name the fault? Faults are added to Good photos so the answer is known."""
import json
from itertools import product
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from retake.degrade import fault_versions
from retake.metrics import macro_f1, patient_bootstrap
from retake.quality_features import quality_features

VERSIONS_PER_PHOTO = 3
N_TUNING_PHOTOS = 1500
COLUMNS = ["focus", "laplacian", "high_freq", "light_spread", "light_range", "green_contrast", "contrast"]


def one_photo(job):
    cv2.setNumThreads(1)
    name, patient, seed = job
    image = cv2.imread(f"data/eyepacs/cache512/{name}.png")

    rows = []
    for damaged, reason, strength in fault_versions(image, seed, VERSIONS_PER_PHOTO):
        row = {"patient": patient, "reason": reason, "strength": strength}
        row.update(quality_features(damaged))
        rows.append(row)
    return rows


def build(table, seed, pool):
    jobs = []
    for offset, (_, row) in enumerate(table.iterrows()):
        jobs.append((row["image"], row["patient"], seed + offset))
    results = pool.map(one_photo, jobs, chunksize=8)

    flat = []
    for rows in results:
        flat.extend(rows)
    return pd.DataFrame(flat)


def apply_rules(table, blur_below, light_above, contrast_below, blur_column):
    answers = []
    columns = zip(table[blur_column], table["light_spread"], table["green_contrast"])
    for blur_value, light, contrast in columns:
        if blur_value < blur_below:
            answers.append("blur")
        elif light > light_above:
            answers.append("uneven_light")
        elif contrast < contrast_below:
            answers.append("low_contrast")
        else:
            answers.append("none")
    return np.array(answers)


def tune_rules(train, blur_column):
    levels = np.linspace(0.05, 0.6, 12)
    blur_options = np.quantile(train[blur_column], levels)
    light_options = np.quantile(train["light_spread"], 1 - levels)
    contrast_options = np.quantile(train["green_contrast"], levels)

    best = None
    for blur, light, contrast in product(blur_options, light_options, contrast_options):
        answers = apply_rules(train, blur, light, contrast, blur_column)
        score = macro_f1(train["reason"], answers)
        if best is None or score > best[0]:
            best = (score, float(blur), float(light), float(contrast))
    return best


def by_strength(table, answers):
    scores = {}
    for strength in [1, 2, 3]:
        # undamaged photos are scored at every strength so the class stays in the score
        keep = (table["strength"] == strength) | (table["reason"] == "none")
        keep = keep.to_numpy()
        scores[strength] = round(macro_f1(table["reason"][keep], answers[keep]), 4)
    return scores


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    good = eyeq[eyeq["quality"] == 0]
    train_photos = good[good["split"] == "train"].sample(N_TUNING_PHOTOS, random_state=0)
    val_photos = good[good["split"] == "val"]

    with Pool(12) as pool:
        train = build(train_photos, 0, pool)
        val = build(val_photos, 1000000, pool)
    print("versions", len(train), len(val), flush=True)

    results = {"attempts": []}
    best_answers = None
    best_score = -1

    for blur_column in ["laplacian", "focus", "high_freq"]:
        tuned = tune_rules(train, blur_column)
        answers = apply_rules(val, tuned[1], tuned[2], tuned[3], blur_column)
        score = macro_f1(val["reason"], answers)
        results["attempts"].append(
            {
                "attempt": f"three thresholds, blur from {blur_column}",
                "tuning_macro_f1": round(tuned[0], 4),
                "val_macro_f1": round(score, 4),
                "thresholds": [round(value, 4) for value in tuned[1:]],
                "val_by_strength": by_strength(val, answers),
            }
        )
        print(blur_column, round(score, 4), flush=True)
        if score > best_score:
            best_score = score
            best_answers = answers
            results["best"] = f"three thresholds, blur from {blur_column}"

    for depth in [3, 4, 6]:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0)
        tree.fit(train[COLUMNS], train["reason"])
        answers = tree.predict(val[COLUMNS])
        score = macro_f1(val["reason"], answers)
        results["attempts"].append(
            {
                "attempt": f"decision tree, depth {depth}",
                "val_macro_f1": round(score, 4),
                "val_by_strength": by_strength(val, answers),
            }
        )
        print("tree", depth, round(score, 4), flush=True)
        if score > best_score:
            best_score = score
            best_answers = answers
            results["best"] = f"decision tree, depth {depth}"

    results["best_val_macro_f1"] = round(best_score, 4)
    results["best_ci"] = patient_bootstrap(
        macro_f1, val["reason"], best_answers, val["patient"], rounds=300
    )

    with open("results/phase1_reasons.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
