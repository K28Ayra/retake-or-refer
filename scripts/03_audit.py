"""Check splits and images, write results/phase0_audit.json.

EyeQ test images are only counted from the labels here. They stay unopened until the final run.
"""
import hashlib
import json
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from retake.data import shared_patients


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_image(path):
    info = {"path": path, "readable": False, "bytes": 0}
    if not Path(path).exists():
        return info
    image = cv2.imread(path)
    if image is None:
        return info
    info["readable"] = True
    info["bytes"] = Path(path).stat().st_size
    info["height"] = image.shape[0]
    info["width"] = image.shape[1]
    info["sha256"] = file_hash(path)
    return info


def summarise_images(paths, pool):
    rows = pool.map(inspect_image, paths, chunksize=16)
    rows = pd.DataFrame(rows)
    readable = rows[rows["readable"]]
    duplicates = readable["sha256"].duplicated().sum()
    summary = {
        "listed": len(rows),
        "readable": len(readable),
        "duplicates_by_hash": int(duplicates),
    }
    if len(readable) > 0:
        summary["median_kb"] = round(float(readable["bytes"].median()) / 1024, 1)
        sizes = readable["width"].astype(str) + "x" + readable["height"].astype(str)
        summary["common_sizes"] = sizes.value_counts().head(3).to_dict()
    return summary


def eyeq_paths(table):
    paths = []
    for _, row in table.iterrows():
        if row["split"] == "train":
            paths.append(f"data/eyepacs/cache512/{row['image']}.png")
        else:
            paths.append(f"data/eyepacs/original/{row['image']}")
    return paths


def main():
    splits = Path("data/splits")
    eyeq = pd.read_csv(splits / "eyeq.csv", dtype={"patient": str})
    eyeq_test = pd.read_csv(splits / "eyeq_test.csv", dtype={"patient": str})
    deepdrid = pd.read_csv(splits / "deepdrid.csv", dtype={"patient": str})
    deepdrid_test = pd.read_csv(splits / "deepdrid_test.csv", dtype={"patient": str})

    audit = {"label_counts": {}, "patient_overlap": {}, "images": {}, "split_sha256": {}}

    groups = {
        "eyeq_train": eyeq[eyeq["split"] == "train"],
        "eyeq_val": eyeq[eyeq["split"] == "val"],
        "eyeq_test": eyeq_test,
        "deepdrid_train": deepdrid[deepdrid["split"] == "train"],
        "deepdrid_val": deepdrid[deepdrid["split"] == "val"],
        "deepdrid_test": deepdrid_test,
    }

    for name, table in groups.items():
        counts = {
            "images": len(table),
            "patients": int(table["patient"].nunique()),
            "referable": int(table["referable"].sum()),
        }
        if "quality" in table:
            counts["quality"] = table["quality"].value_counts().sort_index().to_dict()
        else:
            quality = table["Overall quality"].value_counts().sort_index()
            counts["overall_quality"] = quality.to_dict()
        audit["label_counts"][name] = counts

    # patient numbers only mean something within one dataset
    for family in ["eyeq", "deepdrid"]:
        names = [name for name in groups if name.startswith(family)]
        for first, second in combinations(names, 2):
            overlap = shared_patients(groups[first], groups[second])
            audit["patient_overlap"][f"{first}|{second}"] = len(overlap)

    with Pool(8) as pool:
        audit["images"]["eyeq_train"] = summarise_images(
            eyeq_paths(groups["eyeq_train"]), pool
        )
        audit["images"]["eyeq_val"] = summarise_images(
            eyeq_paths(groups["eyeq_val"]), pool
        )
        for name in ["deepdrid_train", "deepdrid_val", "deepdrid_test"]:
            paths = groups[name]["image_path"].tolist()
            audit["images"][name] = summarise_images(paths, pool)

    for path in sorted(splits.glob("*.csv")):
        audit["split_sha256"][path.name] = file_hash(path)

    Path("results").mkdir(exist_ok=True)
    with open("results/phase0_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=int)
    print(json.dumps(audit, indent=2, default=int))


if __name__ == "__main__":
    main()
