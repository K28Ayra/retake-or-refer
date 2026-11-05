"""Compute classical features for EyeQ train/val and DeepDRiD train/val into outputs/phase1."""
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retake.images import crop_to_retina, resize_long_side
from retake.quality_features import lesion_features, quality_features

OUT = Path("outputs/phase1")


def load_512(path):
    image = cv2.imread(path)
    return resize_long_side(crop_to_retina(image), 512)


def features_for(path):
    cv2.setNumThreads(1)
    image = load_512(path)
    row = {"path": path}
    row.update(quality_features(image))
    row.update(lesion_features(image))
    return row


def run(table, name, pool):
    rows = pool.map(features_for, table["image_path"].tolist(), chunksize=16)
    features = pd.DataFrame(rows)
    keep = table.drop(columns=["image_path"]).reset_index(drop=True)
    merged = pd.concat([keep, features], axis=1)
    merged.to_csv(OUT / f"{name}.csv", index=False)
    print(name, merged.shape, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    eyeq = pd.read_csv("data/splits/eyeq.csv")
    eyeq["image_path"] = "data/eyepacs/cache512/" + eyeq["image"] + ".png"

    deepdrid = pd.read_csv("data/splits/deepdrid.csv")

    with Pool(12) as pool:
        run(eyeq, "eyeq", pool)
        run(deepdrid, "deepdrid", pool)


if __name__ == "__main__":
    main()
