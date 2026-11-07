"""Before and after pictures, from DeepDRiD only, because its licence clearly allows showing them."""
import json

import cv2
import numpy as np
import pandas as pd

from retake.degrade import degrade
from retake.enhance import apply_method

CELL = 240
METHODS = ["clahe", "graham", "routed", "routed_flat", "unet"]
FAULTS = [("blur", 2), ("uneven_light", 3), ("low_contrast", 3)]

with open("results/phase4_settings.json") as f:
    SETTINGS = json.load(f)
SETTINGS["unet_path"] = "outputs/phase4/unet_deep.pt"


def labelled(image, text):
    cell = cv2.resize(image, (CELL, CELL), interpolation=cv2.INTER_AREA)
    cv2.putText(cell, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def pad_square(image):
    size = max(image.shape[:2])
    top = (size - image.shape[0]) // 2
    left = (size - image.shape[1]) // 2
    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    return canvas


def main():
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    good = val.sort_values("Clarity", ascending=False).iloc[10]
    clean = cv2.imread(good["cache_path"])
    rows = []
    for reason, strength in FAULTS:
        damaged = degrade(clean, reason, strength, np.random.default_rng(1))
        cells = [labelled(pad_square(clean), "clean"), labelled(pad_square(damaged), f"{reason} {strength}")]
        for method in METHODS:
            fixed = apply_method(method, damaged, reason, SETTINGS)
            cells.append(labelled(pad_square(fixed), method))
        rows.append(np.hstack(cells))
    cv2.imwrite("docs/figures/phase4_added_faults.png", np.vstack(rows))

    worst = val.sort_values("Clarity").iloc[:3]
    rows = []
    for _, photo in worst.iterrows():
        image = cv2.imread(photo["cache_path"])
        cells = [labelled(pad_square(image), f"real, clarity {photo['Clarity']}")]
        for method in ["clahe", "graham", "unet"]:
            cells.append(labelled(pad_square(apply_method(method, image, "none", SETTINGS)), method))
        for reason in ["blur", "uneven_light", "low_contrast"]:
            fixed = apply_method("routed", image, reason, SETTINGS)
            cells.append(labelled(pad_square(fixed), f"as {reason}"))
        rows.append(np.hstack(cells))
    cv2.imwrite("docs/figures/phase4_real_photos.png", np.vstack(rows))


if __name__ == "__main__":
    main()
