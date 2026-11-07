"""Pick settings for each fix on training photos only, so validation stays clean."""
import json
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from retake.degrade import degrade
from retake.enhance import clahe, flat_field, ssim, stretch, tilt_fix, unsharp
from retake.quality_features import fundus_mask, inner_mask

N_PHOTOS = 300
CLIPS = [1.0, 2.0, 3.0, 4.0]
FLAT_SIGMAS = [20, 40, 80, 120]
UNSHARP_SIGMAS = [2.0, 4.0, 7.0]
UNSHARP_AMOUNTS = [0.5, 1.0, 2.0, 3.0]


def retina_spread(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray[inner_mask(fundus_mask(image))].std())


def one_photo(job):
    cv2.setNumThreads(1)
    path, target_std, seed = job
    rng = np.random.default_rng(seed)
    image = cv2.imread(path)
    mask = fundus_mask(image)
    rows = []

    def add(fault, fix, damaged, output):
        rows.append({"fault": fault, "fix": fix, "ssim": ssim(image, output, mask)})

    for strength in [1, 2, 3]:
        blurred = degrade(image, "blur", strength, rng)
        shaded = degrade(image, "uneven_light", strength, rng)
        flat = degrade(image, "low_contrast", strength, rng)

        for fault, damaged in [("blur", blurred), ("uneven_light", shaded), ("low_contrast", flat)]:
            add(fault, "nothing", damaged, damaged)
            for clip in CLIPS:
                add(fault, f"clahe {clip}", damaged, clahe(damaged, clip))

        for sigma in UNSHARP_SIGMAS:
            for amount in UNSHARP_AMOUNTS:
                add("blur", f"unsharp {sigma} {amount}", blurred, unsharp(blurred, sigma, amount))
        for sigma in FLAT_SIGMAS:
            add("uneven_light", f"flat {sigma}", shaded, flat_field(shaded, sigma))
        add("uneven_light", "tilt", shaded, tilt_fix(shaded))
        add("low_contrast", "stretch", flat, stretch(flat, target_std))
    return rows


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    good = eyeq[(eyeq["split"] == "train") & (eyeq["quality"] == 0)]
    photos = good.sample(N_PHOTOS, random_state=0)
    paths = ("data/eyepacs/cache512/" + photos["image"] + ".png").tolist()

    with Pool(12) as pool:
        spreads = pool.map(retina_spread, paths)
        target_std = float(np.median(spreads))
        jobs = []
        for offset, path in enumerate(paths):
            jobs.append((path, target_std, 2000000 + offset))
        results = pool.map(one_photo, jobs, chunksize=4)

    flat_rows = []
    for rows in results:
        flat_rows.extend(rows)
    table = pd.DataFrame(flat_rows)
    means = table.groupby(["fault", "fix"])["ssim"].mean().round(4)
    print(means.to_string())

    every_fault = means.unstack("fault").dropna()
    clip_rows = every_fault[every_fault.index.str.startswith("clahe")].mean(axis=1)
    best_clip = float(clip_rows.idxmax().split()[1])

    unsharp_rows = means["blur"][means["blur"].index.str.startswith("unsharp")]
    best_unsharp = unsharp_rows.idxmax().split()
    flat_rows = means["uneven_light"][means["uneven_light"].index.str.startswith("flat")]
    best_flat = flat_rows.idxmax().split()

    light_fix = "flat"
    if means["uneven_light"]["tilt"] > flat_rows.max():
        light_fix = "tilt"

    contrast = means["low_contrast"]
    contrast_clahe = contrast[contrast.index.str.startswith("clahe")]
    contrast_fix = "stretch"
    if contrast_clahe.max() > contrast["stretch"]:
        contrast_fix = "clahe"

    settings = {
        "clahe_clip": best_clip,
        "light_fix": light_fix,
        "flat_sigma": int(best_flat[1]),
        "unsharp_sigma": float(best_unsharp[1]),
        "unsharp_amount": float(best_unsharp[2]),
        "contrast_fix": contrast_fix,
        "target_std": round(target_std, 2),
    }
    with open("results/phase4_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    means.reset_index().to_csv("results/phase4_tuning.csv", index=False)
    print(settings)


if __name__ == "__main__":
    main()
