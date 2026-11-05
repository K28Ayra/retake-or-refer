"""Test the not-a-fundus check on ordinary photos and plain frames, then time the features."""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retake.images import crop_to_retina, resize_long_side
from retake.quality_features import lesion_features, looks_like_fundus, quality_features

N_PHOTOS = 500
N_TIMED = 300


def load_512(path):
    image = cv2.imread(str(path))
    return resize_long_side(crop_to_retina(image), 512)


def plain_frames(rng):
    """The kind of thing a camera sees when it is not on an eye."""
    frames = []
    for _ in range(25):
        colour = rng.integers(0, 256, 3).astype(np.uint8)
        frames.append(np.full((384, 512, 3), colour, dtype=np.uint8))
    for _ in range(25):
        frames.append(rng.integers(0, 256, (384, 512, 3)).astype(np.uint8))
    for _ in range(25):
        level = int(rng.integers(0, 25))
        frames.append(np.full((384, 512, 3), level, dtype=np.uint8))
    for _ in range(25):
        ramp = np.linspace(0, 255, 512).astype(np.uint8)
        gray = np.tile(ramp, (384, 1))
        frames.append(cv2.merge([gray, gray, gray]))
    return frames


def share_rejected(rows, red_lead, blue_ratio, corner):
    rejected = 0
    for row in rows:
        if not looks_like_fundus(row, red_lead, blue_ratio, corner):
            rejected += 1
    return rejected / len(rows)


def tune(fundus_rows, other_rows):
    """Reject as many non-fundus frames as possible while losing at most 1% of real photos."""
    best = None
    for red_lead in np.arange(1.0, 1.41, 0.05):
        for blue_ratio in np.arange(0.5, 1.01, 0.05):
            for corner in [15, 20, 30, 40, 60, 80, 120]:
                lost = share_rejected(fundus_rows, red_lead, blue_ratio, corner)
                if lost > 0.01:
                    continue
                caught = share_rejected(other_rows, red_lead, blue_ratio, corner)
                if best is None or caught > best["caught"]:
                    best = {
                        "red_lead": round(float(red_lead), 2),
                        "blue_ratio": round(float(blue_ratio), 2),
                        "corner": corner,
                        "caught": caught,
                        "lost": lost,
                    }
    return best


def timing(paths):
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    cv2.setNumThreads(1)

    quality_ms = []
    lesion_ms = []
    for path in paths:
        image = load_512(path)
        start = time.perf_counter()
        quality_features(image)
        middle = time.perf_counter()
        lesion_features(image)
        end = time.perf_counter()
        quality_ms.append((middle - start) * 1000)
        lesion_ms.append((end - middle) * 1000)

    return {
        "quality_median_ms": round(float(np.median(quality_ms)), 1),
        "quality_p95_ms": round(float(np.percentile(quality_ms, 95)), 1),
        "lesion_median_ms": round(float(np.median(lesion_ms)), 1),
    }


def main():
    rng = np.random.default_rng(0)

    eyeq = pd.read_csv("outputs/phase1/eyeq.csv")
    deepdrid = pd.read_csv("outputs/phase1/deepdrid.csv")
    # Reject photos are left out: many are nearly black, and turning those away is fine
    train_eyeq = eyeq[(eyeq["split"] == "train") & (eyeq["quality"] < 2)]
    train_deepdrid = deepdrid[deepdrid["split"] == "train"]
    tuning_fundus = pd.concat([train_eyeq, train_deepdrid])
    # a random 2,000 is enough to place a 1% limit and keeps the search short
    tuning_fundus = tuning_fundus.sample(2000, random_state=0).to_dict("records")

    val_eyeq = eyeq[eyeq["split"] == "val"]
    val_deepdrid = deepdrid[deepdrid["split"] == "val"]
    readable = val_eyeq[val_eyeq["quality"] < 2].to_dict("records") + val_deepdrid.to_dict("records")
    rejects = val_eyeq[val_eyeq["quality"] == 2].to_dict("records")

    all_photos = sorted(Path("data/raw/imagenette").rglob("*.JPEG"))
    order = rng.permutation(len(all_photos))
    picked = [all_photos[index] for index in order[: 2 * N_PHOTOS]]

    other_rows = [quality_features(load_512(path)) for path in picked]
    tune_rows = other_rows[: N_PHOTOS]
    test_rows = other_rows[N_PHOTOS:]

    plain_rows = []
    for frame in plain_frames(rng):
        plain_rows.append(quality_features(resize_long_side(crop_to_retina(frame), 512)))

    best = tune(tuning_fundus, tune_rows)
    red_lead = best["red_lead"]
    blue_ratio = best["blue_ratio"]
    corner = best["corner"]

    results = {
        "tuned": best,
        "held_out_photos_rejected": share_rejected(test_rows, red_lead, blue_ratio, corner),
        "plain_frames_rejected": share_rejected(plain_rows, red_lead, blue_ratio, corner),
        "real_good_usable_and_deepdrid_lost": share_rejected(readable, red_lead, blue_ratio, corner),
        "real_eyeq_reject_flagged": share_rejected(rejects, red_lead, blue_ratio, corner),
    }

    paths = "data/eyepacs/cache512/" + val_eyeq["image"] + ".png"
    results["timing"] = timing(paths.tolist()[:N_TIMED])

    with open("results/phase1_not_fundus_timing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
