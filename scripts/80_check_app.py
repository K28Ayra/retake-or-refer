"""Check run.py against the Phase 8 targets.

    python scripts/80_check_app.py

1. Dataset mode on 200 EyeQ validation photos: every output present, same decisions as Phase 7.
2. Replayed clips with synthetic blur: does live mode pick a sharp frame?
Clips are made from DeepDRiD validation photos and deleted afterwards.
"""
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from retake.decide import DECISIONS, apply_platt, asymmetric_rule, softmax
from retake.live import run_live
from retake.pipeline import Screener, run_dataset

MODEL = "outputs/phase6/s1/int8_weights.onnx"
OUT = Path("outputs/phase8")
PHOTOS = 200
CLIPS = 50
FRAMES = 60
FRAME_WIDTH = 1280
SHARP_ENOUGH = 1.0


def phase7_decisions(rows):
    predictions = np.load("outputs/phase7/tuning_predictions.npz")
    with open("configs/decision.yaml") as f:
        config = yaml.safe_load(f)
    quality = softmax(predictions["eyeq__quality"][rows] / config["quality_temperature"]).argmax(axis=1)
    referable = apply_platt(predictions["eyeq__referable"][rows, 0], config["referable_platt"])
    decisions = asymmetric_rule(quality, referable, config["thresholds"])
    return [DECISIONS[decision] for decision in decisions]


def check_dataset_mode(screener):
    eyeq = pd.read_csv("data/splits/eyeq.csv")
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    rows = np.sort(np.random.default_rng(0).choice(len(val), PHOTOS, replace=False))
    listing = OUT / "check_photos.csv"
    pd.DataFrame({"path": "data/eyepacs/original/" + val["image"].iloc[rows]}).to_csv(listing, index=False)

    table = run_dataset(screener, listing, OUT / "dataset_mode")
    wanted = ["quality", "reason", "referable_probability", "decision", "confidence", "enhanced_path"]
    complete = table[wanted].notna().all(axis=1)
    enhanced = table["enhanced_path"].map(lambda path: Path(path).exists())
    same = table["decision"].to_numpy() == np.array(phase7_decisions(rows))
    return {
        "photos": int(len(table)),
        "all_outputs_present": int((complete & enhanced).sum()),
        "same_decision_as_phase7": int(same.sum()),
    }


def make_clip(photo, path, rng):
    """The camera drifts into focus and out again. Only frames near one moment are sharp."""
    height, width = photo.shape[:2]
    scale = FRAME_WIDTH / width
    base = cv2.resize(photo, (FRAME_WIDTH, round(height * scale)), interpolation=cv2.INTER_AREA)
    size = (base.shape[1], base.shape[0])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    sharp_at = int(rng.integers(5, FRAMES - 5))
    blur = []
    for index in range(FRAMES):
        sigma = 0.3 * abs(index - sharp_at) + float(rng.uniform(0, 0.3))
        frame = base
        if sigma > 0.3:
            frame = cv2.GaussianBlur(base, (0, 0), sigma)
        shift = np.float32([[1, 0, rng.uniform(-8, 8)], [0, 1, rng.uniform(-8, 8)]])
        frame = cv2.warpAffine(frame, shift, size)
        noise = rng.normal(0, 3, frame.shape)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        blur.append(sigma)
    writer.release()
    return blur


def check_replay(screener):
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv")
    val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(val), CLIPS, replace=False)
    folder = OUT / "clips"
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for number, pick in enumerate(picks):
        photo = cv2.imread(val["image_path"].iloc[pick])
        path = folder / f"clip_{number}.mp4"
        blur = make_clip(photo, path, rng)
        result = run_live(screener, path, show=False, window=FRAMES)
        chosen = result["frame"]
        rows.append(
            {
                "clip": number,
                "sharpest_frame": int(np.argmin(blur)),
                "chosen_frame": chosen,
                "chosen_blur": blur[chosen],
                "sharp": blur[chosen] <= SHARP_ENOUGH,
                "checks_per_second": result["checks_per_second"],
                "decide_ms": result["decide_ms"],
            }
        )
    shutil.rmtree(folder)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    screener = Screener(MODEL)

    start = time.time()
    summary = {"dataset_mode": check_dataset_mode(screener)}
    print(summary, round(time.time() - start), "s", flush=True)

    clips = check_replay(screener)
    clips.round(3).to_csv("results/phase8_replay.csv", index=False)
    summary["replay"] = {
        "clips": int(len(clips)),
        "sharp_frame_picked": int(clips["sharp"].sum()),
        "exact_sharpest_picked": int((clips["chosen_frame"] == clips["sharpest_frame"]).sum()),
        "checks_per_second_median": float(clips["checks_per_second"].median()),
        "decide_ms_median": float(clips["decide_ms"].median()),
        "decide_ms_max": float(clips["decide_ms"].max()),
    }
    print(json.dumps(summary, indent=1))
    with open("results/phase8_checks.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
