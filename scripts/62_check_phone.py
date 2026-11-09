"""Does the phone's JavaScript preprocessing give the model the same input as Python, on real photos?

    python scripts/62_check_phone.py outputs/phase6/s1/int8_weights.onnx
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from retake.compress import model_input
from retake.export import run_onnx

PHOTOS = 50


def phone_input(bgr, folder):
    rgb = bgr[:, :, ::-1].copy()
    source = Path(folder) / "in.rgb"
    target = Path(folder) / "out.rgb"
    source.write_bytes(rgb.tobytes())
    height, width = rgb.shape[:2]
    command = [shutil.which("node"), "tests/js/run_preprocess.js", str(source), str(width), str(height), str(target)]
    subprocess.run(command, check=True)
    return np.frombuffer(target.read_bytes(), dtype=np.uint8).reshape(384, 384, 3)


def main():
    session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
    eyeq = pd.read_csv("data/splits/eyeq.csv")
    val = eyeq[eyeq["split"] == "val"].sample(PHOTOS, random_state=0)

    pixel_gaps = []
    output_gaps = []
    with tempfile.TemporaryDirectory() as folder:
        for image in val["image"]:
            bgr = cv2.imread("data/eyepacs/original/" + image)
            python = model_input(bgr, 384)
            phone = phone_input(bgr, folder)
            pixel_gaps.append(np.abs(python.astype(int) - phone.astype(int)))

            python_out = run_onnx(session, [python])
            phone_out = run_onnx(session, [phone])
            worst = 0.0
            for name in python_out:
                worst = max(worst, float(np.abs(python_out[name] - phone_out[name]).max()))
            output_gaps.append(worst)

    pixels = np.stack(pixel_gaps)
    summary = {
        "photos": PHOTOS,
        "pixel_gap_max": int(pixels.max()),
        "pixels_off_share": float((pixels > 0).mean()),
        "output_gap_max": max(output_gaps),
        "output_gap_median": float(np.median(output_gaps)),
    }
    print(json.dumps(summary))
    with open("results/phase6_phone_check.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
