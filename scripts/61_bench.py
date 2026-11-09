"""Time the whole phone path on one CPU core: read the original photo, crop and resize, run the model.

    python scripts/61_bench.py outputs/phase6/s1/fp32.onnx outputs/phase6/s1/int8_weights.onnx

Each model runs in its own process pinned to one core, so peak memory belongs to that model alone.
"""
import json
import os
import resource
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

PHOTOS = 500
WARM_UP = 10
CORE = "3"


def measure(model_path):
    import onnxruntime as ort

    from retake.compress import model_input

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # turns int8 weights back into floats once at load, instead of on every photo
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    before_model = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    eyeq = pd.read_csv("data/splits/eyeq.csv")
    val = eyeq[eyeq["split"] == "val"]
    paths = ("data/eyepacs/original/" + val["image"]).tolist()[: PHOTOS + WARM_UP]

    rows = []
    for index, path in enumerate(paths):
        start = time.perf_counter()
        photo = cv2.imread(path)
        read = time.perf_counter()
        rgb = model_input(photo, 384)
        prepared = time.perf_counter()
        session.run(None, {"pixels": rgb[None]})
        done = time.perf_counter()
        if index < WARM_UP:
            continue
        rows.append(
            {
                "read_ms": (read - start) * 1000,
                "prepare_ms": (prepared - read) * 1000,
                "model_ms": (done - prepared) * 1000,
                "total_ms": (done - start) * 1000,
                "pixels": photo.shape[0] * photo.shape[1],
            }
        )

    table = pd.DataFrame(rows)
    summary = {"model": model_path, "mb": os.path.getsize(model_path) / 1e6, "photos": len(table)}
    summary["megapixels_median"] = float(table["pixels"].median() / 1e6)
    for column in ["read_ms", "prepare_ms", "model_ms", "total_ms"]:
        summary[column.replace("_ms", "") + "_median_ms"] = float(table[column].median())
        summary[column.replace("_ms", "") + "_p95_ms"] = float(np.percentile(table[column], 95))
    # Linux reports this in KB
    summary["peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary["memory_before_model_mb"] = before_model
    print(json.dumps(summary))


def main():
    if sys.argv[1] == "--one":
        measure(sys.argv[2])
        return

    results = []
    for model_path in sys.argv[1:]:
        command = ["taskset", "-c", CORE, sys.executable, __file__, "--one", model_path]
        env = dict(os.environ, OMP_NUM_THREADS="1")
        finished = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
        summary = json.loads(finished.stdout.strip().splitlines()[-1])
        print(summary, flush=True)
        results.append(summary)
    pd.DataFrame(results).round(2).to_csv("results/phase6_bench.csv", index=False)


if __name__ == "__main__":
    main()
