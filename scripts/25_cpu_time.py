"""Time each backbone on one CPU thread. Run it with nothing else training.

This is a server CPU in float32, so it only ranks the models. Phase 6 times the real phone export.
"""
import time

import numpy as np
import pandas as pd
import torch

from retake.models import create_model

CHOICES = [
    ("mobilenetv3_small_100", 224),
    ("mobilenetv3_small_100", 320),
    ("mobilenetv3_small_100", 384),
    ("mobilenetv3_small_100", 448),
    ("mobilenetv3_large_100", 320),
    ("mobilenetv3_large_100", 384),
    ("efficientnet_lite0", 320),
    ("resnet18", 320),
]
REPEATS = 60
WARMUP = 10


def time_model(name, size):
    model = create_model(name, 3, pretrained=False).eval()
    image = torch.zeros(1, 3, size, size)
    times = []
    with torch.no_grad():
        for _ in range(REPEATS):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000)
    kept = times[WARMUP:]
    return {
        "model": name,
        "size": size,
        "median_ms": round(float(np.median(kept)), 1),
        "p95_ms": round(float(np.percentile(kept, 95)), 1),
    }


def main():
    torch.set_num_threads(1)
    rows = []
    for name, size in CHOICES:
        rows.append(time_model(name, size))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/phase2_cpu_time.csv", index=False)


if __name__ == "__main__":
    main()
