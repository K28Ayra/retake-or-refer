"""512 px PNG cache for DeepDRiD train and val, plus the original photo size."""
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from retake.images import crop_to_retina, resize_long_side

CACHE = Path("data/deepdrid/cache512")


def cache_one(job):
    cv2.setNumThreads(1)
    source, image_id = job
    image = cv2.imread(source)
    height, width = image.shape[:2]
    small = resize_long_side(crop_to_retina(image), 512)
    cv2.imwrite(str(CACHE / f"{image_id}.png"), small)
    return f"{width}x{height}"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv("data/splits/deepdrid.csv")
    jobs = list(zip(table["image_path"], table["image_id"]))

    with Pool(12) as pool:
        sizes = pool.map(cache_one, jobs, chunksize=8)

    table["photo_size"] = sizes
    table["cache_path"] = [str(CACHE / f"{image_id}.png") for image_id in table["image_id"]]
    table.to_csv("data/splits/deepdrid_cache.csv", index=False)
    print(len(table), "cached")


if __name__ == "__main__":
    main()
