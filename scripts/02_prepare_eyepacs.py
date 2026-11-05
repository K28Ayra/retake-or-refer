"""Pull only the EyeQ images out of the EyePACS zip, then delete the zip.

    python scripts/02_prepare_eyepacs.py train
    python scripts/02_prepare_eyepacs.py test

Train images are cached as 512 px PNG. Val and test also keep the original JPEG
bytes, because the compression sweep has to start from the real file.
"""
import argparse
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retake.images import crop_to_retina, resize_long_side

ZIP_FOLDER = Path("data/raw/eyepacs")
CACHE = Path("data/eyepacs/cache512")
ORIGINALS = Path("data/eyepacs/original")
LONG_SIDE = 512

zip_handle = None


def open_zip(zip_path):
    global zip_handle
    zip_handle = zipfile.ZipFile(zip_path)


def process_one(job):
    member, keep_original = job
    # the name comes from our label table, never from the archive path
    name = Path(member).name

    raw = zip_handle.read(member)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return name, False

    if keep_original:
        (ORIGINALS / name).write_bytes(raw)

    small = resize_long_side(crop_to_retina(image), LONG_SIDE)
    cv2.imwrite(str(CACHE / (name + ".png")), small)
    return name, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    ORIGINALS.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        table = pd.read_csv("data/splits/eyeq.csv")
    else:
        table = pd.read_csv("data/splits/eyeq_test.csv")

    keep_original = {}
    for _, row in table.iterrows():
        keep_original[row["image"]] = row["split"] != "train"

    zip_path = ZIP_FOLDER / f"{args.split}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    jobs = []
    for member in members:
        name = Path(member).name
        if name in keep_original:
            jobs.append((member, keep_original[name]))
    print(f"{len(jobs)} of {len(table)} labelled images found in the zip", flush=True)

    failed = []
    with Pool(args.workers, initializer=open_zip, initargs=(zip_path,)) as pool:
        results = pool.imap_unordered(process_one, jobs, chunksize=16)
        for done, (name, ok) in enumerate(results, start=1):
            if not ok:
                failed.append(name)
            if done % 1000 == 0:
                print(done, flush=True)

    print("failed:", failed)
    found = len(jobs) - len(failed)
    if found == len(table):
        zip_path.unlink()
        print("all there, zip deleted")
    else:
        print("zip kept, images are missing")


if __name__ == "__main__":
    main()
