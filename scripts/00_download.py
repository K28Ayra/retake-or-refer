"""Download data into data/raw. Run from the repo root.

    python scripts/00_download.py eyeq_labels
    python scripts/00_download.py deepdrid
    python scripts/00_download.py eyepacs_train
    python scripts/00_download.py eyepacs_test
"""
import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from requests.exceptions import ChunkedEncodingError

RAW = Path("data/raw")
TOKEN_FILE = Path(".kaggle/access_token")
KAGGLE_URL = "https://www.kaggle.com/api/v1/competitions/data/download"
COMPETITION = "diabetic-retinopathy-detection"
CHUNK = 1024 * 1024
MAX_ATTEMPTS = 50

BIG_PART = 8388608000
PART_SIZES = {
    "train": [BIG_PART] * 4 + [1434013506],
    "test": [BIG_PART] * 6 + [2957950430],
}
NEEDED_GB = {"train": 60, "test": 80}


def download_eyeq_labels():
    folder = RAW / "eyeq"
    folder.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/HzFu/EyeQ/master/data"
    for split in ["train", "test"]:
        name = f"Label_EyeQ_{split}.csv"
        urllib.request.urlretrieve(f"{base}/{name}", folder / name)


def fetch_github_file(job):
    url, path, size = job
    if path.exists() and path.stat().st_size == size:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(partial, "wb") as out:
        for chunk in response.iter_content(CHUNK):
            out.write(chunk)
    partial.rename(path)
    return size


def download_deepdrid(workers=8):
    """File by file instead of git clone: a dropped connection only loses one image."""
    target = RAW / "deepdrid"
    repo = "deepdrdoc/DeepDRiD"
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
    tree = requests.get(tree_url, timeout=120).json()
    if tree["truncated"]:
        sys.exit("GitHub cut the file list short")

    jobs = []
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            sys.exit(f"Odd path in the repo: {item['path']}")
        url = f"https://raw.githubusercontent.com/{repo}/master/{quote(item['path'])}"
        jobs.append((url, target / relative, item["size"]))
    print(len(jobs), "files", flush=True)

    fetched = 0
    with ThreadPoolExecutor(workers) as pool:
        for done, size in enumerate(pool.map(fetch_github_file, jobs), start=1):
            fetched += size
            if done % 100 == 0:
                print(done, f"{fetched / 1e6:.0f} MB this run", flush=True)
    print("finished", flush=True)


def read_token():
    if not TOKEN_FILE.exists():
        sys.exit(f"Put your Kaggle token in {TOKEN_FILE} first (see README).")
    # Kaggle's copy button also grabs its instructions, the token is the first word
    words = TOKEN_FILE.read_text().split()
    if len(words) == 0:
        sys.exit(f"{TOKEN_FILE} is empty")
    return words[0]


def kaggle_headers(token):
    # Kaggle answers 404 to the default python-requests User-Agent
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": "kaggle-api/v1.7.0",
    }


def remote_size(url, token):
    headers = kaggle_headers(token)
    headers["Range"] = "bytes=0-0"
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    return int(response.headers["Content-Range"].split("/")[1])


def fetch_part(url, path, token):
    """Carries on from whatever is on disk, so a dropped connection costs little."""
    total = remote_size(url, token)

    for attempt in range(MAX_ATTEMPTS):
        have = 0
        if path.exists():
            have = path.stat().st_size
        if have == total:
            return

        headers = kaggle_headers(token)
        headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                # a server that ignores Range sends the whole file again
                mode = "ab" if response.status_code == 206 else "wb"
                with open(path, mode) as out:
                    for chunk in response.iter_content(CHUNK):
                        out.write(chunk)
        except (requests.ConnectionError, requests.Timeout, ChunkedEncodingError):
            print(f"connection dropped at {path.stat().st_size} bytes, retrying", flush=True)
            time.sleep(30)

    sys.exit(f"{path.name}: gave up after {MAX_ATTEMPTS} attempts. Run again to resume.")


def unwrap_onto(wrapped_path, part_name, expected_size, joined):
    """Kaggle sends each part inside its own small zip."""
    with zipfile.ZipFile(wrapped_path) as wrapper:
        info = wrapper.getinfo(part_name)
        if info.file_size != expected_size:
            sys.exit(f"{part_name}: holds {info.file_size} bytes, expected {expected_size}")
        with wrapper.open(info) as inner, open(joined, "ab") as out:
            shutil.copyfileobj(inner, out, CHUNK)


def download_eyepacs(split):
    """Each finished part goes onto one zip and is deleted, so there is only one copy on disk."""
    token = read_token()

    free_gb = shutil.disk_usage(".").free / 1e9
    if free_gb < NEEDED_GB[split]:
        sys.exit(f"Only {free_gb:.0f} GB free, need {NEEDED_GB[split]} GB")

    folder = RAW / "eyepacs"
    folder.mkdir(parents=True, exist_ok=True)
    joined = folder / f"{split}.zip"
    if not joined.exists():
        joined.touch()

    part_start = 0
    for number, size in enumerate(PART_SIZES[split], start=1):
        part_end = part_start + size
        if joined.stat().st_size >= part_end:
            part_start = part_end
            continue

        # an interrupted join leaves half a part at the end
        os.truncate(joined, part_start)

        name = f"{split}.zip.{number:03d}"
        wrapped = folder / (name + ".wrapped")
        url = f"{KAGGLE_URL}/{COMPETITION}/{name}"
        fetch_part(url, wrapped, token)
        unwrap_onto(wrapped, name, size, joined)
        wrapped.unlink()
        print("joined", name, flush=True)
        part_start = part_end


def main():
    parser = argparse.ArgumentParser()
    choices = ["eyeq_labels", "deepdrid", "eyepacs_train", "eyepacs_test"]
    parser.add_argument("what", choices=choices)
    args = parser.parse_args()

    if args.what == "eyeq_labels":
        download_eyeq_labels()
    elif args.what == "deepdrid":
        download_deepdrid()
    elif args.what == "eyepacs_train":
        download_eyepacs("train")
    else:
        download_eyepacs("test")


if __name__ == "__main__":
    main()
