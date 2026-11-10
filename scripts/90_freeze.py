"""Record exactly what the final test will use, before any test photo is read.

    python scripts/90_freeze.py

Refuses if the code has uncommitted changes. 91_final_test.py refuses to run if anything listed here
has changed since.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import yaml

FROZEN = "results/final_freeze.json"
ALWAYS = [
    "configs/default.yaml",
    "configs/decision.yaml",
    "configs/systems.yaml",
    "phone/decision.json",
    "results/phase4_settings.json",
    "data/splits/eyeq_test.csv",
    "data/splits/deepdrid_test.csv",
]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_files():
    with open("configs/systems.yaml") as f:
        systems = yaml.safe_load(f)["systems"]
    paths = list(ALWAYS)
    for system in systems.values():
        for path in system["files"].values():
            if path not in paths:
                paths.append(path)
    return paths


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    changed = git("status", "--porcelain", "--untracked-files=no")
    if changed:
        sys.exit("commit the code first, these files have changes:\n" + changed)

    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "files": {path: file_hash(path) for path in frozen_files()},
    }
    with open(FROZEN, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
