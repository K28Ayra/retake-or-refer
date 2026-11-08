"""Feed models every compression setting, always starting from the original file.

Models share one pass, because making the compressed photos is the slow part.

    python scripts/51_sweep.py plain_s1=outputs/phase3/joint_384_s1_nodeepdridquality_grade
    python scripts/51_sweep.py --encoders jpeg444_retina extra_s1=outputs/phase3/...

With --encoders the answers go into <name>.npz next to any that are already there.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from retake.compress import compress_photo, model_input
from retake.models import JointModel, pick_gpu
from retake.train_utils import to_tensor

SMALL = "mobilenetv3_small_100"
SIZE = 384
HEADS = {"quality": 3, "referable": 1, "reason": 3, "grade": 5}
SIDES = [1024, 768, 512, 384, 256]
QUALITIES = [95, 85, 75, 60, 50, 40, 30, 20, 10]
ENCODERS = {
    "jpeg": ("jpeg", False),
    "webp": ("webp", False),
    "jpeg_retina": ("jpeg", True),
    "webp_retina": ("webp", True),
    "jpeg444": ("jpeg444", False),
    "jpeg444_retina": ("jpeg444", True),
}
DEFAULT_ENCODERS = ["jpeg", "webp", "jpeg_retina", "webp_retina"]
OUT = Path("outputs/phase5/sweeps")


class SweepPhotos(Dataset):
    def __init__(self, paths, setting):
        self.paths = list(paths)
        self.setting = setting

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        photo = cv2.imread(self.paths[index])
        n_bytes = 0
        if self.setting is not None:
            file_format, retina_first, side, quality = self.setting
            photo, n_bytes = compress_photo(photo, file_format, quality, side, retina_first)
        return to_tensor(model_input(photo, SIZE)), n_bytes


def empty_parts():
    return {"referable": [], "quality": [], "reason": []}


def run_setting(models, paths, setting, device):
    """Returns the answers of each model, and the file sizes they all share."""
    loader = DataLoader(SweepPhotos(paths, setting), batch_size=64, num_workers=12, pin_memory=True)
    parts = {}
    for name in models:
        parts[name] = empty_parts()
    sizes = []
    with torch.no_grad():
        for images, n_bytes in loader:
            images = images.to(device, non_blocking=True)
            sizes.append(n_bytes)
            for name, model in models.items():
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                parts[name]["referable"].append(outputs["referable"].float().cpu()[:, 0])
                parts[name]["quality"].append(outputs["quality"].float().cpu().argmax(dim=1))
                parts[name]["reason"].append(outputs["reason"].float().cpu())

    answers = {}
    for name, part in parts.items():
        answers[name] = {
            "referable": torch.cat(part["referable"]).numpy().astype(np.float32),
            "quality": torch.cat(part["quality"]).numpy().astype(np.int8),
            "reason": torch.cat(part["reason"]).numpy().astype(np.float16),
        }
    return answers, torch.cat(sizes).numpy().astype(np.int32)


def setting_name(setting):
    file_format, retina_first, side, quality = setting
    encoder = f"{file_format}_retina" if retina_first else file_format
    return f"{encoder}_{side}_{quality}"


def load_earlier(name):
    saved = OUT / f"{name}.npz"
    if not saved.exists():
        return {}
    return dict(np.load(saved))


def load_models(pairs, device):
    models = {}
    for pair in pairs:
        name, folder = pair.split("=")
        model = JointModel(SMALL, HEADS, pretrained=False)
        state = torch.load(f"{folder}/best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        models[name] = model.to(device).eval()
    return models


def add_answers(results, answers, label, sizes):
    for name, answer in answers.items():
        for key, value in answer.items():
            results[name][f"{label}__{key}"] = value
        results[name][f"{label}__bytes"] = sizes


def save_all(results):
    for name, saved in results.items():
        np.savez_compressed(OUT / f"{name}.npz", **saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+")
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS))
    args = parser.parse_args()
    encoders = args.encoders.split(",")

    OUT.mkdir(parents=True, exist_ok=True)
    gpu, _ = pick_gpu()
    device = torch.device(f"cuda:{gpu}")
    models = load_models(args.pairs, device)

    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    paths = ("data/eyepacs/original/" + val["image"]).tolist()
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    results = {}
    for name in models:
        results[name] = load_earlier(name)
    start = time.time()

    answers, sizes = run_setting(models, paths, None, device)
    add_answers(results, answers, "original", sizes)
    print("original done", round((time.time() - start) / 60, 1), "min", flush=True)

    answers, _ = run_setting(models, deepdrid_val["image_path"].tolist(), None, device)
    for name, answer in answers.items():
        results[name]["deepdrid_original__referable"] = answer["referable"]

    for encoder in encoders:
        file_format, retina_first = ENCODERS[encoder]
        for side in SIDES:
            for quality in QUALITIES:
                setting = (file_format, retina_first, side, quality)
                answers, sizes = run_setting(models, paths, setting, device)
                label = setting_name(setting)
                add_answers(results, answers, label, sizes)
                minutes = (time.time() - start) / 60
                kb = sizes.mean() / 1024
                print(label, f"{kb:.1f} KB, {minutes:.1f} min", flush=True)
        save_all(results)

    for pair in args.pairs:
        name, folder = pair.split("=")
        with open(OUT / f"{name}.json", "w") as f:
            json.dump({"folder": folder, "settings": len(results[name])}, f)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
