"""A few real batches on the EyeQ cache to measure GPU memory and speed before long runs."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "data/hf_cache")

import cv2
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

from retake.images import crop_to_retina, pad_to_square, resize_long_side

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SmokeDataset(Dataset):
    def __init__(self, table, size):
        self.table = table
        self.size = size

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        image = cv2.imread(row["image_path"])
        image = crop_to_retina(image)
        image = resize_long_side(image, self.size)
        image = pad_to_square(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        tensor = (tensor - MEAN) / STD
        return tensor, int(row["referable"])


def pick_gpu():
    best_gpu = 0
    best_free = -1
    for gpu in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(gpu)
        if free > best_free:
            best_gpu = gpu
            best_free = free
    return best_gpu, best_free / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenetv3_large_100")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()

    gpu, free_gb = pick_gpu()
    device = torch.device(f"cuda:{gpu}")
    print(f"gpu {gpu}, {free_gb:.1f} GB free")

    table = pd.read_csv("data/splits/eyeq.csv")
    table = table[table["split"] == "train"].copy()
    table["image_path"] = "data/eyepacs/cache512/" + table["image"] + ".png"
    dataset = SmokeDataset(table, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    seen = 0
    start = None
    for step, (images, labels) in enumerate(loader):
        if step == args.batches:
            break
        # the first batch waits on worker start-up, so the clock starts after it
        if step == 1:
            start = time.time()
            seen = 0

        images = images.to(device)
        labels = labels.float().to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        seen += len(images)

    seconds = time.time() - start
    result = {
        "model": args.model,
        "size": args.size,
        "batch": args.batch,
        "gpu_free_gb_before": round(free_gb, 1),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 2),
        "images_per_second": round(seen / seconds, 1),
        "last_loss": round(loss.item(), 3),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/phase0_smoke.json", "w") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
