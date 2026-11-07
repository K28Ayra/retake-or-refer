"""Train the tiny U-Net on Good photos with added faults. It keeps the last epoch.

    python scripts/43_train_unet.py --deep
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from retake.degrade import REASONS, degrade, fault_versions
from retake.images import pad_to_square
from retake.models import count_params, pick_gpu
from retake.train_utils import worker_rng
from retake.unet import TinyUNet, photo_to_tensor


class TrainPhotos(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.rng = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.rng is None:
            self.rng = worker_rng()
        clean = pad_to_square(cv2.imread(self.paths[index]))
        reason = REASONS[self.rng.integers(0, len(REASONS))]
        strength = int(self.rng.integers(1, 4))
        damaged = degrade(clean, reason, strength, self.rng)

        if self.rng.random() < 0.5:
            clean = clean[:, ::-1].copy()
            damaged = damaged[:, ::-1].copy()
        return photo_to_tensor(damaged), photo_to_tensor(clean)


class ValPhotos(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        clean = pad_to_square(cv2.imread(self.paths[index]))
        damaged, _, _ = fault_versions(clean, 3000000 + index, 1)[0]
        return photo_to_tensor(damaged), photo_to_tensor(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    gpu, free_gb = pick_gpu()
    if gpu is None:
        sys.exit(f"Only {free_gb:.1f} GB free on the best GPU")
    device = torch.device(f"cuda:{gpu}")

    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    good = eyeq[eyeq["quality"] == 0]
    paths = "data/eyepacs/cache512/" + good["image"] + ".png"
    train_paths = paths[good["split"] == "train"]
    val_paths = paths[good["split"] == "val"]

    train_loader = DataLoader(
        TrainPhotos(train_paths),
        batch_size=args.batch,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(ValPhotos(val_paths), batch_size=args.batch, num_workers=8, persistent_workers=True)

    model = TinyUNet(deep=args.deep).to(device)
    print("params", count_params(model), flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.L1Loss()

    weights_name = "unet_deep.pt" if args.deep else "unet.pt"
    out = Path("outputs/phase4")
    out.mkdir(parents=True, exist_ok=True)
    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for damaged, clean in train_loader:
            damaged = damaged.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(damaged), clean)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total += loss.item()

        model.eval()
        val_total = 0.0
        val_seen = 0
        with torch.no_grad():
            for damaged, clean in val_loader:
                damaged = damaged.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    error = loss_fn(model(damaged), clean)
                val_total += error.item() * len(damaged)
                val_seen += len(damaged)

        minutes = (time.time() - start_time) / 60
        row = {"epoch": epoch, "train_l1": total / len(train_loader), "val_l1": val_total / val_seen}
        history.append(row)
        print(row, f"{minutes:.1f} min", flush=True)
        torch.save(model.state_dict(), out / weights_name)

    with open(f"results/phase4_{weights_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
