"""Train one single-task model and add a row to results/phase2_runs.csv.

    python scripts/21_train_single.py --task quality --model mobilenetv3_large_100 --size 320
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from retake.metrics import accuracy, macro_f1
from retake.models import count_params, create_model, float_size_mb, pick_gpu
from retake.train_utils import PhotoDataset, class_weights, predict, train_one_epoch

RESULTS = Path("results/phase2_runs.csv")


def load_tables():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    eyeq["path"] = "data/eyepacs/cache512/" + eyeq["image"] + ".png"
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid["path"] = deepdrid["cache_path"]
    return eyeq, deepdrid


def make_loader(table, label_column, size, batch, train):
    labels = table[label_column].astype(int).tolist()
    dataset = PhotoDataset(table["path"], labels, size, train)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=train,
        num_workers=8,
        pin_memory=True,
        drop_last=train,
        persistent_workers=True,
    )


def score_quality(val, logits):
    predicted = logits.argmax(axis=1)
    labels = val["quality"].to_numpy()
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    recalls = matrix.diagonal() / matrix.sum(axis=1)
    return {
        "val_score": macro_f1(labels, predicted),
        "accuracy": accuracy(labels, predicted),
        "macro_f1": macro_f1(labels, predicted),
        "good_recall": recalls[0],
        "usable_recall": recalls[1],
        "reject_recall": recalls[2],
    }


def score_disease(eyeq_val, eyeq_logits, deepdrid_val, deepdrid_logits):
    eyeq_scores = eyeq_logits[:, 0]
    readable = (eyeq_val["quality"] < 2).to_numpy()
    results = {
        "auroc_all": roc_auc_score(eyeq_val["referable"], eyeq_scores),
        "auroc_good_usable": roc_auc_score(eyeq_val["referable"][readable], eyeq_scores[readable]),
        "deepdrid_auroc": roc_auc_score(deepdrid_val["referable"], deepdrid_logits[:, 0]),
    }
    results["val_score"] = results["auroc_good_usable"]

    # DeepDRiD referable rates differ a lot by camera, so the overall number can flatter
    for photo_size in ["1736x1824", "1976x1984"]:
        inside = (deepdrid_val["photo_size"] == photo_size).to_numpy()
        labels = deepdrid_val["referable"][inside]
        results[f"deepdrid_auroc_{photo_size}"] = roc_auc_score(labels, deepdrid_logits[inside, 0])
    return results


def cpu_time_ms(model_name, n_outputs, size):
    """Rough single-thread float time. Phase 6 measures the real exported model."""
    torch.set_num_threads(1)
    model = create_model(model_name, n_outputs, pretrained=False).eval()
    image = torch.zeros(1, 3, size, size)
    times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times[5:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["quality", "disease"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--include-reject", action="store_true")
    parser.add_argument("--eyeq-only", action="store_true")
    args = parser.parse_args()

    run_name = f"{args.task}_{args.model}_{args.size}_s{args.seed}"
    if args.no_balance:
        run_name += "_nobalance"
    if args.include_reject:
        run_name += "_withreject"
    if args.eyeq_only:
        run_name += "_eyeqonly"

    out = Path("outputs/phase2") / run_name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gpu, free_gb = pick_gpu()
    if gpu is None:
        sys.exit(f"Only {free_gb:.1f} GB free on the best GPU")
    device = torch.device(f"cuda:{gpu}")
    print(run_name, f"gpu {gpu}, {free_gb:.1f} GB free", flush=True)

    eyeq, deepdrid = load_tables()
    eyeq_train = eyeq[eyeq["split"] == "train"]
    eyeq_val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    deepdrid_train = deepdrid[deepdrid["split"] == "train"]
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    if args.task == "quality":
        n_outputs = 3
        train_table = eyeq_train
        label_column = "quality"
    else:
        n_outputs = 1
        label_column = "referable"
        # EyeQ Reject photos often cannot be graded, so their disease labels are shaky
        if args.include_reject:
            eyeq_part = eyeq_train
        else:
            eyeq_part = eyeq_train[eyeq_train["quality"] < 2]
        columns = ["path", "referable"]
        train_table = eyeq_part[columns]
        if not args.eyeq_only:
            train_table = pd.concat([train_table, deepdrid_train[columns]])

    train_loader = make_loader(train_table, label_column, args.size, args.batch, True)
    eyeq_loader = make_loader(eyeq_val, label_column, args.size, 64, False)
    deepdrid_loader = make_loader(deepdrid_val, "referable", args.size, 64, False)

    model = create_model(args.model, n_outputs).to(device)

    if args.task == "quality":
        weights = None
        if not args.no_balance:
            weights = class_weights(train_table["quality"], 3).to(device)
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)

        def loss_fn(logits, labels):
            return cross_entropy(logits, labels)
    else:
        positive_weight = None
        if not args.no_balance:
            positives = train_table["referable"].sum()
            negatives = len(train_table) - positives
            positive_weight = torch.tensor([negatives / positives], dtype=torch.float32).to(device)
        binary = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

        def loss_fn(logits, labels):
            return binary(logits.squeeze(1), labels.float())

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda")

    start_time = time.time()
    best = None
    best_epoch = -1
    best_logits = None
    history = []

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, optimiser, scaler, scheduler, device)
        eyeq_logits = predict(model, eyeq_loader, device)

        if args.task == "quality":
            scores = score_quality(eyeq_val, eyeq_logits)
            deepdrid_logits = None
        else:
            deepdrid_logits = predict(model, deepdrid_loader, device)
            scores = score_disease(eyeq_val, eyeq_logits, deepdrid_val, deepdrid_logits)

        minutes = (time.time() - start_time) / 60
        print(f"epoch {epoch} loss {loss:.4f} val {scores['val_score']:.4f} {minutes:.1f} min", flush=True)
        history.append({"epoch": epoch, "loss": loss, "val": scores["val_score"]})

        if best is None or scores["val_score"] > best["val_score"]:
            best = scores
            best_epoch = epoch
            best_logits = (eyeq_logits, deepdrid_logits)
            torch.save(model.state_dict(), out / "best.pt")

    np.save(out / "eyeq_val_logits.npy", best_logits[0])
    if best_logits[1] is not None:
        np.save(out / "deepdrid_val_logits.npy", best_logits[1])
    with open(out / "history.json", "w") as f:
        json.dump(history, f)

    row = {
        "run": run_name,
        "task": args.task,
        "model": args.model,
        "size": args.size,
        "seed": args.seed,
        "balanced": not args.no_balance,
        "include_reject": args.include_reject,
        "best_epoch": best_epoch,
        "minutes": round((time.time() - start_time) / 60, 1),
        "params_m": round(count_params(model) / 1e6, 2),
        "float_mb": round(float_size_mb(model), 1),
        "cpu_ms": round(cpu_time_ms(args.model, n_outputs, args.size), 1),
    }
    for name, value in best.items():
        row[name] = round(float(value), 4)

    new_row = pd.DataFrame([row])
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        old = old[old["run"] != run_name]
        new_row = pd.concat([old, new_row])
    new_row.to_csv(RESULTS, index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
