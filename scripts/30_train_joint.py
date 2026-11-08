"""Train one joint model and add a row to results/phase3_runs.csv.

    python scripts/30_train_joint.py --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from retake.joint import (
    FAULTS,
    FaultValDataset,
    JointDataset,
    consistency_loss,
    joint_loss,
    reason_answers,
)
from retake.metrics import macro_f1
from retake.models import JointModel, count_params, float_size_mb, pick_gpu
from retake.train_utils import PhotoDataset, class_weights

RESULTS = Path("results/phase3_runs.csv")


def load_tables():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    eyeq["path"] = "data/eyepacs/cache512/" + eyeq["image"] + ".png"
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid["path"] = deepdrid["cache_path"]
    return eyeq, deepdrid


def build_train_table(eyeq, deepdrid, fault_share):
    eyeq_rows = pd.DataFrame(
        {
            "path": eyeq["path"],
            "quality": eyeq["quality"],
            "referable": eyeq["referable"],
            "deepdrid_quality": -1,
            "grade": eyeq["DR_grade"],
            "reason_known": (eyeq["quality"] == 0).astype(int),
            "add_fault": 0,
        }
    )
    deepdrid_rows = pd.DataFrame(
        {
            "path": deepdrid["path"],
            "quality": -1,
            "referable": deepdrid["referable"],
            "deepdrid_quality": deepdrid["Overall quality"],
            "grade": deepdrid["DR_grade"],
            "reason_known": 0,
            "add_fault": 0,
        }
    )
    good = eyeq[eyeq["quality"] == 0]
    extra_rows = good.sample(frac=fault_share, random_state=0)
    fault_rows = pd.DataFrame(
        {
            "path": extra_rows["path"],
            "quality": -1,
            "referable": -1,
            "deepdrid_quality": -1,
            "grade": -1,
            "reason_known": 1,
            "add_fault": 1,
        }
    )
    return pd.concat([eyeq_rows, deepdrid_rows, fault_rows], ignore_index=True)


def predict_photos(model, loader, device):
    model.eval()
    collected = {}
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            for head, logits in outputs.items():
                collected.setdefault(head, []).append(logits.float().cpu())
    results = {}
    for head, parts in collected.items():
        results[head] = torch.cat(parts).numpy()
    return results


def predict_faults(model, loader, device):
    model.eval()
    logits = []
    reasons = []
    strengths = []
    with torch.no_grad():
        for images, reason, strength in loader:
            batch, versions = images.shape[:2]
            flat = images.reshape(batch * versions, *images.shape[2:]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(flat)
            logits.append(outputs["reason"].float().cpu())
            reasons.append(reason.reshape(-1))
            strengths.append(strength.reshape(-1))
    return torch.cat(logits).numpy(), torch.cat(reasons).numpy(), torch.cat(strengths).numpy()


def score_quality(labels, logits):
    predicted = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    recalls = matrix.diagonal() / matrix.sum(axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": macro_f1(labels, predicted),
        "good_recall": recalls[0],
        "usable_recall": recalls[1],
        "reject_recall": recalls[2],
    }


def score_disease(eyeq_val, eyeq_scores, deepdrid_val, deepdrid_scores):
    readable = (eyeq_val["quality"] < 2).to_numpy()
    results = {
        "auroc_all": roc_auc_score(eyeq_val["referable"], eyeq_scores),
        "auroc_good_usable": roc_auc_score(eyeq_val["referable"][readable], eyeq_scores[readable]),
        "deepdrid_auroc": roc_auc_score(deepdrid_val["referable"], deepdrid_scores),
    }
    for photo_size in ["1736x1824", "1976x1984"]:
        inside = (deepdrid_val["photo_size"] == photo_size).to_numpy()
        labels = deepdrid_val["referable"][inside]
        results[f"deepdrid_auroc_{photo_size}"] = roc_auc_score(labels, deepdrid_scores[inside])
    return results


def score_reasons(logits, reasons, strengths, deepdrid_val, deepdrid_reason):
    answers = reason_answers(logits)
    results = {"reason_f1": macro_f1(reasons, answers)}
    for strength in [1, 2, 3]:
        keep = (strengths == strength) | (reasons == 0)
        results[f"reason_f1_strength{strength}"] = macro_f1(reasons[keep], answers[keep])
    blur_scores = deepdrid_reason[:, FAULTS.index("blur")]
    results["blur_clarity_spearman"] = spearmanr(blur_scores, deepdrid_val["Clarity"]).statistic
    return results


def cpu_time_ms(model, size):
    torch.set_num_threads(1)
    model = model.to("cpu").eval()
    image = torch.zeros(1, 3, size, size)
    times = []
    with torch.no_grad():
        for _ in range(30):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times[5:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenetv3_small_100")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reason", action="store_true")
    parser.add_argument("--no-deepdrid-quality", action="store_true")
    parser.add_argument("--grade-head", action="store_true")
    parser.add_argument("--reason-weight", type=float, default=1.0)
    parser.add_argument("--fault-share", type=float, default=1.0)
    parser.add_argument("--compress", choices=["strong", "mild", "light"])
    parser.add_argument("--consistency", type=float, default=0.0)
    parser.add_argument("--init", help="start from this best.pt instead of ImageNet weights")
    parser.add_argument("--teacher", help="a frozen best.pt whose clean answers the model must match")
    parser.add_argument("--distill", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    run_name = f"joint_{args.size}_s{args.seed}"
    if args.no_reason:
        run_name += "_noreason"
    if args.no_deepdrid_quality:
        run_name += "_nodeepdridquality"
    if args.grade_head:
        run_name += "_grade"
    run_name += args.tag

    out = Path("outputs/phase3") / run_name
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

    fault_share = 0.0 if args.no_reason else args.fault_share
    table = build_train_table(eyeq_train, deepdrid_train, fault_share)
    use_consistency = args.consistency > 0
    use_teacher = args.distill > 0
    two_views = use_consistency or use_teacher
    train_data = JointDataset(table, args.size, args.compress, two_views)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    def plain_loader(paths):
        dataset = PhotoDataset(paths, [0] * len(paths), args.size, False)
        return DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True, persistent_workers=True)

    eyeq_loader = plain_loader(eyeq_val["path"])
    deepdrid_loader = plain_loader(deepdrid_val["path"])

    good_val = eyeq_val[eyeq_val["quality"] == 0]
    fault_loader = DataLoader(
        FaultValDataset(good_val["path"], args.size),
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    head_sizes = {"quality": 3, "referable": 1}
    if not args.no_reason:
        head_sizes["reason"] = len(FAULTS)
    if not args.no_deepdrid_quality:
        head_sizes["deepdrid_quality"] = 1
    if args.grade_head:
        head_sizes["grade"] = 5
    model = JointModel(args.model, head_sizes, pretrained=args.init is None)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    model = model.to(device)

    teacher = None
    if use_teacher:
        teacher = JointModel(args.model, head_sizes, pretrained=False)
        teacher.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True))
        teacher = teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

    quality_rows = table[table["quality"] >= 0]
    referable_rows = table[table["referable"] >= 0]
    positives = referable_rows["referable"].sum()
    negatives = len(referable_rows) - positives
    weights = {
        "quality": class_weights(quality_rows["quality"], 3).to(device),
        "referable": torch.tensor([negatives / positives], dtype=torch.float32).to(device),
    }

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda")

    start_time = time.time()
    best = None
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            moved = {}
            for name, value in targets.items():
                moved[name] = value.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = joint_loss(outputs, moved, weights, args.reason_weight)
                if use_consistency:
                    with torch.no_grad():
                        clean_outputs = model(moved["clean"])
                    loss = loss + args.consistency * consistency_loss(outputs, clean_outputs)
                if use_teacher:
                    with torch.no_grad():
                        teacher_outputs = teacher(moved["clean"])
                    loss = loss + args.distill * consistency_loss(outputs, teacher_outputs)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * len(images)
            seen += len(images)

        eyeq_out = predict_photos(model, eyeq_loader, device)
        deepdrid_out = predict_photos(model, deepdrid_loader, device)

        scores = score_quality(eyeq_val["quality"].to_numpy(), eyeq_out["quality"])
        scores.update(
            score_disease(eyeq_val, eyeq_out["referable"][:, 0], deepdrid_val, deepdrid_out["referable"][:, 0])
        )
        if "deepdrid_quality" in head_sizes:
            scores["deepdrid_quality_auroc"] = roc_auc_score(
                deepdrid_val["Overall quality"], deepdrid_out["deepdrid_quality"][:, 0]
            )
        parts = [scores["macro_f1"], scores["auroc_good_usable"]]

        fault_logits = None
        if "reason" in head_sizes:
            fault_logits, reasons, strengths = predict_faults(model, fault_loader, device)
            scores.update(score_reasons(fault_logits, reasons, strengths, deepdrid_val, deepdrid_out["reason"]))
            parts.append(scores["reason_f1"])
        scores["val_score"] = float(np.mean(parts))

        minutes = (time.time() - start_time) / 60
        reason_text = round(scores.get("reason_f1", 0), 4)
        print(
            f"epoch {epoch} loss {total_loss / seen:.4f} quality {scores['macro_f1']:.4f} "
            f"auroc {scores['auroc_good_usable']:.4f} reason {reason_text} {minutes:.1f} min",
            flush=True,
        )
        history.append({"epoch": epoch, "loss": total_loss / seen, **{k: float(v) for k, v in scores.items()}})

        if best is None or scores["val_score"] > best["val_score"]:
            best = scores
            best_epoch = epoch
            torch.save(model.state_dict(), out / "best.pt")
            np.savez(out / "eyeq_val.npz", **eyeq_out)
            np.savez(out / "deepdrid_val.npz", **deepdrid_out)
            if fault_logits is not None:
                np.savez(out / "fault_val.npz", logits=fault_logits, reasons=reasons, strengths=strengths)

    with open(out / "history.json", "w") as f:
        json.dump(history, f)

    row = {
        "run": run_name,
        "seed": args.seed,
        "heads": "+".join(head_sizes),
        "reason_weight": args.reason_weight,
        "fault_share": fault_share,
        "compress": args.compress,
        "consistency": args.consistency,
        "init": args.init or "",
        "distill": args.distill,
        "best_epoch": best_epoch,
        "minutes": round((time.time() - start_time) / 60, 1),
        "params_m": round(count_params(model) / 1e6, 2),
        "float_mb": round(float_size_mb(model), 1),
        "cpu_ms": round(cpu_time_ms(model, args.size), 1),
    }
    for name, value in best.items():
        row[name] = round(float(value), 4)
    row["last_macro_f1"] = round(history[-1]["macro_f1"], 4)
    row["last_auroc_good_usable"] = round(history[-1]["auroc_good_usable"], 4)

    new_row = pd.DataFrame([row])
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        old = old[old["run"] != run_name]
        new_row = pd.concat([old, new_row])
    new_row.to_csv(RESULTS, index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
