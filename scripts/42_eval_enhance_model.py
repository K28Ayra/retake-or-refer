"""Does a fix change what the joint model says about real photos? The model was never retrained."""
import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from retake.degrade import REASONS
from retake.enhance import apply_method
from retake.joint import reason_answers
from retake.metrics import accuracy, macro_f1, paired_gap
from retake.models import JointModel, pick_gpu
from retake.train_utils import prepare_image, to_tensor

SMALL = "mobilenetv3_small_100"
SIZE = 384
HEADS = {"quality": 3, "referable": 1, "reason": 3, "grade": 5}
METHODS = ["none", "clahe", "graham", "flat_field", "routed", "routed_flat", "unet"]
SEEDS = [0, 1, 2]

with open("results/phase4_settings.json") as f:
    SETTINGS = json.load(f)
SETTINGS["unet_path"] = "outputs/phase4/unet_deep.pt"


class FixedPhotos(Dataset):
    def __init__(self, paths, reasons, method):
        self.paths = list(paths)
        self.reasons = list(reasons)
        self.method = method

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        image = cv2.imread(self.paths[index])
        fixed = apply_method(self.method, image, self.reasons[index], SETTINGS)
        return to_tensor(prepare_image(fixed, SIZE))


def predict(model, paths, reasons, method, device):
    dataset = FixedPhotos(paths, reasons, method)
    loader = DataLoader(dataset, batch_size=64, num_workers=10, pin_memory=True)
    collected = {"quality": [], "referable": []}
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            for head in collected:
                collected[head].append(outputs[head].float().cpu())
    return {head: torch.cat(parts).numpy() for head, parts in collected.items()}


def main():
    gpu, _ = pick_gpu()
    device = torch.device(f"cuda:{gpu}")

    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    paths = ("data/eyepacs/cache512/" + val["image"] + ".png").tolist()
    quality = val["quality"].to_numpy()
    referable = val["referable"].to_numpy()
    patients = val["patient"].to_numpy()
    readable = quality < 2
    reject = quality == 2

    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)
    deepdrid_paths = deepdrid_val["cache_path"].tolist()

    results = {}
    for seed in SEEDS:
        folder = f"outputs/phase3/joint_384_s{seed}_nodeepdridquality_grade"
        model = JointModel(SMALL, HEADS, pretrained=False)
        model.load_state_dict(torch.load(f"{folder}/best.pt", map_location="cpu", weights_only=True))
        model = model.to(device).eval()

        eyeq_saved = np.load(f"{folder}/eyeq_val.npz")
        deepdrid_saved = np.load(f"{folder}/deepdrid_val.npz")
        reasons = [REASONS[answer] for answer in reason_answers(eyeq_saved["reason"])]
        deepdrid_reasons = [REASONS[answer] for answer in reason_answers(deepdrid_saved["reason"])]

        baseline = predict(model, paths, reasons, "none", device)
        base_scores = baseline["referable"][:, 0]
        base_grade = baseline["quality"].argmax(axis=1)

        results[seed] = {"share_of_photos_routed": round(float(np.mean([r != "none" for r in reasons])), 4)}
        for method in METHODS:
            output = predict(model, paths, reasons, method, device)
            scores = output["referable"][:, 0]
            grades = output["quality"].argmax(axis=1)

            drive = predict(model, deepdrid_paths, deepdrid_reasons, method, device)
            drive_scores = drive["referable"][:, 0]

            called_reject = reject & (base_grade == 2)
            moved_up = float((grades[called_reject] < 2).mean())
            row = {
                "macro_f1": macro_f1(quality, grades),
                "accuracy": accuracy(quality, grades),
                "auroc_good_usable": roc_auc_score(referable[readable], scores[readable]),
                "auroc_all": roc_auc_score(referable, scores),
                "auroc_reject": roc_auc_score(referable[reject], scores[reject]),
                "deepdrid_auroc": roc_auc_score(deepdrid_val["referable"], drive_scores),
                "reject_moved_up": moved_up,
            }
            if method != "none":
                row["auroc_minus_none_ci"] = paired_gap(
                    roc_auc_score,
                    referable[readable],
                    scores[readable],
                    base_scores[readable],
                    patients[readable],
                )
            for name, value in list(row.items()):
                if isinstance(value, float):
                    row[name] = round(value, 4)
            results[seed][method] = row
            print(seed, method, row, flush=True)

    with open("results/phase4_model.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
