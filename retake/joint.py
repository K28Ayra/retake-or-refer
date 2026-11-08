"""Data and loss for the joint model. A label of -1 means this photo does not have that label."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from retake.compress import random_compress
from retake.degrade import REASONS, degrade, fault_versions
from retake.train_utils import augment, prepare_image, to_tensor, worker_rng

FAULTS = REASONS[1:]
LABELS = ["quality", "referable", "deepdrid_quality", "grade"]
VERSIONS_PER_PHOTO = 3
VAL_FAULT_SEED = 1000000


class JointDataset(Dataset):
    """Rows with add_fault set get a random fault and teach only the reason head."""

    def __init__(self, table, size, compress=None, two_views=False):
        self.compress = compress
        self.two_views = two_views
        self.paths = table["path"].tolist()
        self.labels = {}
        for name in LABELS:
            self.labels[name] = table[name].astype(int).tolist()
        self.reason_known = table["reason_known"].astype(int).tolist()
        self.add_fault = table["add_fault"].astype(int).tolist()
        self.size = size
        self.rng = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.rng is None:
            self.rng = worker_rng()

        image = cv2.imread(self.paths[index])
        reason = np.zeros(len(FAULTS), dtype=np.float32)
        if self.add_fault[index]:
            fault = int(self.rng.integers(0, len(FAULTS)))
            strength = int(self.rng.integers(1, 4))
            image = degrade(image, FAULTS[fault], strength, self.rng)
            reason[fault] = 1

        # both views must get the same flips and rotation, or they could not be compared
        view_seed = int(self.rng.integers(0, 2**31))

        def view(photo):
            return to_tensor(augment(prepare_image(photo, self.size), np.random.default_rng(view_seed)))

        seen = image
        if self.compress:
            seen = random_compress(image, self.rng, self.compress)

        targets = {"reason": reason, "reason_known": self.reason_known[index]}
        for name in LABELS:
            targets[name] = self.labels[name][index]
        targets["clean"] = view(image) if self.two_views else 0
        return view(seen), targets


class FaultValDataset(Dataset):
    """The same fixed faulty versions that Phase 1 scored, three for each Good photo."""

    def __init__(self, paths, size):
        self.paths = list(paths)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        seed = VAL_FAULT_SEED + index
        images = []
        reasons = []
        strengths = []
        for damaged, reason, strength in fault_versions(image, seed, VERSIONS_PER_PHOTO):
            images.append(to_tensor(prepare_image(damaged, self.size)))
            reasons.append(REASONS.index(reason))
            strengths.append(strength)
        return torch.stack(images), torch.tensor(reasons), torch.tensor(strengths)


def reason_answers(reason_logits):
    """0 means no fault. Otherwise the most likely fault, as an index into REASONS."""
    best = reason_logits.argmax(axis=1)
    top = reason_logits.max(axis=1)
    return np.where(top > 0, best + 1, 0)


def joint_loss(outputs, targets, class_weights, reason_weight):
    total = 0.0
    for head, logits in outputs.items():
        if head == "reason":
            known = targets["reason_known"] == 1
            if known.any():
                part = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[known], targets["reason"][known]
                )
                total = total + reason_weight * part
        elif logits.shape[1] == 1:
            known = targets[head] >= 0
            if known.any():
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[known, 0],
                    targets[head][known].float(),
                    pos_weight=class_weights.get(head),
                )
        else:
            known = targets[head] >= 0
            if known.any():
                total = total + torch.nn.functional.cross_entropy(
                    logits[known], targets[head][known], weight=class_weights.get(head)
                )
    return total


def consistency_loss(outputs, clean_outputs):
    """A compressed photo should get the same answer as its clean version."""
    disease = torch.nn.functional.mse_loss(outputs["referable"], clean_outputs["referable"])
    clean_quality = torch.nn.functional.softmax(clean_outputs["quality"], dim=1)
    quality = torch.nn.functional.cross_entropy(outputs["quality"], clean_quality)
    entropy = -(clean_quality * torch.log(clean_quality + 1e-8)).sum(dim=1).mean()
    return disease + quality - entropy
