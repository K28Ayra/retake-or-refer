import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from retake.images import pad_to_square

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_image(image, size):
    """BGR photo in, square network-ready array out. Used for training and for the phone."""
    square = pad_to_square(image)
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def to_tensor(rgb):
    scaled = rgb.astype(np.float32) / 255
    normalised = (scaled - MEAN) / STD
    return torch.from_numpy(normalised).permute(2, 0, 1)


def augment(rgb, rng):
    """Mild only. Heavy colour or blur changes would alter the quality grade the label describes."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :]

    size = rgb.shape[0]
    angle = rng.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rgb = cv2.warpAffine(rgb, matrix, (size, size))

    gain = rng.uniform(0.9, 1.1)
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


class PhotoDataset(Dataset):
    def __init__(self, paths, labels, size, train):
        self.paths = list(paths)
        self.labels = list(labels)
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        rgb = prepare_image(image, self.size)
        if self.train:
            # torch gives each worker a new seed every epoch, so numbers do not repeat
            seed = (torch.initial_seed() + index) % (2**32)
            rgb = augment(rgb, np.random.default_rng(seed))
        return to_tensor(rgb), self.labels[index]


def class_weights(labels, n_classes):
    """Inverse frequency, scaled so the average weight is 1."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weights)


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            outputs.append(logits.float().cpu())
    return torch.cat(outputs).numpy()


def train_one_epoch(model, loader, loss_fn, optimiser, scaler, scheduler, device):
    model.train()
    total_loss = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(images)
        seen += len(images)
    return total_loss / seen
