"""Faults added on purpose, because no dataset labels why a photo is bad."""
import cv2
import numpy as np

REASONS = ["none", "blur", "uneven_light", "low_contrast"]
BLUR_SIGMA = {1: 2.0, 2: 4.0, 3: 7.0}
SHADE_DEPTH = {1: 0.4, 2: 0.6, 3: 0.8}
CONTRAST_KEEP = {1: 0.6, 2: 0.4, 3: 0.25}


def add_blur(image, strength):
    sigma = BLUR_SIGMA[strength]
    return cv2.GaussianBlur(image, (0, 0), sigma)


def add_uneven_light(image, strength, rng):
    """Shade one side of the photo, like a camera that is not held straight."""
    height, width = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    along = cols / width * np.cos(angle) + rows / height * np.sin(angle)
    along = (along - along.min()) / (along.max() - along.min())

    depth = SHADE_DEPTH[strength]
    gain = 1 - depth * along
    shaded = image.astype(np.float32) * gain[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def add_low_contrast(image, strength):
    """Squeeze the retina's colours toward their mean. The black surround stays black."""
    keep = CONTRAST_KEEP[strength]
    mask = image.max(axis=2) > 15
    output = image.astype(np.float32)
    if mask.sum() == 0:
        return image.copy()

    mean = output[mask].mean(axis=0)
    output[mask] = mean + (output[mask] - mean) * keep
    return np.clip(output, 0, 255).astype(np.uint8)


def degrade(image, reason, strength, rng):
    if reason == "none":
        return image.copy()
    if reason == "blur":
        return add_blur(image, strength)
    if reason == "uneven_light":
        return add_uneven_light(image, strength, rng)
    if reason == "low_contrast":
        return add_low_contrast(image, strength)
    raise ValueError(f"unknown reason {reason}")


def fault_versions(image, seed, count):
    """Fixed random faults for one photo, so every model is scored on the same damage."""
    rng = np.random.default_rng(seed)
    versions = []
    for _ in range(count):
        reason = REASONS[rng.integers(0, len(REASONS))]
        strength = int(rng.integers(1, 4))
        damaged = degrade(image, reason, strength, rng)
        versions.append((damaged, reason, strength))
    return versions
