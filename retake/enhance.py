"""Simple fixes for a bad photo. Each takes and returns a BGR photo and leaves the black surround alone."""
import cv2
import numpy as np

from retake.quality_features import fundus_mask, inner_mask

SMALL_SIZE = 128
MAX_GAIN = 3.0
MIN_GAIN = 0.5
MAX_STRETCH = 4.0
BRIGHT_SIDE = 95


def keep_surround(output, image, mask):
    output[~mask] = image[~mask]
    return output


def clahe(image, clip=2.0):
    mask = fundus_mask(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    tool = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = tool.apply(lab[:, :, 0])
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return keep_surround(output, image, mask)


def graham(image, sigma=10):
    """Subtract the local average, a common trick in DR competitions. It is meant for models, not eyes."""
    mask = fundus_mask(image)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    output = cv2.addWeighted(image, 4, blurred, -4, 128)
    return keep_surround(output, image, mask)


def flat_field(image, sigma=40):
    """Even out slow changes in light. The light map is worked out small, because it is smooth."""
    mask = fundus_mask(image)
    if mask.sum() == 0:
        return image.copy()

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small_gray = cv2.resize(gray * mask, (SMALL_SIZE, SMALL_SIZE), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.float32), (SMALL_SIZE, SMALL_SIZE), interpolation=cv2.INTER_AREA)

    small_sigma = sigma * SMALL_SIZE / max(height, width)
    top = cv2.GaussianBlur(small_gray, (0, 0), small_sigma)
    bottom = cv2.GaussianBlur(small_mask, (0, 0), small_sigma)
    light = top / (bottom + 1e-3)

    # lift the dark side up to the bright side, which is the part least hurt by the fault
    known = cv2.resize(mask.astype(np.uint8), (SMALL_SIZE, SMALL_SIZE), interpolation=cv2.INTER_NEAREST) > 0
    gain = np.percentile(light[known], BRIGHT_SIDE) / (light + 1e-3)
    gain = np.clip(gain, MIN_GAIN, MAX_GAIN)
    gain = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)

    # black stays black under a gain, and a dark shaded corner may have fallen out of the mask
    output = image.astype(np.float32) * gain[:, :, None]
    return np.clip(output, 0, 255).astype(np.uint8)


def tilt_fix(image):
    """Fit a flat tilted plane to the light and divide it out. Leaves the normal dark rim alone."""
    mask = fundus_mask(image)
    if mask.sum() == 0:
        return image.copy()

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small_gray = cv2.resize(gray, (SMALL_SIZE, SMALL_SIZE), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), (SMALL_SIZE, SMALL_SIZE), interpolation=cv2.INTER_NEAREST) > 0

    rows, cols = np.mgrid[0:SMALL_SIZE, 0:SMALL_SIZE].astype(np.float32) / SMALL_SIZE
    inputs = np.stack([cols[small_mask], rows[small_mask], np.ones(small_mask.sum())], axis=1)
    answer = np.linalg.lstsq(inputs, small_gray[small_mask], rcond=None)[0]

    full_rows, full_cols = np.mgrid[0:height, 0:width].astype(np.float32)
    light = answer[0] * full_cols / width + answer[1] * full_rows / height + answer[2]
    light = np.maximum(light, 1.0)
    # lift the dark side up to the bright side, which is the part least hurt by the fault
    gain = np.percentile(light[mask], BRIGHT_SIDE) / light
    gain = np.clip(gain, MIN_GAIN, MAX_STRETCH + 1)

    output = image.astype(np.float32) * gain[:, :, None]
    return np.clip(output, 0, 255).astype(np.uint8)


def unsharp(image, sigma=4.0, amount=1.0):
    """Cannot bring back lost detail. It only makes what is left easier to see."""
    mask = fundus_mask(image)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    output = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return keep_surround(output, image, mask)


def stretch(image, target_std=30.0):
    """Spread the retina's colours away from their mean, up to the spread of a typical good photo."""
    mask = fundus_mask(image)
    if mask.sum() == 0:
        return image.copy()

    # the black rim would count as huge contrast, so measure well inside the retina
    inner = inner_mask(mask)
    if inner.sum() == 0:
        return image.copy()

    values = image.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    now = gray[inner].std()
    gain = np.clip(target_std / (now + 1e-3), 1.0, MAX_STRETCH)

    mean = values[inner].mean(axis=0)
    values[mask] = mean + (values[mask] - mean) * gain
    return np.clip(values, 0, 255).astype(np.uint8)


def routed(image, reason, settings):
    """Use only the fix for the fault that was named. No fault means no change."""
    if reason == "none":
        return image.copy()
    if reason == "blur":
        return unsharp(image, settings["unsharp_sigma"], settings["unsharp_amount"])
    if reason == "uneven_light":
        if settings["light_fix"] == "tilt":
            return tilt_fix(image)
        return flat_field(image, settings["flat_sigma"])
    if reason == "low_contrast":
        if settings["contrast_fix"] == "clahe":
            return clahe(image, settings["clahe_clip"])
        return stretch(image, settings["target_std"])
    raise ValueError(f"unknown reason {reason}")


def apply_method(name, image, reason, settings):
    if name == "none":
        return image.copy()
    if name == "clahe":
        return clahe(image, settings["clahe_clip"])
    if name == "graham":
        return graham(image)
    if name == "flat_field":
        return flat_field(image, settings["flat_sigma"])
    if name == "routed":
        return routed(image, reason, settings)
    if name == "routed_flat":
        changed = dict(settings)
        changed["light_fix"] = "flat"
        return routed(image, reason, changed)
    if name == "unet":
        # only this one needs torch, so the app can run without it
        from retake.unet import unet_fix

        return unet_fix(image, settings["unet_path"])
    raise ValueError(f"unknown method {name}")


def psnr(reference, output, mask):
    difference = reference[mask].astype(np.float32) - output[mask].astype(np.float32)
    error = float((difference**2).mean())
    return 10 * np.log10(255**2 / max(error, 1e-6))


def ssim(reference, output, mask):
    """Standard SSIM on the gray photo, averaged inside the retina only."""
    first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mean1 = cv2.GaussianBlur(first, (0, 0), 1.5)
    mean2 = cv2.GaussianBlur(second, (0, 0), 1.5)
    var1 = cv2.GaussianBlur(first * first, (0, 0), 1.5) - mean1 * mean1
    var2 = cv2.GaussianBlur(second * second, (0, 0), 1.5) - mean2 * mean2
    both = cv2.GaussianBlur(first * second, (0, 0), 1.5) - mean1 * mean2

    top = (2 * mean1 * mean2 + c1) * (2 * both + c2)
    bottom = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2)
    return float((top / bottom)[mask].mean())
