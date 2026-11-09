"""Shrink a photo the way a phone would before sending it, and read it back the way the model expects."""
import cv2
import numpy as np

from retake.images import crop_to_retina, prepare_image, resize_long_side
from retake.quality_features import fundus_mask

CACHE_SIDE = 512
TRAINING = {
    "strong": {"clean_share": 0.2, "sides": [256, 384, 512], "qualities": (10, 95)},
    "mild": {"clean_share": 0.5, "sides": [384, 512], "qualities": (50, 95)},
    "light": {"clean_share": 0.5, "sides": [512], "qualities": (75, 95)},
}


def encode_decode(image, file_format, quality):
    """Returns the photo after a round trip, and how many bytes the file took."""
    if file_format == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif file_format == "jpeg444":
        # normal JPEG keeps colour at half resolution, and small red lesions live in colour
        full_colour = [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality] + full_colour)
    elif file_format == "webp":
        ok, encoded = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        raise ValueError(f"unknown format {file_format}")
    if not ok:
        raise ValueError("could not encode the photo")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), int(encoded.size)


def shrink(image, long_side):
    if max(image.shape[:2]) <= long_side:
        return image
    return resize_long_side(image, long_side)


def retina_only(image):
    """Crop to the retina and make the surround pure black, so no bytes go on sensor noise there."""
    cropped = crop_to_retina(image)
    output = cropped.copy()
    output[~fundus_mask(cropped)] = 0
    return output


def compress_photo(image, file_format, quality, long_side, retina_first=False):
    start = retina_only(image) if retina_first else image
    return encode_decode(shrink(start, long_side), file_format, quality)


def model_input(photo, size):
    """The same steps the cached training photos went through, then the network's own."""
    cropped = crop_to_retina(photo)
    return prepare_image(shrink(cropped, CACHE_SIDE), size)


def random_compress(image, rng, level="strong"):
    """For training. Some photos stay clean so the model still sees what an original looks like."""
    settings = TRAINING[level]
    if rng.random() < settings["clean_share"]:
        return image
    sides = settings["sides"]
    side = sides[rng.integers(0, len(sides))]
    lowest, highest = settings["qualities"]
    quality = int(rng.integers(lowest, highest + 1))
    file_format = "jpeg" if rng.random() < 0.5 else "webp"
    photo, _ = compress_photo(image, file_format, quality, side)
    return photo


def threshold_for_sensitivity(labels, scores, target):
    """The highest score threshold that still reaches the target sensitivity."""
    positive_scores = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    allowed_misses = int(np.floor((1 - target) * len(positive_scores)))
    return float(positive_scores[allowed_misses])


def sensitivity(labels, scores, threshold):
    labels = np.asarray(labels)
    called = np.asarray(scores) >= threshold
    return float(called[labels == 1].mean())


def specificity(labels, scores, threshold):
    labels = np.asarray(labels)
    called = np.asarray(scores) >= threshold
    return float((~called[labels == 0]).mean())
