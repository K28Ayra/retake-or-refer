import cv2
import numpy as np


def crop_to_retina(image, threshold=10):
    """Cut away the black border. Returns the input unchanged if nothing is bright."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = gray > threshold

    rows = np.where(bright.any(axis=1))[0]
    cols = np.where(bright.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image

    top = rows[0]
    bottom = rows[-1] + 1
    left = cols[0]
    right = cols[-1] + 1
    return image[top:bottom, left:right]


def resize_long_side(image, long_side):
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    # INTER_AREA when shrinking so small lesions are not aliased away
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=method)


def pad_to_square(image):
    height, width = image.shape[:2]
    size = max(height, width)
    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def prepare_image(image, size):
    """BGR photo in, square network-ready array out. Used for training and for the phone."""
    square = pad_to_square(image)
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
