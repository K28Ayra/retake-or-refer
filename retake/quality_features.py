import cv2
import numpy as np

# How far inside the retina edge to measure sharpness. The rim itself is a very
# strong edge and would make every photo look sharp.
RIM_MARGIN = 15
GRID_SIZE = 4
MASK_SCALE = 4


def fundus_mask(image):
    """Worked out at quarter size, which is plenty for an outline and much faster."""
    height, width = image.shape[:2]
    small_size = (max(1, width // MASK_SCALE), max(1, height // MASK_SCALE))
    brightest = image.max(axis=2)
    small = cv2.resize(brightest, small_size, interpolation=cv2.INTER_AREA)

    mask = (small > 15).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros((height, width), dtype=bool)

    # label 0 is the background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argmax(areas) + 1
    outline = (labels == largest).astype(np.uint8)
    full = cv2.resize(outline, (width, height), interpolation=cv2.INTER_NEAREST)
    return full.astype(bool)


def inner_mask(mask):
    height, width = mask.shape
    small_size = (max(1, width // MASK_SCALE), max(1, height // MASK_SCALE))
    small = cv2.resize(mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)

    size = 2 * (RIM_MARGIN // MASK_SCALE) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    inner = cv2.erode(small, kernel)
    full = cv2.resize(inner, (width, height), interpolation=cv2.INTER_NEAREST)
    return full.astype(bool)


def corner_light(image):
    """Fundus photos sit on black, so bright corners usually mean something else is in view."""
    height, width = image.shape[:2]
    patch_height = max(1, height // 12)
    patch_width = max(1, width // 12)
    corners = [
        image[:patch_height, :patch_width],
        image[:patch_height, -patch_width:],
        image[-patch_height:, :patch_width],
        image[-patch_height:, -patch_width:],
    ]
    levels = [float(corner.max(axis=2).mean()) for corner in corners]
    return float(np.median(levels))


def high_frequency_ratio(green, mask):
    filled = green.astype(np.float32)
    filled[~mask] = filled[mask].mean()
    small = cv2.resize(filled, (256, 256), interpolation=cv2.INTER_AREA)
    spectrum = np.abs(np.fft.rfft2(small - small.mean())) ** 2

    rows = np.fft.fftfreq(256).reshape(-1, 1)
    cols = np.fft.rfftfreq(256).reshape(1, -1)
    radius = np.sqrt(rows**2 + cols**2)

    high = spectrum[radius > 0.1].sum()
    total = spectrum.sum() + 1e-6
    return float(high / total)


def grid_brightness(green, mask):
    """Mean green in each grid cell, with empty cells as nan."""
    size = (GRID_SIZE, GRID_SIZE)
    weight = mask.astype(np.float32)
    totals = cv2.resize(green.astype(np.float32) * weight, size, interpolation=cv2.INTER_AREA)
    covered = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)

    cells = np.full(size, np.nan, dtype=np.float32)
    enough = covered > 0.25
    cells[enough] = totals[enough] / covered[enough]
    return cells


def entropy(values):
    counts = np.bincount(values, minlength=256).astype(np.float64)
    share = counts / (counts.sum() + 1e-6)
    share = share[share > 0]
    return float(-(share * np.log2(share)).sum())


def empty_features():
    names = feature_names()
    return dict.fromkeys(names, 0.0)


def feature_names():
    names = [
        "cover",
        "corner_light",
        "brightness",
        "dark_share",
        "bright_share",
        "red",
        "green",
        "blue",
        "blue_to_red",
        "green_to_red",
        "contrast",
        "green_contrast",
        "green_range",
        "entropy",
        "laplacian",
        "tenengrad",
        "focus",
        "high_freq",
        "vessel_mean",
        "vessel_p99",
        "light_spread",
        "light_range",
        "centre_to_edge",
    ]
    return names


def quality_features(image):
    """Hand-made features for one BGR photo, expected at about 512 px."""
    mask = fundus_mask(image)
    features = empty_features()
    features["cover"] = float(mask.mean())
    features["corner_light"] = corner_light(image)

    inner = inner_mask(mask)
    if inner.sum() < 1000:
        return features

    blue, green, red = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_values = gray[mask]
    green_values = green[mask]

    brightness = float(gray_values.mean())
    red_mean = float(red[mask].mean())
    green_mean = float(green_values.mean())
    blue_mean = float(blue[mask].mean())
    features["brightness"] = brightness
    features["dark_share"] = float((gray_values < 40).mean())
    features["bright_share"] = float((gray_values > 220).mean())
    features["red"] = red_mean
    features["green"] = green_mean
    features["blue"] = blue_mean
    features["blue_to_red"] = blue_mean / (red_mean + 1)
    features["green_to_red"] = green_mean / (red_mean + 1)

    features["contrast"] = float(gray_values.std()) / (brightness + 1)
    features["green_contrast"] = float(green_values.std()) / (green_mean + 1)
    low, high = np.percentile(green_values, [5, 95])
    features["green_range"] = float(high - low)
    features["entropy"] = entropy(gray_values)

    laplacian = cv2.Laplacian(green, cv2.CV_32F)
    features["laplacian"] = float(laplacian[inner].var())
    sobel_x = cv2.Sobel(green, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(green, cv2.CV_32F, 0, 1)
    features["tenengrad"] = float((sobel_x[inner] ** 2 + sobel_y[inner] ** 2).mean())

    # dividing by the variance keeps a dim but sharp photo from counting as blurred
    green_var = float(green[inner].var()) + 1
    features["focus"] = features["laplacian"] / green_var

    features["high_freq"] = high_frequency_ratio(green, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    vessels = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    vessel_values = vessels[inner]
    features["vessel_mean"] = float(vessel_values.mean())
    features["vessel_p99"] = float(np.percentile(vessel_values, 99))

    cells = grid_brightness(green, mask)
    filled = cells[~np.isnan(cells)]
    if len(filled) > 1:
        cell_mean = float(filled.mean()) + 1
        features["light_spread"] = float(filled.std()) / cell_mean
        features["light_range"] = float(filled.max() - filled.min()) / cell_mean

    centre = cells[1:3, 1:3]
    centre = centre[~np.isnan(centre)]
    if len(centre) > 0:
        features["centre_to_edge"] = float(centre.mean()) / (green_mean + 1)

    return features


def lesion_features(image):
    """Rough counts of small dark and bright spots, for the classical disease baseline."""
    names = ["dark_spots", "dark_area", "bright_spots", "bright_area", "dark_blob"]
    features = dict.fromkeys(names, 0.0)

    mask = fundus_mask(image)
    inner = inner_mask(mask)
    if inner.sum() < 1000:
        return features
    green = image[:, :, 1]
    area = float(inner.sum())

    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dark = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, small)
    bright = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, small)
    big_dark = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, large)

    dark_mask = ((dark > 12) & inner).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    # long thin pieces are vessels, round small ones might be lesions
    round_enough = np.maximum(widths, heights) <= 3 * np.minimum(widths, heights)
    spots = (sizes >= 3) & (sizes <= 120) & round_enough
    features["dark_spots"] = float(spots.sum()) * 1e5 / area
    features["dark_area"] = float(sizes[spots].sum()) / area

    bright_mask = ((bright > 15) & inner).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    spots = (sizes >= 3) & (sizes <= 300)
    features["bright_spots"] = float(spots.sum()) * 1e5 / area
    features["bright_area"] = float(sizes[spots].sum()) / area

    features["dark_blob"] = float((big_dark[inner] > 20).mean())
    return features


def looks_like_fundus(features, min_red_lead=1.0, max_blue_to_red=0.85, max_corner=15):
    """A retina is a red disc on black. Few other things in a room are."""
    if features["cover"] < 0.3:
        return False
    if features["corner_light"] > max_corner:
        return False
    red_lead = 1 / (features["green_to_red"] + 1e-6)
    if red_lead < min_red_lead:
        return False
    return features["blue_to_red"] <= max_blue_to_red
