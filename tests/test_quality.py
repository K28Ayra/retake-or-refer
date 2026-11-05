import numpy as np

from retake.degrade import add_blur, add_low_contrast, add_uneven_light
from retake.metrics import accuracy, patient_bootstrap
from retake.quality_features import (
    feature_names,
    fundus_mask,
    lesion_features,
    looks_like_fundus,
    quality_features,
)


def fake_fundus(seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    rows, cols = np.mgrid[0:512, 0:512]
    inside = (rows - 256) ** 2 + (cols - 256) ** 2 < 240**2

    noise = rng.normal(0, 25, (512, 512))
    image[..., 0][inside] = 20
    image[..., 1][inside] = np.clip(80 + noise[inside], 0, 255)
    image[..., 2][inside] = 170
    return image


def test_mask_finds_the_disc():
    mask = fundus_mask(fake_fundus())
    assert 0.6 < mask.mean() < 0.75


def test_all_features_present():
    features = quality_features(fake_fundus())
    assert set(features) == set(feature_names())


def test_black_frame_does_not_crash():
    black = np.zeros((512, 512, 3), dtype=np.uint8)
    assert quality_features(black)["cover"] == 0
    assert lesion_features(black)["dark_spots"] == 0


def test_blur_lowers_focus():
    image = fake_fundus()
    sharp = quality_features(image)["laplacian"]
    blurred = quality_features(add_blur(image, 2))["laplacian"]
    assert blurred < sharp


def test_shading_raises_light_spread():
    image = fake_fundus()
    rng = np.random.default_rng(0)
    even = quality_features(image)["light_spread"]
    shaded = quality_features(add_uneven_light(image, 3, rng))["light_spread"]
    assert shaded > even


def test_low_contrast_lowers_contrast():
    image = fake_fundus()
    before = quality_features(image)["green_contrast"]
    after = quality_features(add_low_contrast(image, 3))["green_contrast"]
    assert after < before


def test_fundus_check():
    assert looks_like_fundus(quality_features(fake_fundus()))
    gray = np.full((512, 512, 3), 128, dtype=np.uint8)
    assert not looks_like_fundus(quality_features(gray))


def test_bootstrap_interval_contains_score():
    labels = np.array([0, 1] * 50)
    predicted = labels.copy()
    predicted[:10] = 1 - predicted[:10]
    patients = np.arange(100)
    low, high = patient_bootstrap(accuracy, labels, predicted, patients, rounds=100)
    assert low <= accuracy(labels, predicted) <= high
