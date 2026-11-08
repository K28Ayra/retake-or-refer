import cv2
import numpy as np

from retake.compress import (
    compress_photo,
    encode_decode,
    model_input,
    random_compress,
    retina_only,
    sensitivity,
    shrink,
    specificity,
    threshold_for_sensitivity,
)


def make_photo():
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 8, (400, 600, 3), dtype=np.uint8)
    cv2.circle(photo, (300, 200), 180, (40, 90, 170), -1)
    for _ in range(30):
        place = rng.integers(180, 320, 2).tolist()
        cv2.circle(photo, place, 5, (20, 50, 110), -1)
    return photo


def test_lower_quality_means_a_smaller_file():
    photo = make_photo()
    for file_format in ["jpeg", "webp"]:
        _, big = encode_decode(photo, file_format, 90)
        _, small = encode_decode(photo, file_format, 10)
        assert small < big


def test_round_trip_keeps_the_shape():
    photo = make_photo()
    decoded, _ = compress_photo(photo, "webp", 50, 256)
    assert max(decoded.shape[:2]) == 256


def test_shrink_never_makes_a_photo_bigger():
    photo = make_photo()
    assert shrink(photo, 2000).shape == photo.shape


def test_retina_only_crops_and_cleans_the_surround():
    photo = make_photo()
    output = retina_only(photo)
    assert output.shape[1] < photo.shape[1]
    assert output[0, 0].sum() == 0


def test_retina_only_is_smaller_on_a_noisy_surround():
    photo = make_photo()
    _, plain = compress_photo(photo, "jpeg", 75, 600)
    _, clean = compress_photo(photo, "jpeg", 75, 600, retina_first=True)
    assert clean < plain


def test_model_input_is_square():
    photo = make_photo()
    assert model_input(photo, 64).shape == (64, 64, 3)


def test_random_compress_keeps_a_usable_photo():
    photo = make_photo()
    rng = np.random.default_rng(0)
    for _ in range(10):
        output = random_compress(photo, rng)
        assert output.ndim == 3


def test_threshold_reaches_the_target():
    labels = [1, 1, 1, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.3, 0.1]
    threshold = threshold_for_sensitivity(labels, scores, 0.75)
    assert sensitivity(labels, scores, threshold) >= 0.75
    assert threshold == 0.7
    assert specificity(labels, scores, threshold) == 1.0


def test_mild_compress_leaves_about_half_untouched():
    photo = np.full((64, 64, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)
    same = 0
    for _ in range(200):
        if random_compress(photo, rng, "mild") is photo:
            same += 1
    assert 70 < same < 130


def test_full_colour_jpeg_is_bigger_than_normal_jpeg():
    photo = make_photo()
    _, normal = compress_photo(photo, "jpeg", 90, 600)
    _, full_colour = compress_photo(photo, "jpeg444", 90, 600)
    assert full_colour > normal
