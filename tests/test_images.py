import numpy as np

from retake.images import crop_to_retina, pad_to_square, resize_long_side


def make_photo():
    photo = np.zeros((100, 200, 3), dtype=np.uint8)
    photo[20:80, 50:150] = 120
    return photo


def test_crop_removes_border():
    assert crop_to_retina(make_photo()).shape == (60, 100, 3)


def test_crop_leaves_all_black_image_alone():
    black = np.zeros((50, 50, 3), dtype=np.uint8)
    assert crop_to_retina(black).shape == (50, 50, 3)


def test_resize_long_side():
    assert resize_long_side(make_photo(), 100).shape == (50, 100, 3)


def test_pad_makes_square():
    assert pad_to_square(make_photo()).shape == (200, 200, 3)
