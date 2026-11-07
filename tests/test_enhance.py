import cv2
import numpy as np
import torch

from retake.degrade import add_low_contrast, add_uneven_light
from retake.unet import TinyUNet
from retake.enhance import (
    apply_method,
    clahe,
    flat_field,
    graham,
    psnr,
    routed,
    ssim,
    stretch,
    tilt_fix,
    unsharp,
)

SETTINGS = {
    "clahe_clip": 2.0,
    "light_fix": "tilt",
    "flat_sigma": 40,
    "unsharp_sigma": 4.0,
    "unsharp_amount": 1.0,
    "contrast_fix": "stretch",
    "target_std": 30.0,
}


def make_photo():
    size = 256
    photo = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(photo, (128, 128), 110, (40, 90, 170), -1)
    rng = np.random.default_rng(0)
    for _ in range(40):
        place = rng.integers(70, 186, 2).tolist()
        cv2.circle(photo, place, 6, (20, 50, 110), -1)
    return photo


def inside(photo):
    return photo.max(axis=2) > 15


def test_all_fixes_keep_the_surround_black():
    photo = make_photo()
    outputs = [clahe(photo), graham(photo), flat_field(photo), unsharp(photo), stretch(photo)]
    for output in outputs:
        assert output.shape == photo.shape
        assert output[0, 0].sum() == 0


def test_routed_no_fault_changes_nothing():
    photo = make_photo()
    assert (routed(photo, "none", SETTINGS) == photo).all()


def test_flat_field_undoes_shading():
    photo = make_photo()
    shaded = add_uneven_light(photo, 3, np.random.default_rng(0))
    mask = inside(photo)
    assert psnr(photo, flat_field(shaded), mask) > psnr(photo, shaded, mask)


def test_tilt_fix_undoes_shading():
    photo = make_photo()
    shaded = add_uneven_light(photo, 3, np.random.default_rng(0))
    mask = inside(photo)
    assert psnr(photo, tilt_fix(shaded), mask) > psnr(photo, shaded, mask)


def test_stretch_undoes_low_contrast():
    photo = make_photo()
    flat = add_low_contrast(photo, 3)
    mask = inside(photo)
    assert psnr(photo, stretch(flat, 30.0), mask) > psnr(photo, flat, mask)


def test_stretch_never_flattens_a_photo():
    photo = make_photo()
    assert (stretch(photo, 1.0) == photo).all()


def test_ssim_of_same_photo_is_one():
    photo = make_photo()
    assert abs(ssim(photo, photo, inside(photo)) - 1) < 1e-6


def test_apply_method_none_copies():
    photo = make_photo()
    output = apply_method("none", photo, "blur", SETTINGS)
    assert (output == photo).all()
    assert output is not photo


def test_tiny_unet_keeps_the_shape():
    model = TinyUNet().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 3, 64, 64)
