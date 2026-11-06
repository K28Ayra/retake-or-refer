import cv2
import numpy as np
import torch

from retake.models import count_params, create_model
from retake.train_utils import PhotoDataset, augment, class_weights, prepare_image, to_tensor


def test_prepare_image_is_square():
    photo = np.zeros((300, 500, 3), dtype=np.uint8)
    assert prepare_image(photo, 224).shape == (224, 224, 3)


def test_prepare_swaps_to_rgb():
    photo = np.zeros((100, 100, 3), dtype=np.uint8)
    photo[:, :, 2] = 200
    assert prepare_image(photo, 50)[0, 0, 0] == 200


def test_tensor_shape():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    assert to_tensor(rgb).shape == (3, 64, 64)


def test_augment_keeps_shape():
    rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert augment(rgb, rng).shape == (64, 64, 3)


def test_rare_class_gets_more_weight():
    weights = class_weights([0, 0, 0, 0, 1], 2)
    assert weights[1] > weights[0]


def test_dataset_reads_a_file(tmp_path):
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), np.full((60, 80, 3), 90, dtype=np.uint8))
    dataset = PhotoDataset([str(path)], [1], 32, train=True)
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert label == 1


def test_small_model_runs():
    model = create_model("mobilenetv3_small_100", 3, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 3)
    assert count_params(model) > 0
