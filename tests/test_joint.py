import cv2
import numpy as np
import pandas as pd
import torch

from retake.degrade import fault_versions
from retake.joint import FaultValDataset, JointDataset, joint_loss, reason_answers
from retake.models import JointModel
from retake.train_utils import PhotoDataset


def make_photo(folder):
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    path = folder / "photo.png"
    cv2.imwrite(str(path), photo)
    return str(path)


def make_table(path, add_fault):
    return pd.DataFrame(
        {
            "path": [path],
            "quality": [-1 if add_fault else 0],
            "referable": [-1 if add_fault else 1],
            "deepdrid_quality": [-1],
            "grade": [-1 if add_fault else 2],
            "reason_known": [1],
            "add_fault": [add_fault],
        }
    )


def test_augmentation_changes_between_calls(tmp_path):
    dataset = PhotoDataset([make_photo(tmp_path)], [0], 32, train=True)
    versions = set()
    for _ in range(10):
        image, _ = dataset[0]
        versions.add(image.numpy().tobytes())
    assert len(versions) > 1


def test_fault_row_marks_one_reason(tmp_path):
    dataset = JointDataset(make_table(make_photo(tmp_path), 1), 32)
    _, targets = dataset[0]
    assert targets["reason"].sum() == 1
    assert targets["quality"] == -1


def test_clean_row_has_no_reason(tmp_path):
    dataset = JointDataset(make_table(make_photo(tmp_path), 0), 32)
    _, targets = dataset[0]
    assert targets["reason"].sum() == 0
    assert targets["referable"] == 1


def test_fault_versions_repeat_with_same_seed():
    photo = np.full((64, 64, 3), 100, dtype=np.uint8)
    first = fault_versions(photo, 5, 3)
    second = fault_versions(photo, 5, 3)
    for one, other in zip(first, second):
        assert one[1:] == other[1:]
        assert (one[0] == other[0]).all()


def test_fault_val_gives_three_versions(tmp_path):
    dataset = FaultValDataset([make_photo(tmp_path)], 32)
    images, reasons, strengths = dataset[0]
    assert images.shape == (3, 3, 32, 32)
    assert reasons.shape == (3,)
    assert strengths.shape == (3,)


def test_reason_answers_uses_none_when_all_low():
    logits = np.array([[-3.0, -2.0, -1.0], [-1.0, 2.0, 1.0]])
    assert reason_answers(logits).tolist() == [0, 2]


def test_model_has_every_head():
    sizes = {"quality": 3, "referable": 1, "reason": 3}
    model = JointModel("mobilenetv3_small_100", sizes, pretrained=False).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
    assert outputs["quality"].shape == (1, 3)
    assert outputs["referable"].shape == (1, 1)


def test_missing_labels_add_no_loss():
    outputs = {"quality": torch.tensor([[5.0, 0.0, 0.0]]), "referable": torch.tensor([[1.0]])}
    targets = {"quality": torch.tensor([-1]), "referable": torch.tensor([-1])}
    assert float(joint_loss(outputs, targets, {}, 1.0)) == 0.0


def test_loss_counts_a_known_label():
    outputs = {"quality": torch.tensor([[0.0, 5.0, 0.0]]), "referable": torch.tensor([[1.0]])}
    targets = {"quality": torch.tensor([0]), "referable": torch.tensor([-1])}
    assert float(joint_loss(outputs, targets, {}, 1.0)) > 1.0
