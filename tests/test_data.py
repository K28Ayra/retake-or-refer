import pandas as pd

from retake.config import load_config
from retake.data import (
    deepdrid_eye_grade,
    eyepacs_patient_id,
    is_referable,
    shared_patients,
    split_eyeq_train,
    subset_eyeq_test,
)


def make_fake_eyeq(n_patients=200):
    rows = []
    for patient in range(n_patients):
        quality = patient % 3
        grade = patient % 5
        for eye in ["left", "right"]:
            rows.append(
                {
                    "image": f"{patient}_{eye}.jpeg",
                    "quality": quality,
                    "DR_grade": grade,
                    "patient": str(patient),
                    "referable": is_referable(grade),
                }
            )
    return pd.DataFrame(rows)


def test_referable_cutoff():
    assert is_referable(1) == 0
    assert is_referable(2) == 1
    assert is_referable(4) == 1


def test_patient_id_from_name():
    assert eyepacs_patient_id("10009_left.jpeg") == "10009"


def test_split_keeps_patients_apart():
    table = split_eyeq_train(make_fake_eyeq(), val_fraction=0.15, seed=1)
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    assert len(val) > 0
    assert len(shared_patients(train, val)) == 0


def test_split_is_repeatable():
    first = split_eyeq_train(make_fake_eyeq(), val_fraction=0.15, seed=1)
    second = split_eyeq_train(make_fake_eyeq(), val_fraction=0.15, seed=1)
    assert first["split"].tolist() == second["split"].tolist()


def test_test_subset_size():
    subset = subset_eyeq_test(make_fake_eyeq(), n_images=60, seed=1)
    assert len(subset) == 60
    assert set(subset["split"]) == {"test"}


def test_deepdrid_uses_filled_eye():
    left_photo = {"left_eye_DR_Level": 3.0, "right_eye_DR_Level": float("nan")}
    right_photo = {"left_eye_DR_Level": float("nan"), "right_eye_DR_Level": 0.0}
    assert deepdrid_eye_grade(left_photo) == 3
    assert deepdrid_eye_grade(right_photo) == 0


def test_realtime_is_off_by_default():
    assert load_config()["realtime"] is False


def test_realtime_can_be_turned_on():
    assert load_config(realtime=True)["realtime"] is True
