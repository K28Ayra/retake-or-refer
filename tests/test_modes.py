import sys

import cv2
import numpy as np
import pytest
import yaml

import run
from retake.config import load_config


class FakeScreener:
    """Stands in for the real model, which isn't in the repo."""

    def __init__(self, model_path):
        pass

    def screen(self, bgr):
        return {
            "quality": "good",
            "quality_probability": 0.9,
            "reason": "none",
            "referable_probability": 0.1,
            "decision": "clear",
            "confidence": 0.9,
            "retake_because": "",
            "never_clear": False,
            "enhanced": bgr,
        }


def fundus(blur=0.0, size=400):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 2 - 10, (40, 90, 160), -1)
    rng = np.random.default_rng(0)
    for _ in range(60):
        x, y = rng.integers(size // 4, 3 * size // 4, 2)
        cv2.circle(image, (int(x), int(y)), 3, (20, 30, 60), -1)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return image


def test_command_line_overrides_the_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"realtime": False, "camera_source": 0}))
    assert load_config(path)["realtime"] is False
    assert load_config(path, realtime=True)["realtime"] is True


def test_dataset_mode_never_opens_a_camera(tmp_path, monkeypatch):
    def no_camera(*args, **kwargs):
        raise AssertionError("a camera was opened")

    monkeypatch.setattr(cv2, "VideoCapture", no_camera)
    monkeypatch.setattr(run, "Screener", FakeScreener)
    sys.modules.pop("retake.live", None)

    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "eye.png"), fundus())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"realtime": False, "camera_source": 0}))

    table = run.main(["--config", str(config), "--input", str(photos), "--out", str(tmp_path / "out")])
    assert list(table["decision"]) == ["clear"]
    assert "retake.live" not in sys.modules
    assert (tmp_path / "out" / "enhanced" / "eye.png").exists()


def test_realtime_flag_needs_on_or_off():
    with pytest.raises(SystemExit):
        run.main(["--realtime", "maybe"])


def test_sharper_frame_scores_higher():
    from retake.live import frame_check

    sharp = frame_check(fundus())
    blurred = frame_check(fundus(blur=3))
    assert sharp["message"] == "ok"
    assert sharp["sharpness"] > blurred["sharpness"]


def test_black_frame_is_not_a_retina():
    from retake.live import frame_check

    assert frame_check(np.zeros((300, 400, 3), dtype=np.uint8))["message"] == "no retina in view"


def test_best_frame_prefers_usable_over_sharp_but_dark():
    from retake.live import BestFrame

    window = BestFrame(3)
    window.add(0, "dark", {"message": "too dark", "sharpness": 900.0})
    window.add(1, "soft", {"message": "ok", "sharpness": 10.0})
    window.add(2, "crisp", {"message": "ok", "sharpness": 50.0})
    assert window.best()[0] == 2


def test_drawing_the_overlay_keeps_the_frame_size():
    from retake.live import draw, frame_check

    frame = fundus()
    answer = FakeScreener(None).screen(frame)
    shown = draw(frame, frame_check(frame), 100.0, answer)
    assert shown.shape == frame.shape
    assert not np.array_equal(shown, frame)
