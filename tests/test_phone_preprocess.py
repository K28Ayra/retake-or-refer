import shutil
import subprocess

import cv2
import numpy as np
import pytest

from retake.compress import model_input

NODE = shutil.which("node")


def fake_fundus(width, height, seed):
    """A bright noisy disc on black, off centre, like a wide fundus photo."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 6, (height, width, 3)).astype(np.uint8)
    disc = np.zeros((height, width), dtype=np.uint8)
    centre = (width // 2 + 7, height // 2 - 3)
    cv2.circle(disc, centre, int(min(width, height) * 0.45), 1, -1)
    colour = rng.integers(40, 200, (height, width, 3)).astype(np.uint8)
    image[disc == 1] = colour[disc == 1]
    return image


@pytest.mark.skipif(NODE is None, reason="needs node")
@pytest.mark.parametrize("width,height", [(1500, 1000), (700, 523), (900, 900), (2048, 1536)])
def test_phone_preprocessing_matches_python(tmp_path, width, height):
    bgr = fake_fundus(width, height, width)
    rgb = bgr[:, :, ::-1].copy()
    source = tmp_path / "in.rgb"
    target = tmp_path / "out.rgb"
    source.write_bytes(rgb.tobytes())

    script = "tests/js/run_preprocess.js"
    subprocess.run([NODE, script, str(source), str(width), str(height), str(target)], check=True)
    phone = np.frombuffer(target.read_bytes(), dtype=np.uint8).reshape(384, 384, 3)

    python = model_input(bgr, 384)
    assert np.array_equal(phone, python)
