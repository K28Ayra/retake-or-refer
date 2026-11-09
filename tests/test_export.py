import numpy as np
import onnxruntime as ort
import torch

from retake.export import export_onnx, int8_weights, run_onnx
from retake.models import JointModel
from retake.train_utils import to_tensor

HEADS = {"quality": 3, "referable": 1, "reason": 3, "grade": 5}


def small_model():
    torch.manual_seed(0)
    return JointModel("mobilenetv3_small_100", HEADS, pretrained=False).eval()


def test_onnx_matches_pytorch_on_pixels(tmp_path):
    model = small_model()
    path = tmp_path / "model.onnx"
    export_onnx(model, path, 64)
    rgb = np.random.default_rng(0).integers(0, 256, (64, 64, 3)).astype(np.uint8)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = run_onnx(session, [rgb])
    with torch.no_grad():
        torch_out = model(to_tensor(rgb)[None])
    for name in HEADS:
        assert np.abs(onnx_out[name] - torch_out[name].numpy()).max() < 1e-3


def test_int8_weights_are_smaller_and_close(tmp_path):
    model = small_model()
    float_path = tmp_path / "fp32.onnx"
    int8_path = tmp_path / "int8.onnx"
    export_onnx(model, float_path, 64)
    int8_weights(float_path, int8_path)
    assert int8_path.stat().st_size < float_path.stat().st_size * 0.4

    rgb = np.random.default_rng(1).integers(0, 256, (64, 64, 3)).astype(np.uint8)
    full = run_onnx(ort.InferenceSession(str(float_path), providers=["CPUExecutionProvider"]), [rgb])
    small = run_onnx(ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]), [rgb])
    gap = np.abs(full["quality"] - small["quality"]).max()
    assert gap < 0.1 * np.abs(full["quality"]).max() + 0.05
