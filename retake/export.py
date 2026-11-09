"""Turn a trained joint model into one ONNX file that takes plain pixels, so a phone needs nothing else."""
from pathlib import Path

import numpy as np
import torch

from retake.train_utils import MEAN, STD

OUTPUTS = ["quality", "referable", "reason", "grade"]


class PixelModel(torch.nn.Module):
    """Takes a uint8 RGB image of shape (1, size, size, 3) and returns the raw scores of each head."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pixels):
        image = pixels.permute(0, 3, 1, 2).float() / 255
        image = (image - self.mean) / self.std
        outputs = self.model(image)
        answers = []
        for name in OUTPUTS:
            answers.append(outputs[name])
        return tuple(answers)


def export_onnx(model, path, size):
    wrapped = PixelModel(model).eval()
    example = torch.zeros((1, size, size, 3), dtype=torch.uint8)
    torch.onnx.export(
        wrapped,
        example,
        path,
        input_names=["pixels"],
        output_names=OUTPUTS,
        opset_version=17,
        dynamic_axes={"pixels": {0: "batch"}},
    )


def run_onnx(session, rgb_images):
    """rgb_images is a list of (size, size, 3) uint8 arrays. Returns one array per head."""
    batch = np.stack(rgb_images).astype(np.uint8)
    answers = session.run(OUTPUTS, {"pixels": batch})
    named = {}
    for name, value in zip(OUTPUTS, answers):
        named[name] = value
    return named


class CalibrationPhotos:
    """Feeds photos to the int8 calibration one at a time, the way ONNX Runtime asks for them."""

    def __init__(self, rgb_images):
        self.images = list(rgb_images)
        self.position = 0

    def get_next(self):
        if self.position >= len(self.images):
            return None
        image = self.images[self.position]
        self.position += 1
        return {"pixels": image[None].astype(np.uint8)}

    def rewind(self):
        self.position = 0


def percentile_ranges(model_path, rgb_images, percentile, work_folder):
    """Activation ranges that ignore the rarest extreme values.

    Plain min-max lets one odd photo stretch a range so far that ordinary values round to zero.
    """
    from onnxruntime.quantization import CalibrationMethod, create_calibrator

    calibrator = create_calibrator(
        str(model_path),
        augmented_model_path=str(Path(work_folder) / "augmented.onnx"),
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"percentile": percentile, "symmetric": False},
    )
    calibrator.collect_data(CalibrationPhotos(rgb_images))
    found = calibrator.compute_data()

    overrides = {}
    for name, tensor in found.data.items():
        lowest, highest = tensor.range_value
        overrides[name] = [{"rmin": np.float32(min(lowest, 0)), "rmax": np.float32(max(highest, 0))}]
    return overrides


def int8_weights(in_path, out_path):
    """Store Conv and Gemm weights as int8, one scale per output channel, and keep activations in float.

    Full int8 failed here: the first layers have channels whose ranges differ by hundreds of times,
    and one int8 scale per activation tensor can't cover them. The weights quantise fine.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(in_path))
    graph = model.graph
    weights = {}
    for initializer in graph.initializer:
        weights[initializer.name] = initializer

    new_nodes = []
    for node in graph.node:
        if node.op_type in ["Conv", "Gemm"] and node.input[1] in weights:
            name = node.input[1]
            values = numpy_helper.to_array(weights[name])
            per_channel = np.abs(values).reshape(values.shape[0], -1).max(axis=1)
            scale = np.maximum(per_channel, 1e-8) / 127
            shape = [-1] + [1] * (values.ndim - 1)
            stored = np.clip(np.round(values / scale.reshape(shape)), -127, 127).astype(np.int8)

            graph.initializer.remove(weights[name])
            graph.initializer.append(numpy_helper.from_array(stored, name + "_int8"))
            graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), name + "_scale"))
            zero = np.zeros(values.shape[0], dtype=np.int8)
            graph.initializer.append(numpy_helper.from_array(zero, name + "_zero"))
            inputs = [name + "_int8", name + "_scale", name + "_zero"]
            new_nodes.append(helper.make_node("DequantizeLinear", inputs, [name + "_float"], axis=0))
            node.input[1] = name + "_float"
        new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
