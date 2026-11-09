"""Run the model that ships on the tuning photos, the way the phone would, and save its raw scores.

    python scripts/70_predict_tuning.py outputs/phase6/s1/int8_weights.onnx

EyeQ validation photos are read from the original files, once as they are and once after the
compression Phase 5 picked, so the decision rule can be checked on both.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from retake.compress import compress_photo, model_input
from retake.export import OUTPUTS

SIZE = 384
SENT = ("jpeg444", 95, 384, True)
OUT = Path("outputs/phase7")


class Photos(Dataset):
    def __init__(self, paths, compress):
        self.paths = list(paths)
        self.compress = compress

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        photo = cv2.imread(self.paths[index])
        if self.compress:
            file_format, quality, side, retina_first = SENT
            photo, _ = compress_photo(photo, file_format, quality, side, retina_first)
        return model_input(photo, SIZE)


def predict(session, paths, compress):
    loader = DataLoader(Photos(paths, compress), batch_size=32, num_workers=12)
    parts = {}
    for batch in loader:
        answers = session.run(OUTPUTS, {"pixels": batch.numpy()})
        for name, value in zip(OUTPUTS, answers):
            parts.setdefault(name, []).append(value)
    joined = {}
    for name, value in parts.items():
        joined[name] = np.concatenate(value)
    return joined


def main():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    session = ort.InferenceSession(sys.argv[1], options, providers=["CPUExecutionProvider"])

    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    eyeq_paths = ("data/eyepacs/original/" + val["image"]).tolist()
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid_val = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)

    saved = {"model": np.array(sys.argv[1])}
    for name, paths, compress in [
        ("eyeq", eyeq_paths, False),
        ("eyeq_sent", eyeq_paths, True),
        ("deepdrid", deepdrid_val["image_path"].tolist(), False),
    ]:
        answers = predict(session, paths, compress)
        for head, value in answers.items():
            saved[f"{name}__{head}"] = value
        print(name, "done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "tuning_predictions.npz", **saved)


if __name__ == "__main__":
    main()
