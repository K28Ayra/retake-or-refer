"""Running the systems the final test compares on a list of photo files. Shared by scripts 85 and 91."""
import cv2
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader, Dataset

from retake.compress import compress_photo, model_input
from retake.images import crop_to_retina, resize_long_side
from retake.quality_features import lesion_features, quality_features


class Originals(Dataset):
    """Each photo read once from the original file, prepared at every size a system needs."""

    def __init__(self, paths, sizes, sent=None, change=None):
        self.paths = list(paths)
        self.sizes = list(sizes)
        self.sent = sent
        self.change = change

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        photo = cv2.imread(self.paths[index])
        if self.change is not None:
            photo = self.change(index, photo)
        n_bytes = 0
        if self.sent is not None:
            file_format, quality, side, retina_first = self.sent
            photo, n_bytes = compress_photo(photo, file_format, quality, side, retina_first)
        return [model_input(photo, size) for size in self.sizes], n_bytes


def session_for(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def predict(models, paths, sent=None, change=None):
    """models maps a name to (onnx path, input size). Returns name__output arrays, plus file bytes
    when sent is (format, quality, long side, retina first) and each photo is compressed first.
    change(index, photo) alters each photo before anything else, inside the loader workers."""
    sizes = sorted({size for _, size in models.values()})
    sessions = {name: session_for(path) for name, (path, _) in models.items()}
    loader = DataLoader(Originals(paths, sizes, sent, change), batch_size=32, num_workers=12)
    parts = {}
    for batch, n_bytes in loader:
        parts.setdefault("bytes", []).append(n_bytes.numpy())
        by_size = dict(zip(sizes, batch))
        for name, (_, size) in models.items():
            session = sessions[name]
            outputs = [output.name for output in session.get_outputs()]
            answers = session.run(outputs, {"pixels": by_size[size].numpy()})
            for output, value in zip(outputs, answers):
                parts.setdefault(f"{name}__{output}", []).append(value)
    joined = {}
    for key, value in parts.items():
        joined[key] = np.concatenate(value)
    return joined


def classical_row(path):
    """Phase 1's classical features, from the same 512 px retina crop as training."""
    cv2.setNumThreads(1)
    image = resize_long_side(crop_to_retina(cv2.imread(path)), 512)
    row = quality_features(image)
    row.update(lesion_features(image))
    return row


def classical_features(paths, workers=12):
    import pandas as pd
    from multiprocessing import Pool

    with Pool(workers) as pool:
        rows = pool.map(classical_row, list(paths), chunksize=8)
    return pd.DataFrame(rows)

