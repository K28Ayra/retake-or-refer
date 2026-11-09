"""One photo in, the four answers out. Used by both modes of run.py, and needs no torch."""
import json
from pathlib import Path

import cv2
import yaml

from retake.compress import CACHE_SIDE, model_input, shrink
from retake.decide import (
    DECISIONS,
    REJECT,
    apply_platt,
    asymmetric_rule,
    confidence,
    reason_answers,
    softmax,
)
from retake.degrade import REASONS
from retake.enhance import routed
from retake.images import crop_to_retina

QUALITY = ["good", "usable", "reject"]
SIZE = 384


class Screener:
    def __init__(self, model_path, decision_path="configs/decision.yaml", enhance_path="results/phase4_settings.json"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        # turns int8 weights back into floats once at load, instead of on every photo
        options.add_session_config_entry("session.disable_quant_qdq", "1")
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

        with open(decision_path) as f:
            self.decision = yaml.safe_load(f)
        with open(enhance_path) as f:
            self.enhance_settings = json.load(f)

    def scores(self, bgr):
        rgb = model_input(bgr, SIZE)
        names = ["quality", "referable", "reason", "grade"]
        answers = self.session.run(names, {"pixels": rgb[None]})
        return dict(zip(names, answers))

    def screen(self, bgr):
        raw = self.scores(bgr)
        quality_probabilities = softmax(raw["quality"] / self.decision["quality_temperature"])
        referable = apply_platt(raw["referable"][:, 0], self.decision["referable_platt"])
        quality_class = quality_probabilities.argmax(axis=1)
        decisions = asymmetric_rule(quality_class, referable, self.decision["thresholds"])
        sure, why = confidence(decisions, quality_probabilities, referable, self.decision["thresholds"])

        reason = REASONS[reason_answers(raw["reason"])[0]]
        retina = shrink(crop_to_retina(bgr), CACHE_SIDE)
        # for a person to look at only. Phase 4 showed the model must read the original
        enhanced = routed(retina, reason, self.enhance_settings)

        return {
            "quality": QUALITY[quality_class[0]],
            "quality_probability": float(quality_probabilities[0, quality_class[0]]),
            "reason": reason,
            "referable_probability": float(referable[0]),
            "decision": DECISIONS[decisions[0]],
            "confidence": float(sure[0]),
            "retake_because": why[0],
            "never_clear": bool(quality_class[0] == REJECT),
            "enhanced": enhanced,
        }


def photo_list(source):
    """A folder of photos, or a csv with a path column."""
    source = Path(source)
    if source.suffix == ".csv":
        import pandas as pd

        return [Path(path) for path in pd.read_csv(source)["path"]]
    found = []
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        found.extend(source.glob(pattern))
    return sorted(set(found))


def run_dataset(screener, source, out_folder):
    """Writes results.csv and one enhanced picture per photo. Never touches a camera."""
    import pandas as pd

    out_folder = Path(out_folder)
    (out_folder / "enhanced").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in photo_list(source):
        bgr = cv2.imread(str(path))
        if bgr is None:
            rows.append({"photo": str(path), "error": "could not read"})
            continue
        answer = screener.screen(bgr)
        enhanced_path = out_folder / "enhanced" / (path.stem + ".png")
        cv2.imwrite(str(enhanced_path), answer.pop("enhanced"))
        rows.append({"photo": str(path), **answer, "enhanced_path": str(enhanced_path)})
    table = pd.DataFrame(rows)
    table.to_csv(out_folder / "results.csv", index=False)
    return table
