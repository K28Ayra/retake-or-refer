import json
import shutil
import subprocess

import numpy as np
import pytest

from retake.decide import DECISIONS, apply_platt, asymmetric_rule, softmax

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_phone_rule_matches_python(tmp_path):
    with open("phone/decision.json") as f:
        config = json.load(f)
    rng = np.random.default_rng(0)
    quality_logits = rng.normal(0, 3, (2000, 3))
    referable_logits = rng.normal(0, 4, 2000)

    cases = [{"quality": list(q), "referable": float(r)} for q, r in zip(quality_logits, referable_logits)]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    command = [NODE, "tests/js/run_decide.js", str(tmp_path / "cases.json"), "phone/decision.json"]
    subprocess.run(command + [str(tmp_path / "out.json")], check=True)
    phone = json.loads((tmp_path / "out.json").read_text())

    quality = softmax(quality_logits / config["quality_temperature"])
    referable = apply_platt(referable_logits, config["referable_platt"])
    python = asymmetric_rule(quality.argmax(axis=1), referable, config["thresholds"])
    assert [answer["decision"] for answer in phone] == [DECISIONS[d] for d in python]
    assert np.allclose([answer["referable"] for answer in phone], referable)


def test_phone_rule_file_matches_the_config():
    import yaml

    with open("configs/decision.yaml") as f:
        expected = yaml.safe_load(f)
    with open("phone/decision.json") as f:
        assert json.load(f) == expected
