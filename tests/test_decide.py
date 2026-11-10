import numpy as np

from retake.decide import (
    CLEAR,
    REFER,
    REJECT,
    RETAKE,
    asymmetric_rule,
    binary_ece,
    fit_temperature,
    follow_up_rates,
    photo_costs,
    standard_gate,
)

THRESHOLDS = {"good_refer": 0.5, "usable_refer": 0.6, "usable_clear": 0.3, "reject_refer": 0.8}


def test_a_reject_photo_is_never_cleared():
    rng = np.random.default_rng(0)
    referable = rng.random(5000)
    quality = np.full(5000, REJECT)
    for low in [0.0, 0.2, 0.5, 0.99, 1.0]:
        loose = dict(THRESHOLDS, reject_refer=low, usable_clear=1.0, good_refer=1.0)
        decisions = asymmetric_rule(quality, referable, loose)
        assert not (decisions == CLEAR).any()


def test_a_reject_photo_with_strong_disease_is_referred():
    decisions = asymmetric_rule(np.array([REJECT, REJECT]), np.array([0.9, 0.5]), THRESHOLDS)
    assert list(decisions) == [REFER, RETAKE]


def test_usable_photo_between_thresholds_is_retaken():
    decisions = asymmetric_rule(np.array([1, 1, 1]), np.array([0.7, 0.4, 0.1]), THRESHOLDS)
    assert list(decisions) == [REFER, RETAKE, CLEAR]


def test_standard_gate_retakes_every_reject():
    decisions = standard_gate(np.array([REJECT, 0]), np.array([0.99, 0.99]), 0.5)
    assert list(decisions) == [RETAKE, REFER]


def test_costs_follow_the_weights():
    decisions = np.array([CLEAR, CLEAR, REFER, REFER, RETAKE, RETAKE])
    labels = np.array([1, 0, 1, 0, 1, 0])
    cost = photo_costs(decisions, labels, follow_up=(0.1, 0.2))
    assert np.allclose(cost, [10, 0, 0, 1, 0.2 + 1.0, 0.2 + 0.2])


def test_follow_up_rates_use_only_good_photos():
    decisions = np.array([CLEAR, REFER, REFER, CLEAR])
    labels = np.array([1, 1, 0, 0])
    good = np.array([True, True, True, False])
    assert follow_up_rates(decisions, labels, good) == (0.5, 1.0)


def test_temperature_fixes_overconfident_scores():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1.5, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    loud = true_logit * 3
    temperature = fit_temperature(loud, labels, binary=True)
    assert 2.5 < temperature < 3.5
    calibrated = 1 / (1 + np.exp(-loud / temperature))
    assert binary_ece(calibrated, labels) < 0.02


def test_platt_removes_a_shift_that_temperature_cannot():
    from retake.decide import apply_platt, fit_platt

    rng = np.random.default_rng(1)
    true_logit = rng.normal(-1.5, 1.5, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    shifted = true_logit + 2.0
    platt = fit_platt(shifted, labels)
    assert binary_ece(apply_platt(shifted, platt), labels) < 0.02
    temperature = fit_temperature(shifted, labels, binary=True)
    assert binary_ece(1 / (1 + np.exp(-shifted / temperature)), labels) > 0.03


def test_single_threshold_variant_uses_one_number():
    from retake.tune import threshold_grid, tune_asymmetric

    rng = np.random.default_rng(2)
    referable = rng.random(600)
    data = {
        "quality_class": rng.integers(0, 3, 600),
        "referable": referable,
        "label": (rng.random(600) < referable).astype(int),
        "good_photo": rng.random(600) < 0.6,
    }
    data["grid"] = threshold_grid(referable)
    found = tune_asymmetric(data, {"miss": 10.0, "over_refer": 1.0, "retake": 0.2}, "single")
    assert found["good_refer"] == found["usable_refer"] == found["usable_clear"] == found["reject_refer"]
