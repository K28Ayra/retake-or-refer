"""Pick the decision thresholds by expected cost, and compare my rule with the standard gate.

    python scripts/71_tune_decision.py

Reads outputs/phase7/tuning_predictions.npz from 70_predict_tuning.py.
"""
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from retake.decide import (
    CLEAR,
    COSTS,
    REFER,
    REJECT,
    RETAKE,
    asymmetric_rule,
    confidence,
    apply_platt,
    expected_calibration_error,
    photo_costs,
    softmax,
    standard_gate,
)
from retake.tune import (
    SPLITS,
    apply_asymmetric,
    apply_standard,
    calibration,
    cross_validated_costs,
    describe,
    load,
    paired_cost_interval,
    tune_both,
    with_reject_level,
)

VARIANTS = ["reject_only", "shared", "full", "shared_reject_level"]
MISS_COSTS = [5, 10, 20, 50]


def exeter_point(data):
    readable = data["true_quality"] < 2
    false_positive, true_positive, _ = roc_curve(data["label"][readable], data["referable"][readable])
    specificity_at_80 = float(1 - false_positive[np.searchsorted(true_positive, 0.80)])
    sensitivity_at_95 = float(true_positive[np.searchsorted(false_positive, 0.05, side="right") - 1])
    return {"specificity_at_80_sensitivity": specificity_at_80, "sensitivity_at_95_specificity": sensitivity_at_95}


def held_out_photo_costs(data, costs, variant):
    """Average held-out cost of each photo over many patient splits, for both rules."""
    standard_runs = []
    asymmetric_runs = []
    for seed in range(SPLITS):
        standard_cost, asymmetric_cost, _, _ = cross_validated_costs(data, costs, seed, variant)
        standard_runs.append(standard_cost)
        asymmetric_runs.append(asymmetric_cost)
    return np.mean(standard_runs, axis=0), np.mean(asymmetric_runs, axis=0)


def compare(data, costs, variant):
    standard_rule, thresholds = tune_both(data, costs, variant)
    standard_decisions, standard_rates = apply_standard(data, standard_rule)
    asymmetric_decisions, asymmetric_rates = apply_asymmetric(data, thresholds)
    standard_cost = photo_costs(standard_decisions, data["label"], standard_rates, costs)
    asymmetric_cost = photo_costs(asymmetric_decisions, data["label"], asymmetric_rates, costs)
    held_standard, held_asymmetric = held_out_photo_costs(data, costs, variant)
    interval = paired_cost_interval(held_standard, held_asymmetric, data["patient"])
    row = {
        "variant": variant,
        "miss_cost": costs["miss"],
        "standard_cost": float(standard_cost.mean()),
        "asymmetric_cost": float(asymmetric_cost.mean()),
        "in_sample_change": float(asymmetric_cost.mean() / standard_cost.mean() - 1),
        "held_out_standard_cost": float(held_standard.mean()),
        "held_out_asymmetric_cost": float(held_asymmetric.mean()),
        "held_out_change": float(held_asymmetric.mean() / held_standard.mean() - 1),
        "held_out_change_ci_low": interval[0],
        "held_out_change_ci_high": interval[1],
    }
    details = {
        "standard_rule": standard_rule,
        "thresholds": thresholds,
        "standard_decisions": standard_decisions,
        "asymmetric_decisions": asymmetric_decisions,
        "standard_rates": standard_rates,
        "asymmetric_rates": asymmetric_rates,
        "standard_cost": standard_cost,
        "asymmetric_cost": asymmetric_cost,
    }
    return row, details


def pick_variant(rows):
    """Written down before the results: the simplest rule wins unless a bigger one saves more than 0.005."""
    chosen = rows[0]
    for row in rows[1:]:
        if row["held_out_asymmetric_cost"] < chosen["held_out_asymmetric_cost"] - 0.005:
            chosen = row
    return chosen["variant"]


def check_other_photos(predictions, eyeq, temperatures, details, costs, name):
    other = load(predictions, eyeq, name, temperatures)
    decisions, rates = apply_asymmetric(other, details["thresholds"])
    gate, gate_rates = apply_standard(other, details["standard_rule"])
    return {
        "asymmetric_cost": float(photo_costs(decisions, other["label"], rates, costs).mean()),
        "standard_cost": float(photo_costs(gate, other["label"], gate_rates, costs).mean()),
        "same_decision_share": float((decisions == details["asymmetric_decisions"]).mean()),
        **describe(decisions, other, rates),
    }


def deepdrid_check(predictions, temperatures, details, costs):
    """DeepDRiD has no three-class quality label, so the retake follow-up uses EyeQ's rates."""
    deepdrid = pd.read_csv("data/splits/deepdrid_cache.csv", dtype={"patient": str})
    deepdrid = deepdrid[deepdrid["split"] == "val"].reset_index(drop=True)
    probabilities = softmax(predictions["deepdrid__quality"] / temperatures["quality"])
    referable = apply_platt(predictions["deepdrid__referable"][:, 0], temperatures["referable"])
    label = deepdrid["referable"].to_numpy()
    photos = {"quality_probabilities": probabilities, "quality_class": probabilities.argmax(axis=1)}

    def classes(rule):
        if "reject_level" in rule:
            return with_reject_level(photos, rule["reject_level"])["quality_class"]
        return photos["quality_class"]

    quality = classes(details["thresholds"])
    asymmetric = asymmetric_rule(quality, referable, details["thresholds"])
    gate = standard_gate(classes(details["standard_rule"]), referable, details["standard_rule"]["refer"])
    return {
        "photos": int(len(label)),
        "predicted_reject_share": float((quality == REJECT).mean()),
        "asymmetric_cost": float(photo_costs(asymmetric, label, details["asymmetric_rates"], costs).mean()),
        "standard_cost": float(photo_costs(gate, label, details["standard_rates"], costs).mean()),
        "sensitivity_now": float((asymmetric[label == 1] == REFER).mean()),
        "retake_rate": float((asymmetric == RETAKE).mean()),
    }


def confidence_check(data, details):
    """Is a decision said to be 80% sure right about 80% of the time? Retakes have no right answer here."""
    decisions = details["asymmetric_decisions"]
    thresholds = details["thresholds"]
    if "reject_level" in thresholds:
        data = with_reject_level(data, thresholds["reject_level"])
    probability, _ = confidence(decisions, data["quality_probabilities"], data["referable"], thresholds)
    sick = data["label"] == 1
    report = {}
    for name, code, right in [("refer", REFER, sick), ("clear", CLEAR, ~sick)]:
        chosen = decisions == code
        report[name] = {
            "photos": int(chosen.sum()),
            "mean_confidence": float(probability[chosen].mean()),
            "share_right": float(right[chosen].mean()),
            "ece": expected_calibration_error(probability[chosen], right[chosen]),
        }
    return report


def main():
    predictions = np.load("outputs/phase7/tuning_predictions.npz")
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    eyeq = eyeq[eyeq["split"] == "val"].reset_index(drop=True)
    temperatures, calibration_report = calibration(predictions, eyeq)
    print(calibration_report, flush=True)

    data = load(predictions, eyeq, "eyeq", temperatures)
    summary = {"calibration": calibration_report, "exeter": exeter_point(data)}
    summary["true_reject_share"] = float((data["true_quality"] == REJECT).mean())

    variant_rows = []
    for variant in VARIANTS:
        row, _ = compare(data, COSTS, variant)
        variant_rows.append(row)
        print(row, flush=True)
    pd.DataFrame(variant_rows).round(4).to_csv("results/phase7_variants.csv", index=False)
    chosen = pick_variant(variant_rows)
    summary["chosen_variant"] = chosen
    print("chosen", chosen, flush=True)

    cost_rows = []
    for miss in MISS_COSTS:
        costs = dict(COSTS, miss=float(miss))
        row, details = compare(data, costs, chosen)
        cost_rows.append(row)
        print(row, flush=True)
        if miss != COSTS["miss"]:
            continue

        summary["standard_rule"] = details["standard_rule"]
        summary["asymmetric_thresholds"] = details["thresholds"]
        summary["standard"] = describe(details["standard_decisions"], data, details["standard_rates"])
        summary["asymmetric"] = describe(details["asymmetric_decisions"], data, details["asymmetric_rates"])

        # referable photos the standard gate would have sent back for a retake
        sick = data["label"] == 1
        sent_back = details["standard_decisions"] == RETAKE
        referred_now = sent_back & (details["asymmetric_decisions"] == REFER)
        summary["sick_sent_back_by_standard"] = int((sent_back & sick).sum())
        summary["of_those_referred_now"] = float(referred_now[sick & sent_back].sum() / (sick & sent_back).sum())
        summary["early_referrals"] = int(referred_now.sum())
        summary["early_referrals_correct"] = float(sick[referred_now].mean()) if referred_now.any() else None

        table = pd.crosstab(data["true_quality"], details["asymmetric_decisions"])
        table.index = ["true good", "true usable", "true reject"]
        table.columns = [["clear", "refer", "retake"][c] for c in table.columns]
        summary["decisions_by_true_quality"] = table.to_dict(orient="index")
        summary["confidence"] = confidence_check(data, details)
        summary["sent_photos"] = check_other_photos(predictions, eyeq, temperatures, details, costs, "eyeq_sent")
        summary["deepdrid"] = deepdrid_check(predictions, temperatures, details, costs)

    pd.DataFrame(cost_rows).round(4).to_csv("results/phase7_costs.csv", index=False)
    with open("results/phase7_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)

    config = {
        "model": str(predictions["model"]),
        "rule": chosen,
        "quality_temperature": round(temperatures["quality"], 4),
        "referable_platt": {key: round(value, 4) for key, value in temperatures["referable"].items()},
        "costs": COSTS,
        "thresholds": {key: round(value, 4) for key, value in summary["asymmetric_thresholds"].items()},
        "standard_gate_threshold": round(summary["standard_rule"]["refer"], 4),
    }
    with open("configs/decision.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    # the phone page reads the same numbers, and a browser can't read yaml
    with open("phone/decision.json", "w") as f:
        json.dump(config, f, indent=1)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
