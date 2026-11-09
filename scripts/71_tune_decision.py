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
    binary_ece,
    confidence,
    apply_platt,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    photo_costs,
    softmax,
    standard_gate,
)

GRID_POINTS = 80
NEVER = 1.01
SPLITS = 20
VARIANTS = ["reject_only", "shared", "full", "shared_reject_level"]
REJECT_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7]
BOOTSTRAP = 2000
MISS_COSTS = [5, 10, 20, 50]


def threshold_grid(referable):
    quantiles = np.quantile(referable, np.linspace(0.005, 0.995, GRID_POINTS))
    return np.unique(np.append(quantiles, NEVER))


def follow_up(referable, label, good_photo, good_refer):
    """A retake is assumed to give a Good photo of the same eye, decided with the Good threshold."""
    sick = good_photo & (label == 1)
    well = good_photo & (label == 0)
    miss_rate = float((referable[sick] < good_refer).mean())
    over_rate = float((referable[well] >= good_refer).mean())
    return miss_rate, over_rate


def mean_cost(decisions, label, rates, costs):
    return float(photo_costs(decisions, label, rates, costs).mean())


def with_reject_level(data, level):
    """Call a photo Reject when its Reject probability reaches level, otherwise Good or Usable."""
    probabilities = data["quality_probabilities"]
    classes = probabilities[:, :REJECT].argmax(axis=1)
    classes[probabilities[:, REJECT] >= level] = REJECT
    changed = dict(data)
    changed["quality_class"] = classes
    return changed


def tune_with_levels(data, costs, tune, apply):
    """Try each Reject level and keep the cheapest. The level is stored with the thresholds."""
    best = None
    for level in REJECT_LEVELS:
        changed = with_reject_level(data, level)
        rule = tune(changed, costs)
        decisions, rates = apply(changed, rule)
        cost = mean_cost(decisions, changed["label"], rates, costs)
        if best is None or cost < best[0]:
            best = (cost, dict(rule, reject_level=level))
    return best[1]


def tune_standard(data, costs):
    best = None
    for threshold in data["grid"]:
        decisions = standard_gate(data["quality_class"], data["referable"], threshold)
        rates = follow_up(data["referable"], data["label"], data["good_photo"], threshold)
        cost = mean_cost(decisions, data["label"], rates, costs)
        if best is None or cost < best[0]:
            best = (cost, float(threshold))
    return {"refer": best[1]}


def best_for_class(referable, label, rates, costs, grid, clear_options):
    """Cheapest thresholds for the photos of one predicted quality class, summed cost.

    clear_options None means every clear threshold up to the refer one is tried.
    """
    best = None
    for refer in grid:
        options = clear_options
        if options is None:
            options = [c for c in grid if c <= refer]
        for clear in options:
            if clear > refer:
                continue
            decisions = np.full(len(referable), RETAKE)
            decisions[referable >= refer] = REFER
            decisions[referable < clear] = CLEAR
            cost = photo_costs(decisions, label, rates, costs).sum()
            if best is None or cost < best[0]:
                best = (cost, float(refer), float(clear))
    return best


def tune_asymmetric(data, costs, variant="full"):
    """full: four thresholds. shared: Good and the Usable clear share one. reject_only: the standard
    gate plus one threshold that lets a Reject photo refer."""
    quality_class = data["quality_class"]
    referable = data["referable"]
    label = data["label"]
    grid = data["grid"]
    good = quality_class == 0
    usable = quality_class == 1
    reject = quality_class == REJECT

    best = None
    for good_refer in grid:
        if good_refer >= NEVER:
            continue
        rates = follow_up(referable, label, data["good_photo"], good_refer)
        good_decisions = np.where(referable[good] >= good_refer, REFER, CLEAR)
        cost = photo_costs(good_decisions, label[good], rates, costs).sum()
        if variant == "full":
            usable_best = best_for_class(referable[usable], label[usable], rates, costs, grid, None)
        elif variant == "shared":
            usable_best = best_for_class(referable[usable], label[usable], rates, costs, grid, [good_refer])
        else:
            usable_best = best_for_class(referable[usable], label[usable], rates, costs, [good_refer], [good_refer])
        reject_best = best_for_class(referable[reject], label[reject], rates, costs, grid, [0.0])
        total = cost + usable_best[0] + reject_best[0]
        if best is None or total < best[0]:
            thresholds = {
                "good_refer": float(good_refer),
                "usable_refer": usable_best[1],
                "usable_clear": usable_best[2],
                "reject_refer": reject_best[1],
            }
            best = (total, thresholds)
    return best[1]


def tune_both(data, costs, variant):
    if variant == "shared_reject_level":
        def shared(part, part_costs):
            return tune_asymmetric(part, part_costs, "shared")

        standard_rule = tune_with_levels(data, costs, tune_standard, apply_standard)
        thresholds = tune_with_levels(data, costs, shared, apply_asymmetric)
        return standard_rule, thresholds
    return tune_standard(data, costs), tune_asymmetric(data, costs, variant)


def apply_standard(data, thresholds):
    if "reject_level" in thresholds:
        data = with_reject_level(data, thresholds["reject_level"])
    decisions = standard_gate(data["quality_class"], data["referable"], thresholds["refer"])
    rates = follow_up(data["referable"], data["label"], data["good_photo"], thresholds["refer"])
    return decisions, rates


def apply_asymmetric(data, thresholds):
    if "reject_level" in thresholds:
        data = with_reject_level(data, thresholds["reject_level"])
    decisions = asymmetric_rule(data["quality_class"], data["referable"], thresholds)
    rates = follow_up(data["referable"], data["label"], data["good_photo"], thresholds["good_refer"])
    return decisions, rates


def subset(data, rows):
    part = {}
    for key, value in data.items():
        if key == "grid":
            part[key] = value
        else:
            part[key] = value[rows]
    return part


def cross_validated_costs(data, costs, seed, variant="full"):
    """Each patient's photos are decided with thresholds tuned on the other half of the patients."""
    patients = np.unique(data["patient"])
    rng = np.random.default_rng(seed)
    first_half = set(rng.choice(patients, len(patients) // 2, replace=False))
    in_first = np.array([patient in first_half for patient in data["patient"]])

    standard_cost = np.zeros(len(in_first))
    asymmetric_cost = np.zeros(len(in_first))
    decisions_standard = np.zeros(len(in_first), dtype=int)
    decisions_asymmetric = np.zeros(len(in_first), dtype=int)
    for tune_rows in [in_first, ~in_first]:
        tune = subset(data, tune_rows)
        held = subset(data, ~tune_rows)
        standard_rule, asymmetric_thresholds = tune_both(tune, costs, variant)

        decisions, rates = apply_standard(held, standard_rule)
        standard_cost[~tune_rows] = photo_costs(decisions, held["label"], rates, costs)
        decisions_standard[~tune_rows] = decisions
        decisions, rates = apply_asymmetric(held, asymmetric_thresholds)
        asymmetric_cost[~tune_rows] = photo_costs(decisions, held["label"], rates, costs)
        decisions_asymmetric[~tune_rows] = decisions
    return standard_cost, asymmetric_cost, decisions_standard, decisions_asymmetric


def paired_cost_interval(first, second, patients, seed=0):
    """95% interval for (second - first) / first, resampling whole patients."""
    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(BOOTSTRAP):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        changes.append(second[rows].mean() / first[rows].mean() - 1)
    low, high = np.percentile(changes, [2.5, 97.5])
    return float(low), float(high)


def describe(decisions, data, rates):
    label = data["label"]
    sick = label == 1
    now = float((decisions[sick] == REFER).mean())
    retaken_sick = float((decisions[sick] == RETAKE).mean())
    after = now + retaken_sick * (1 - rates[0])
    return {
        "refer_rate": float((decisions == REFER).mean()),
        "retake_rate": float((decisions == RETAKE).mean()),
        "clear_rate": float((decisions == CLEAR).mean()),
        "sensitivity_now": now,
        "sensitivity_after_retake": float(after),
        "specificity_now": float((decisions[~sick] == CLEAR).mean()),
        "missed_now": int(((decisions == CLEAR) & sick).sum()),
    }


def load(predictions, eyeq, name, temperatures):
    quality_logits = predictions[f"{name}__quality"]
    referable_logits = predictions[f"{name}__referable"][:, 0]
    quality_probabilities = softmax(quality_logits / temperatures["quality"])
    data = {
        "quality_class": quality_probabilities.argmax(axis=1),
        "quality_probabilities": quality_probabilities,
        "referable": apply_platt(referable_logits, temperatures["referable"]),
        "label": eyeq["referable"].to_numpy(),
        "good_photo": eyeq["quality"].to_numpy() == 0,
        "true_quality": eyeq["quality"].to_numpy(),
        "patient": eyeq["patient"].to_numpy(),
    }
    data["grid"] = threshold_grid(data["referable"])
    return data


def exeter_point(data):
    readable = data["true_quality"] < 2
    false_positive, true_positive, _ = roc_curve(data["label"][readable], data["referable"][readable])
    specificity_at_80 = float(1 - false_positive[np.searchsorted(true_positive, 0.80)])
    sensitivity_at_95 = float(true_positive[np.searchsorted(false_positive, 0.05, side="right") - 1])
    return {"specificity_at_80_sensitivity": specificity_at_80, "sensitivity_at_95_specificity": sensitivity_at_95}


def calibration(predictions, eyeq):
    quality_logits = predictions["eyeq__quality"]
    referable_logits = predictions["eyeq__referable"][:, 0]
    quality_label = eyeq["quality"].to_numpy()
    referable_label = eyeq["referable"].to_numpy()

    temperatures = {
        "quality": fit_temperature(quality_logits, quality_label, binary=False),
        "referable": fit_platt(referable_logits, referable_label),
    }
    report = {"calibration": temperatures}
    report["referable_temperature_only"] = fit_temperature(referable_logits, referable_label, binary=True)
    nothing = {"quality": 1.0, "referable": {"scale": 1.0, "shift": 0.0}}
    only_temperature = {"scale": 1 / report["referable_temperature_only"], "shift": 0.0}
    for when, scale in [("before", nothing), ("after", temperatures)]:
        probabilities = softmax(quality_logits / scale["quality"])
        top = probabilities.max(axis=1)
        right = probabilities.argmax(axis=1) == quality_label
        report[f"quality_ece_{when}"] = expected_calibration_error(top, right)
        referable = apply_platt(referable_logits, scale["referable"])
        report[f"referable_ece_{when}"] = binary_ece(referable, referable_label)
    referable = apply_platt(referable_logits, only_temperature)
    report["referable_ece_temperature_only"] = binary_ece(referable, referable_label)
    return temperatures, report


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
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
