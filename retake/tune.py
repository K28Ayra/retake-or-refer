"""Choosing decision thresholds by expected cost, shared by Phase 7 and the Phase 9 baselines."""
import numpy as np

from retake.decide import (
    CLEAR,
    REFER,
    REJECT,
    RETAKE,
    apply_platt,
    asymmetric_rule,
    binary_ece,
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
REJECT_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7]
BOOTSTRAP = 2000


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
    gate plus one threshold that lets a Reject photo refer. single: one threshold for every quality,
    still never clearing a Reject."""
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
        reject_grid = [good_refer] if variant == "single" else grid
        reject_best = best_for_class(referable[reject], label[reject], rates, costs, reject_grid, [0.0])
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


def build_data(quality_probabilities, referable, table):
    """The same dict as load, from probabilities that are already calibrated."""
    data = {
        "quality_class": quality_probabilities.argmax(axis=1),
        "quality_probabilities": quality_probabilities,
        "referable": referable,
        "label": table["referable"].to_numpy(),
        "good_photo": table["quality"].to_numpy() == 0,
        "true_quality": table["quality"].to_numpy(),
        "patient": table["patient"].to_numpy(),
    }
    data["grid"] = threshold_grid(referable)
    return data
