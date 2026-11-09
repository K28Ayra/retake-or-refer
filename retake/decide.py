"""Turn model scores into refer, retake or clear, with a confidence, and score that by expected cost."""
import numpy as np
from scipy.optimize import minimize_scalar

CLEAR = 0
REFER = 1
RETAKE = 2
DECISIONS = ["clear", "refer", "retake"]
GOOD = 0
USABLE = 1
REJECT = 2
COSTS = {"miss": 10.0, "over_refer": 1.0, "retake": 0.2}


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    powers = np.exp(shifted)
    return powers / powers.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels, binary):
    """One number that divides the logits so the probabilities match how often the model is right."""
    labels = np.asarray(labels)

    def loss(log_temperature):
        temperature = np.exp(log_temperature)
        if binary:
            probability = np.clip(sigmoid(np.asarray(logits) / temperature), 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
        probabilities = np.clip(softmax(np.asarray(logits) / temperature), 1e-7, 1)
        return -np.mean(np.log(probabilities[np.arange(len(labels)), labels]))

    found = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    return float(np.exp(found.x))


def fit_platt(logits, labels):
    """A scale and a shift for a yes/no output. Training with a class weight shifts the scores,
    and a temperature alone can only stretch them."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6).fit(np.asarray(logits).reshape(-1, 1), np.asarray(labels))
    return {"scale": float(model.coef_[0, 0]), "shift": float(model.intercept_[0])}


def apply_platt(logits, platt):
    return sigmoid(platt["scale"] * np.asarray(logits) + platt["shift"])


def expected_calibration_error(confidence, correct, bins=10):
    """Average gap between how sure the model says it is and how often it is right, by confidence bin."""
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            gap = abs(confidence[inside].mean() - correct[inside].mean())
            total += inside.mean() * gap
    return float(total)


def binary_ece(probability, labels):
    """For a yes/no output: does a probability of 0.3 mean yes 30% of the time?"""
    return expected_calibration_error(np.asarray(probability), np.asarray(labels) == 1)


def standard_gate(quality_class, referable, threshold):
    """The usual pipeline: a bad photo is always retaken, the rest get one disease threshold."""
    decisions = np.where(referable >= threshold, REFER, CLEAR)
    return np.where(quality_class == REJECT, RETAKE, decisions)


def asymmetric_rule(quality_class, referable, thresholds):
    """A bad photo can refer, but it can never clear.

    thresholds holds good_refer, usable_refer, usable_clear and reject_refer. A usable photo between
    its two thresholds is retaken, because the photo is not good enough to settle a borderline score.
    """
    decisions = np.full(len(referable), RETAKE)

    good = quality_class == GOOD
    decisions[good & (referable >= thresholds["good_refer"])] = REFER
    decisions[good & (referable < thresholds["good_refer"])] = CLEAR

    usable = quality_class == USABLE
    decisions[usable & (referable >= thresholds["usable_refer"])] = REFER
    decisions[usable & (referable < thresholds["usable_clear"])] = CLEAR

    reject = quality_class == REJECT
    decisions[reject & (referable >= thresholds["reject_refer"])] = REFER
    return decisions


def follow_up_rates(decisions, referable_label, good_photo):
    """What a retake leads to. I assume the second photo is a typical Good photo of the same eye."""
    good_sick = good_photo & (referable_label == 1)
    good_well = good_photo & (referable_label == 0)
    miss_rate = float((decisions[good_sick] == CLEAR).mean()) if good_sick.any() else 0.0
    over_rate = float((decisions[good_well] == REFER).mean()) if good_well.any() else 0.0
    return miss_rate, over_rate


def photo_costs(decisions, referable_label, follow_up, costs=COSTS):
    miss_rate, over_rate = follow_up
    sick = referable_label == 1
    cost = np.zeros(len(decisions))
    cost[(decisions == CLEAR) & sick] = costs["miss"]
    cost[(decisions == REFER) & ~sick] = costs["over_refer"]

    retake = decisions == RETAKE
    after_sick = costs["retake"] + miss_rate * costs["miss"]
    after_well = costs["retake"] + over_rate * costs["over_refer"]
    cost[retake & sick] = after_sick
    cost[retake & ~sick] = after_well
    return cost


def confidence(decisions, quality_probabilities, referable, thresholds):
    """How sure the rule is: the calibrated probability behind the answer, and why a retake was asked for."""
    probability = np.zeros(len(decisions))
    probability[decisions == REFER] = referable[decisions == REFER]
    probability[decisions == CLEAR] = 1 - referable[decisions == CLEAR]
    retake = decisions == RETAKE
    probability[retake] = quality_probabilities[retake, REJECT]

    reasons = np.full(len(decisions), "", dtype=object)
    quality_class = quality_probabilities.argmax(axis=1)
    reasons[retake & (quality_class == REJECT)] = "poor photo"
    reasons[retake & (quality_class != REJECT)] = "borderline disease score on a usable photo"
    return probability, reasons
