import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(labels, predictions):
    return f1_score(labels, predictions, average="macro", zero_division=0)


def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)


def patient_bootstrap(metric, labels, scores, patients, rounds=1000, seed=0):
    """95% interval that resamples whole patients, since two eyes of one person are not independent."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    patients = np.asarray(patients)

    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        # a resample can miss a class by chance, and some metrics cannot handle that
        try:
            values.append(metric(labels[rows], scores[rows]))
        except ValueError:
            continue

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_gap(metric, labels, first, second, patients, rounds=1000, seed=0):
    """95% interval for first minus second on the same photos, resampling whole patients."""
    labels = np.asarray(labels)
    first = np.asarray(first)
    second = np.asarray(second)

    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())

    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(rounds):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        gap = metric(labels[rows], first[rows]) - metric(labels[rows], second[rows])
        gaps.append(gap)

    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(low), float(high)
