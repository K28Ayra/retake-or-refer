"""Re-score the frozen test decisions with other miss costs. No threshold changes, only the cost of a miss.

    python scripts/92_cost_weights.py

The plan asked for miss costs of 5, 20 and 50 beside the 10 everything was tuned for. Writes
results/final_cost_weights.csv.
"""
import importlib.util

import numpy as np
import pandas as pd
import yaml

from retake.decide import COSTS, photo_costs
from retake.systems import predict
from retake.tune import follow_up

MISS_COSTS = [5, 10, 20, 50]


def load_final_test():
    spec = importlib.util.spec_from_file_location("final_test", "scripts/91_final_test.py")
    final_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final_test)
    return final_test


def cost_change_interval(full_cost, b2_cost, patients):
    rows_by_patient = {}
    for row, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(row)
    groups = list(rows_by_patient.values())
    rng = np.random.default_rng(0)
    changes = []
    for _ in range(2000):
        picks = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[pick] for pick in picks])
        changes.append(full_cost[rows].mean() / b2_cost[rows].mean() - 1)
    return np.percentile(changes, 2.5), np.percentile(changes, 97.5)


def main():
    final_test = load_final_test()
    final_test.check_freeze()
    with open("configs/systems.yaml") as f:
        systems = yaml.safe_load(f)["systems"]

    table = final_test.eyeq_set(False)
    files = final_test.model_files(systems)
    raw = predict(files, table["path"].tolist())
    classical = final_test.classical_quality(table["path"].tolist(), systems["b1"]["files"]["quality"])

    label = table["referable"].to_numpy()
    good = table["quality"].to_numpy() == 0
    patients = table["patient"].to_numpy()

    rows = []
    for miss in MISS_COSTS:
        costs = dict(COSTS, miss=miss)
        per_photo = {}
        for name in final_test.DECIDING:
            system = systems[name]
            quality, referable = final_test.calibrated(system, name, raw, classical)
            decisions = final_test.decide(system, quality, referable)
            rates = follow_up(referable, label, good, final_test.good_threshold(system))
            per_photo[name] = photo_costs(decisions, label, rates, costs)
        low, high = cost_change_interval(per_photo["full"], per_photo["b2"], patients)
        row = {"miss_cost": miss}
        for name, cost in per_photo.items():
            row[name] = float(cost.mean())
        row["full_vs_b2"] = row["full"] / row["b2"] - 1
        row["full_vs_b2_low"] = low
        row["full_vs_b2_high"] = high
        print(row, flush=True)
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv("results/final_cost_weights.csv", index=False)


if __name__ == "__main__":
    main()
