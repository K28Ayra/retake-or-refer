"""Run a list of trainings one after another. Safe to restart: finished runs are skipped.

    python scripts/31_run_phase3.py variants
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SINGLE_RESULTS = Path("results/phase2_runs.csv")
JOINT_RESULTS = Path("results/phase3_runs.csv")
RETRIES = 6
WAIT_SECONDS = 600
SMALL = "mobilenetv3_small_100"


def fixed_single_runs():
    """The chosen Phase 2 models again, with the augmentation bug fixed, so the bar is fair."""
    runs = []
    for seed in [0, 1, 2]:
        name = f"quality_{SMALL}_384_s{seed}_augfixed"
        flags = ["--task", "quality", "--model", SMALL, "--size", "384", "--seed", str(seed)]
        flags += ["--tag", "_augfixed"]
        runs.append((name, "scripts/21_train_single.py", flags))

        name = f"disease_{SMALL}_384_s{seed}_withreject_augfixed"
        flags = ["--task", "disease", "--model", SMALL, "--size", "384", "--seed", str(seed)]
        flags += ["--include-reject", "--tag", "_augfixed"]
        runs.append((name, "scripts/21_train_single.py", flags))
    return runs


def joint_variants():
    script = "scripts/30_train_joint.py"
    return [
        ("joint_384_s0_noreason_nodeepdridquality", script, ["--no-reason", "--no-deepdrid-quality"]),
        ("joint_384_s0_nodeepdridquality", script, ["--no-deepdrid-quality"]),
        ("joint_384_s0", script, []),
        ("joint_384_s0_grade", script, ["--grade-head"]),
    ]


def variants():
    return fixed_single_runs() + joint_variants()


def lighter_reason():
    script = "scripts/30_train_joint.py"
    base = "joint_384_s0_nodeepdridquality"
    plain = ["--no-deepdrid-quality"]
    return [
        (base + "_faults30", script, plain + ["--fault-share", "0.3", "--tag", "_faults30"]),
        (base + "_weight30", script, plain + ["--reason-weight", "0.3", "--tag", "_weight30"]),
        (base + "_grade", script, plain + ["--grade-head"]),
    ]


def more_seeds():
    script = "scripts/30_train_joint.py"
    runs = []
    for seed in ["1", "2"]:
        base = f"joint_384_s{seed}_nodeepdridquality"
        plain = ["--no-deepdrid-quality", "--seed", seed]
        runs.append((base, script, plain))
        runs.append((base + "_grade", script, plain + ["--grade-head"]))
        fewer = ["--fault-share", "0.3", "--tag", "_faults30"]
        runs.append((base + "_faults30", script, plain + fewer))
    runs.append(("joint_384_s1", script, ["--seed", "1"]))
    return runs


STAGES = {
    "variants": variants,
    "lighter_reason": lighter_reason,
    "more_seeds": more_seeds,
}


def finished_runs():
    done = set()
    for path in [SINGLE_RESULTS, JOINT_RESULTS]:
        if path.exists():
            done.update(pd.read_csv(path)["run"])
    return done


def main():
    for name, script, flags in STAGES[sys.argv[1]]():
        if name in finished_runs():
            print("already done:", name, flush=True)
            continue

        command = [sys.executable, script] + flags
        # other people's jobs can fill the GPU, so a failed run waits and tries again
        for attempt in range(RETRIES):
            if subprocess.run(command).returncode == 0:
                break
            print(f"failed, attempt {attempt + 1} of {RETRIES}", flush=True)
            time.sleep(WAIT_SECONDS)
    print("stage finished", flush=True)


if __name__ == "__main__":
    main()
