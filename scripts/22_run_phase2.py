"""Run a list of trainings one after another. Safe to restart: finished runs are skipped.

    python scripts/22_run_phase2.py backbones
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RESULTS = Path("results/phase2_runs.csv")
RETRIES = 6
WAIT_SECONDS = 600
PHONE_MODELS = ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_lite0"]


def backbones():
    runs = []
    for task in ["quality", "disease"]:
        for model in PHONE_MODELS + ["resnet18"]:
            runs.append({"task": task, "model": model, "size": 320})
    return runs


def sizes():
    runs = []
    for task in ["quality", "disease"]:
        for model in ["mobilenetv3_small_100", "mobilenetv3_large_100"]:
            for size in [224, 384]:
                runs.append({"task": task, "model": model, "size": size})
    return runs


def variants():
    runs = []
    for model in ["mobilenetv3_small_100", "mobilenetv3_large_100"]:
        for task in ["quality", "disease"]:
            no_balance = {"task": task, "model": model, "size": 320}
            no_balance["flags"] = ["--no-balance"]
            no_balance["suffix"] = ["_nobalance"]
            runs.append(no_balance)
        with_reject = {"task": "disease", "model": model, "size": 320}
        with_reject["flags"] = ["--include-reject"]
        with_reject["suffix"] = ["_withreject"]
        runs.append(with_reject)
    return runs


def seeds():
    runs = []
    for task in ["quality", "disease"]:
        for model in ["mobilenetv3_small_100", "mobilenetv3_large_100"]:
            for seed in [1, 2]:
                runs.append({"task": task, "model": model, "size": 320, "seed": seed})
    return runs


def extras():
    runs = []
    for model in ["mobilenetv3_small_100", "mobilenetv3_large_100"]:
        for seed in [1, 2]:
            runs.append(
                {
                    "task": "disease",
                    "model": model,
                    "size": 320,
                    "seed": seed,
                    "flags": ["--include-reject"],
                    "suffix": ["_withreject"],
                }
            )
        runs.append(
            {
                "task": "disease",
                "model": model,
                "size": 384,
                "flags": ["--include-reject"],
                "suffix": ["_withreject"],
            }
        )
    runs.append(
        {
            "task": "disease",
            "model": "mobilenetv3_small_100",
            "size": 320,
            "flags": ["--eyeq-only"],
            "suffix": ["_eyeqonly"],
        }
    )
    return runs


def final_checks():
    small = "mobilenetv3_small_100"
    with_reject = {"flags": ["--include-reject"], "suffix": ["_withreject"]}
    runs = []
    for seed in [1, 2]:
        runs.append({"task": "disease", "model": small, "size": 384, "seed": seed, **with_reject})
        runs.append({"task": "quality", "model": small, "size": 384, "seed": seed})
    runs.append({"task": "disease", "model": small, "size": 448, **with_reject})
    return runs


def everything_after_backbones():
    return sizes() + seeds() + variants()


def everything():
    return backbones() + everything_after_backbones() + extras()


STAGES = {
    "backbones": backbones,
    "sizes": sizes,
    "variants": variants,
    "seeds": seeds,
    "rest": everything_after_backbones,
    "extras": extras,
    "everything": everything,
    "final": final_checks,
}


def run_name(run):
    return f"{run['task']}_{run['model']}_{run['size']}_s{run.get('seed', 0)}" + "".join(
        run.get("suffix", [])
    )


def finished_runs():
    if not RESULTS.exists():
        return set()
    return set(pd.read_csv(RESULTS)["run"])


def main():
    for run in STAGES[sys.argv[1]]():
        if run_name(run) in finished_runs():
            print("already done:", run_name(run), flush=True)
            continue

        command = [
            sys.executable,
            "scripts/21_train_single.py",
            "--task", run["task"],
            "--model", run["model"],
            "--size", str(run["size"]),
            "--seed", str(run.get("seed", 0)),
        ]
        command += run.get("flags", [])

        # other people's jobs can fill the GPU, so a failed run waits and tries again
        for attempt in range(RETRIES):
            if subprocess.run(command).returncode == 0:
                break
            print(f"failed, attempt {attempt + 1} of {RETRIES}", flush=True)
            time.sleep(WAIT_SECONDS)
    print("stage finished", flush=True)


if __name__ == "__main__":
    main()
