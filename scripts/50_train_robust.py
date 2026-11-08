"""Train the final joint model with compression added. Safe to restart: finished runs are skipped.

    python scripts/50_train_robust.py augmented
    python scripts/50_train_robust.py consistency
    python scripts/50_train_robust.py mild 0
"""
import subprocess
import sys
import time

import pandas as pd

RESULTS = "results/phase3_runs.csv"
RETRIES = 6
WAIT_SECONDS = 600
FINAL = ["--no-deepdrid-quality", "--grade-head"]
PLAIN = "outputs/phase3/joint_384_s{seed}_nodeepdridquality_grade/best.pt"
FINE_TUNE = ["--init", PLAIN, "--epochs", "6", "--lr", "3e-5"]
TEACHER = ["--teacher", PLAIN, "--distill", "1.0"]
EXTRA = {
    "augmented": ["--compress", "strong", "--tag", "_compress"],
    "consistency": ["--compress", "strong", "--consistency", "1.0", "--tag", "_consistency"],
    "mild": ["--compress", "mild", "--tag", "_mild"],
    "light": ["--compress", "light", "--tag", "_light"],
    "mild_consistency": ["--compress", "mild", "--consistency", "0.1", "--tag", "_mildconsistency"],
    "tune_light": FINE_TUNE + ["--compress", "light", "--tag", "_tunelight"],
    "teacher_mild": FINE_TUNE + TEACHER + ["--compress", "mild", "--tag", "_teachermild"],
    "teacher_strong": FINE_TUNE + TEACHER + ["--compress", "strong", "--tag", "_teacherstrong"],
}


def main():
    stage = sys.argv[1]
    seeds = sys.argv[2:] or ["0", "1", "2"]
    for seed in seeds:
        name = f"joint_384_s{seed}_nodeepdridquality_grade_{EXTRA[stage][-1][1:]}"
        if name in set(pd.read_csv(RESULTS)["run"]):
            print("already done:", name, flush=True)
            continue

        flags = []
        for flag in EXTRA[stage]:
            flags.append(flag.replace("{seed}", seed))
        command = [sys.executable, "scripts/30_train_joint.py"] + FINAL + flags
        command += ["--seed", seed]
        for attempt in range(RETRIES):
            if subprocess.run(command).returncode == 0:
                break
            print(f"failed, attempt {attempt + 1} of {RETRIES}", flush=True)
            time.sleep(WAIT_SECONDS)
    print("stage finished", flush=True)


if __name__ == "__main__":
    main()
