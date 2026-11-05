"""Write every split to data/splits. Needs labels only, no EyePACS images."""
from pathlib import Path

import pandas as pd

from retake.config import load_config
from retake.data import (
    load_deepdrid_online,
    load_deepdrid_regular,
    load_eyeq_labels,
    split_eyeq_train,
    subset_eyeq_test,
)


def main():
    config = load_config()
    seed = config["seed"]
    out = Path(config["paths"]["data"]) / "splits"
    out.mkdir(parents=True, exist_ok=True)

    eyeq_train = load_eyeq_labels("data/raw/eyeq/Label_EyeQ_train.csv")
    val_fraction = config["splits"]["val_fraction"]
    eyeq_train = split_eyeq_train(eyeq_train, val_fraction, seed)
    eyeq_train.to_csv(out / "eyeq.csv", index=False)

    eyeq_test = load_eyeq_labels("data/raw/eyeq/Label_EyeQ_test.csv")
    n_test = config["splits"]["eyeq_test_subset"]
    eyeq_test = subset_eyeq_test(eyeq_test, n_test, seed)
    eyeq_test.to_csv(out / "eyeq_test.csv", index=False)

    root = "data/raw/deepdrid"
    train = load_deepdrid_regular(root, "regular-fundus-training", "train")
    val = load_deepdrid_regular(root, "regular-fundus-validation", "val")
    pd.concat([train, val]).to_csv(out / "deepdrid.csv", index=False)

    online = load_deepdrid_online(root)
    online.to_csv(out / "deepdrid_test.csv", index=False)

    print("eyeq", eyeq_train["split"].value_counts().to_dict())
    print("eyeq test", len(eyeq_test))
    print("deepdrid", len(train), len(val), "online", len(online))


if __name__ == "__main__":
    main()
