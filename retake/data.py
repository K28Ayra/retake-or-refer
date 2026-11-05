import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

QUALITY_NAMES = {0: "good", 1: "usable", 2: "reject"}


def is_referable(grade, referable_grade=2):
    return int(grade >= referable_grade)


def eyepacs_patient_id(image_name):
    # EyePACS names look like 10009_left.jpeg
    return image_name.split("_")[0]


def load_eyeq_labels(csv_path):
    table = pd.read_csv(csv_path, index_col=0)
    table["patient"] = table["image"].map(eyepacs_patient_id)
    table["referable"] = table["DR_grade"].map(is_referable)
    return table


def _strata(table):
    quality = table["quality"].astype(str)
    referable = table["referable"].astype(str)
    return quality + "_" + referable


def split_eyeq_train(table, val_fraction, seed):
    """Patient-disjoint split, stratified on quality and referable together."""
    n_folds = round(1 / val_fraction)
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = _strata(table)
    fold_iter = folds.split(table, labels, groups=table["patient"])
    train_idx, val_idx = next(fold_iter)

    table = table.copy()
    table["split"] = "train"
    table.iloc[val_idx, table.columns.get_loc("split")] = "val"
    return table


def subset_eyeq_test(table, n_images, seed):
    labels = _strata(table)
    kept, _ = train_test_split(
        table,
        train_size=n_images,
        stratify=labels,
        random_state=seed,
    )
    kept = kept.sort_index().copy()
    kept["split"] = "test"
    return kept


def deepdrid_eye_grade(row):
    """The csv has one column per eye and only the photographed eye is filled."""
    left = row["left_eye_DR_Level"]
    right = row["right_eye_DR_Level"]
    if pd.notna(left):
        return int(left)
    if pd.notna(right):
        return int(right)
    return None


def deepdrid_image_path(folder, patient, image_id):
    # the paths inside the csv files skip the Images folder, so build our own
    return f"{folder}/Images/{patient}/{image_id}.jpg"


def load_deepdrid_regular(root, name, split):
    folder = f"{root}/regular_fundus_images/{name}"
    table = pd.read_csv(f"{folder}/{name}.csv")

    table["DR_grade"] = table.apply(deepdrid_eye_grade, axis=1)
    table["referable"] = table["DR_grade"].map(is_referable)
    table["patient"] = table["patient_id"].astype(str)
    table["split"] = split

    paths = []
    for _, row in table.iterrows():
        path = deepdrid_image_path(folder, row["patient"], row["image_id"])
        paths.append(path)
    table["image_path"] = paths

    keep = [
        "patient",
        "image_id",
        "image_path",
        "split",
        "DR_grade",
        "referable",
        "Overall quality",
        "Clarity",
        "Field definition",
        "Artifact",
    ]
    return table[keep]


def load_deepdrid_online(root):
    folder = f"{root}/regular_fundus_images/Online-Challenge1&2-Evaluation"
    grades = pd.read_excel(f"{folder}/Challenge1_labels.xlsx")
    quality = pd.read_excel(f"{folder}/Challenge2_labels.xlsx")

    grades = grades.rename(columns={"DR_Levels": "DR_grade"})
    table = grades.merge(quality, on="image_id", validate="one_to_one")

    table["referable"] = table["DR_grade"].map(is_referable)
    table["patient"] = table["image_id"].str.split("_").str[0]
    table["split"] = "test"

    paths = []
    for _, row in table.iterrows():
        path = deepdrid_image_path(folder, row["patient"], row["image_id"])
        paths.append(path)
    table["image_path"] = paths
    return table


def shared_patients(table_a, table_b):
    patients_a = set(table_a["patient"])
    patients_b = set(table_b["patient"])
    return patients_a & patients_b
