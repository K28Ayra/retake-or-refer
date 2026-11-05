# retake-or-refer

Takes a colour retinal photo and tells a screening worker what to do next: refer the patient, retake the photo, or clear them. The goal is to catch a bad photo while the patient is still in the room, on a phone, with photos small enough to send over a weak network.

Right now only the data side works. The models come next. This is a research project and not a medical device.

## Setup

```bash
git clone https://github.com/K28Ayra/retake-or-refer.git
cd retake-or-refer
conda env create -p envs/retina -f environment.yml
conda activate ./envs/retina
pytest
```

The tests don't need data or a GPU.

## Getting the data

You need a Kaggle account for EyePACS. Accept the rules on the [competition page](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules) first, because the download fails without that. Then create an API token and save it in `.kaggle/access_token` in the repo folder.

```bash
python scripts/00_download.py eyeq_labels
python scripts/00_download.py deepdrid
python scripts/00_download.py eyepacs_train
python scripts/01_make_splits.py
python scripts/02_prepare_eyepacs.py train
python scripts/03_audit.py
```

`eyepacs_train` won't start with less than 60 GB free, though it peaks at about 45 GB. The zip gets deleted once the images are out, and you end up with about 6 GB in `data/`. Leave `eyepacs_test` alone until the final evaluation. It's another 53 GB and nothing before then should touch it.

To check your GPU can handle training:

```bash
python scripts/04_smoke_train.py --size 384 --batch 32
```

## Datasets

| Dataset | Used for | Images | Quality label |
|---|---|---|---|
| [EyeQ](https://github.com/HzFu/EyeQ) (on EyePACS photos) | train, val, test | 12,543 train, 4,000 of 16,249 test | good / usable / reject |
| [DeepDRiD](https://github.com/deepdrdoc/DeepDRiD) regular | train, val, test | 1,200 / 400 / 400 | binary |
| DeepDRiD ultra-widefield | outside camera check | 256 | none |

Referable means ICDR grade 2 or higher. Splits are by patient.

There's no handheld camera data in this project. mBRSET is the one public handheld set with DR grades I could find, and it needs credentialed PhysioNet access, so I left it out. Results here won't tell you how the model does on handheld photos, and you should keep that in mind when reading any number in this repo.

Licences: DeepDRiD is CC-BY-SA-4.0, the EyeQ code is CC BY-NC-SA 4.0, and EyePACS comes with Kaggle's competition rules. No images are in this repo, so get them from the source and follow their terms.

## Layout

`retake/` is the small package with the reusable code. `scripts/` runs in number order. `configs/default.yaml` holds settings, including the `realtime` switch: `false` reads photos from a dataset, `true` will read from a camera. Only the switch exists so far, the camera code isn't written yet.

## Phase 0: data

Built the splits and checked them before doing anything else. No patient appears in two splits, every image opens, and there are no exact duplicates.

Two things in the data changed the plan. DeepDRiD quality is binary, so the 0.65 to 0.70 from its challenge can't be compared with a three-class score. I'll compare against the EyeQ paper (0.8914 accuracy for ResNet18) instead. And no dataset labels why a photo is bad, so the blur / uneven light / low contrast output has to be trained on damage I add to photos myself.

A small training run on the EyeQ cache used 2.64 GB of GPU memory at 384 px, batch 32, at 283 images per second. More in [docs/phase0.md](docs/phase0.md).
