# retake-or-refer

Takes a colour retinal photo and tells a screening worker what to do next: refer the patient, retake the photo, or clear them. The goal is to catch a bad photo while the patient is still in the room, on a phone, with photos small enough to send over a weak network.

Right now you can run the data pipeline, a classical baseline and two separate small CNNs, one for quality and one for disease. The joint model comes next, and nothing runs on a phone yet. This is a research project and not a medical device.

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
python scripts/00_download.py imagenette
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
| Imagenette | testing the not-a-fundus check | 1,000 sampled | none |

Referable means ICDR grade 2 or higher. Splits are by patient.

There's no handheld camera data in this project. mBRSET is the one public handheld set with DR grades I could find, and it needs credentialed PhysioNet access, so I left it out. Results here won't tell you how the model does on handheld photos, and you should keep that in mind when reading any number in this repo.

Licences: DeepDRiD is CC-BY-SA-4.0, the EyeQ code is CC BY-NC-SA 4.0, and EyePACS comes with Kaggle's competition rules. No images are in this repo, so get them from the source and follow their terms.

## Layout

`retake/` is the small package with the reusable code. `scripts/` runs in number order. `configs/default.yaml` holds settings, including the `realtime` switch: `false` reads photos from a dataset, `true` will read from a camera. Only the switch exists so far, the camera code isn't written yet.

## Phase 0: data

Built the splits and checked them before doing anything else. No patient appears in two splits, every image opens, and there are no exact duplicates.

Two things in the data changed the plan. DeepDRiD quality is binary, so the 0.65 to 0.70 from its challenge can't be compared with a three-class score. I'll compare against the EyeQ paper (0.8914 accuracy for ResNet18) instead. And no dataset labels why a photo is bad, so the blur / uneven light / low contrast output has to be trained on damage I add to photos myself.

A small training run on the EyeQ cache used 2.64 GB of GPU memory at 384 px, batch 32, at 283 images per second. More in [docs/phase0.md](docs/phase0.md).

## Phase 1: classical baseline

Hand-made features (sharpness, light evenness, contrast, colour, lesion-like spots) with gradient boosting.

```bash
python scripts/10_extract_features.py
python scripts/11_classical_quality.py
python scripts/12_check_leakage.py
python scripts/13_reason_rules.py
python scripts/14_not_fundus_and_timing.py
```

The whole thing takes a few minutes on CPU. Validation results:

| Task | Result |
|---|---|
| EyeQ quality, three classes | 0.924 accuracy, 0.867 macro-F1 |
| Same, reweighted to the EyeQ test class mix | about 0.882 accuracy, 0.851 macro-F1 |
| DeepDRiD quality, binary | 0.755 accuracy |
| Fault reason on added faults | 0.738 macro-F1 with thresholds, 0.844 with a small tree |
| Not-a-fundus | rejects 99.6% of ordinary photos, loses 0.2% of real ones |
| Referable AUROC on EyeQ Good + Usable | 0.696 |
| Feature time | 22.5 ms at 512 px on one thread |

Don't read that 0.924 as beating the published deep models. EyeQ validation has half the share of Usable photos that test has, and quality follows the camera: 17% of photos are Good at one photo size and 96% at another.

DeepDRiD has it worse. 74% of the 1976x1984 photos are referable and 29% of the 1736x1824 ones, so a model can score well by spotting the camera. From here on I report DeepDRiD disease scores per photo size. Details and what I tried are in [docs/phase1.md](docs/phase1.md).

## Phase 2: single-task CNNs

One network for quality, one for disease, and a look at how small they can get.

```bash
python scripts/20_make_deepdrid_cache.py
python scripts/22_run_phase2.py everything
python scripts/22_run_phase2.py final
python scripts/23_compare_phase2.py
python scripts/25_cpu_time.py
```

That's 42 trainings and about 5 hours on one GPU. A run is 5 to 12 minutes, and finished runs get skipped if you restart. To train just one, use `scripts/21_train_single.py`.

MobileNetV3 small at 384 px won, because nothing bigger did better. It's 6.1 MB and 12.9 ms per photo on one server CPU thread, which only tells you how the models rank. These are validation means over three seeds:

| Task | Classical | CNN |
|---|---|---|
| EyeQ quality accuracy | 0.924 | 0.943 |
| EyeQ quality macro-F1 | 0.867 | 0.901 |
| Usable recall | 0.735 | 0.851 |
| Referable AUROC on EyeQ Good + Usable | 0.696 | 0.907 |
| DeepDRiD disease AUROC | 0.81 | 0.966 |

I missed the stretch goal of 0.92 AUROC. The best run got 0.916, and going up to 448 px added 0.001.

My early stopping broke the first batch. It stopped runs while the learning rate was still high and scores were jumping around, so 18 of 30 quit too soon and the size comparison was meaningless. Small at 384 px went from 0.868 to 0.901 once every run got all 20 epochs. Putting EyeQ's Reject photos into disease training also helped, even though I'd left them out on purpose.

Don't take the DeepDRiD 0.966 at face value. A model that never saw DeepDRiD still gets 0.939, because that set has far more severe disease than EyeQ (22% of photos against 2.6%) and is easier for reasons I can't fully explain. Quality scores are also flattered by picking the best epoch on validation. The rest is in [docs/phase2.md](docs/phase2.md).
