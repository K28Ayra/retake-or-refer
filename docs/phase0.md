# Data notes

I expected this to be the boring part. Mostly it was, except that reading the actual label files knocked out two assumptions in my project brief.

The first was the DeepDRiD quality benchmark. My brief says the top challenge teams got 0.65 to 0.70 accuracy on quality, and I had planned to measure my three-class grader against that. But `Overall quality` in the DeepDRiD csv files is 0 or 1, with Clarity, Field definition and Artifact scored separately. The paper doesn't help much either: its abstract calls the range accuracy, and its results section attributes 0.6981 to 0.6938 to weighted kappa. So I now use the numbers in Table 2 of the EyeQ paper (Fu et al., MICCAI 2019) as the real target, 0.8372 accuracy for their non-deep baseline and 0.8914 for ResNet18 on RGB.

The second was the quality reasons. I need to say whether a photo failed from blur, uneven light or low contrast. No dataset has that. I'll have to add those faults to photos myself so the true answer is known, and the only check on real photos is an indirect one against DeepDRiD Clarity.

I also dropped mBRSET, the handheld set from the brief, because of the PhysioNet credentialing it needs. I looked for a handheld set with DR grades that I could just download and didn't find one. That means the handheld result in the brief won't exist. The outside-camera test will use DeepDRiD ultra-widefield, and possibly DDR, and neither is handheld.

## What the audit found

Numbers come from `scripts/03_audit.py`, saved in `results/phase0_audit.json`.

| Split | Images | Patients | Referable | Quality counts |
|---|---|---|---|---|
| EyeQ train | 10,759 | 7,383 | 2,058 | 7,157 / 1,619 / 1,983 |
| EyeQ val | 1,784 | 1,231 | 335 | 1,190 / 257 / 337 |
| EyeQ test subset | 4,000 | 3,752 | 859 | 2,085 / 1,122 / 793 |
| DeepDRiD train | 1,200 | 300 | 520 | 624 / 576 |
| DeepDRiD val | 400 | 100 | 180 | 218 / 182 |
| DeepDRiD test | 400 | 100 | 164 | 220 / 180 |

Quality counts are good / usable / reject for EyeQ, and 0 / 1 for DeepDRiD.

Zero patients overlap between splits inside each dataset. All 10,759 + 1,784 EyeQ images and all 2,000 DeepDRiD images open, and no two files have the same hash. That hash check won't catch the same photo saved twice at different settings.

The EyeQ README table is slightly off from its own csv. The csv has 1,876 Usable in train and 4,558 in test, and 8,471 Good in test. The README says 1,896, 4,559 and 8,470. I went with the csv.

Val came out at 14.2% of EyeQ train and not the 15% in my config, because `StratifiedGroupKFold` needs a whole number of folds and I used 7.

## Things worth knowing if you run this

Kaggle wraps every part of the EyePACS zip in its own zip, so part 1 arrives as 8,391,159,087 bytes when the file list says 8,388,608,000. `scripts/00_download.py` unwraps each part, checks the inner size and then appends it. I only noticed because the reported size didn't match, and my first version would have thrown away a finished download.

EyePACS and DeepDRiD photos are all JPEG already (EyeQ val median 1,086 KB, DeepDRiD about 530 KB). So for the compression work, "uncompressed" doesn't exist and the baseline is the file as distributed. That's why val keeps the original bytes, while train only keeps a 512 px PNG cache.

## Test data

I tried to stay away from it, but not perfectly. To build the splits I printed EyeQ test label counts, and I looked at two rows of the DeepDRiD test label files to learn the format. The audit opened the DeepDRiD test images to check they load. The EyeQ test images aren't downloaded yet. No model has seen any of it.

## GPU run

`scripts/04_smoke_train.py` with `mobilenetv3_large_100` from timm 1.0.26 on torch 2.4.1, 384 px, batch 32, 40 batches: 2.64 GB peak GPU memory, 283 images per second. That's about 40 seconds per EyeQ epoch if the GPU stays that free, which is a lot faster than I had guessed.

## Still open

- `environment.yml` changed after I built my env (dropped the kaggle package, added requests and the editable install). I haven't rebuilt from scratch yet.
- CI hasn't run on GitHub because nothing is pushed. The same tests pass in a clean venv from `requirements-ci.txt`, 12 tests in about a second.
- I haven't read the full Kaggle competition rules.
- The ultra-widefield split isn't built yet. It's only needed for the final run.
- I haven't checked whether DDR is a clean download.
