# Model card: retake-or-refer joint model

Not a medical device. Not clinically validated. Never tested on a handheld camera. It's a research model, and nobody should use it to decide anything about a real patient.

## What it is

One MobileNetV3 small (timm `mobilenetv3_small_100`, ImageNet weights to start) that reads a 384 x 384 colour fundus photo and gives four outputs:

- quality: good, usable or reject
- chance of referable diabetic retinopathy (ICDR grade 2 or higher)
- fault reason: blur, uneven light, low contrast (three yes/no outputs)
- DR grade 0 to 4

The shipped file is `int8_weights.onnx`, 1.66 MB. Its weights are stored as int8 and turned back into floats once when the file loads. The input is uint8 RGB, shape (1, 384, 384, 3), after cropping to the retina and resizing. Normalisation is inside the file. It isn't in this repo; `scripts/60_export.py` builds it from the trained checkpoint.

The model's raw outputs aren't the decision. `configs/decision.yaml` holds the calibration (a temperature for quality, Platt scaling for disease) and the thresholds that turn them into refer, retake or clear. Rejected photos can refer but never clear.

## Training data

- EyeQ training photos (EyePACS, 12,543 photos, minus a patient-disjoint share kept for validation): quality labels and DR grades.
- DeepDRiD regular training photos: DR grades only. Adding its quality labels made results worse, so they're left out.
- Fault reasons come from blur, uneven light and low contrast added on purpose to good photos, because no dataset labels the reason.

All thresholds and calibration were set on validation photos. The test sets were used once, after everything was frozen.

## Results on test

All from `results/final_summary.json`. The CIs resample patients.

| Set | Number | Value |
|---|---|---|
| EyeQ test, 4,000 photos | quality accuracy | 0.891 (0.882 to 0.900) |
| | quality macro-F1 | 0.876 (0.866 to 0.887) |
| | referable AUROC, good + usable photos | 0.896 (0.880 to 0.911) |
| | sensitivity, referred straight away | 0.812 |
| | specificity | 0.617 |
| DeepDRiD Online-Challenge, 400 photos | referable AUROC | 0.967 (0.940 to 0.985) |
| | quality agreement with DeepDRiD's label | 0.658 |
| DeepDRiD ultra-widefield, 256 photos | referable AUROC | 0.743 (0.670 to 0.810) |
| | photos called reject | all 256 |

Speed on one server CPU core: 6.9 ms for the model, 66 ms including reading and resizing a 10-megapixel photo. In Chrome on a laptop (Intel i5-1135G7) the model took 27 to 30 ms. It hasn't been timed on a phone.

## Limits

- It has only seen desk fundus cameras. On ultra-widefield photos it calls everything a reject, and the AUROC drop is bigger than the 0.15 I set in advance as the line for not field-ready. A handheld camera may behave like that or worse, and nobody has checked.
- On a different desk camera (DeepDRiD) the disease score transferred well but quality didn't: 44% of photos were called reject.
- EyePACS disease labels are noisy, so part of what the AUROC measures is label noise.
- The refer, retake or clear rule depends on costs I chose myself: a missed referable case costs 10, an unnecessary referral 1, a retake 0.2. A clinician should set those. With other costs the thresholds should be retuned.
- The rule assumes a retaken photo comes back good. If it doesn't, sensitivity is lower than the 0.892 that assumption gives.
- The enhanced picture is for a person to look at. The model reads the original photo. Feeding it enhanced photos made quality grading worse.
- The model only ever saw retinal photos and has no "this isn't an eye" output. The classical check `looks_like_fundus` in `retake/quality_features.py` is there for that, but `run.py` doesn't call it.
- I haven't split results by patient age, sex or ethnicity.

## Licences

EyePACS photos come under Kaggle's competition rules, EyeQ labels under CC BY-NC-SA 4.0, and DeepDRiD under CC-BY-SA-4.0. The trained weights are derived from that data. I haven't confirmed that the EyePACS rules allow sharing them, so they aren't published here.
