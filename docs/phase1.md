# Classical baseline: too good at first, then less so

My first quality model was gradient boosting on hand-made features, and it hit 0.923 accuracy on EyeQ validation. The best deep model in the EyeQ paper gets 0.9175 on test, and the paper's non-deep baseline gets 0.8372. Hand-made features shouldn't beat MCF-Net, so I went looking for what was wrong before believing it.

Two things turned up.

Validation is easier than test. EyeQ train is 14.4% Usable and test is 28.1%, and Usable is the class everyone gets wrong (my recall on it is 0.735, against 0.984 for Good). Reweighting my validation confusion matrix to the test mix gives about 0.882 accuracy and 0.851 macro-F1. That's an estimate, not a test result. It's still above the paper's non-deep baseline, but no longer above the deep models.

Quality also follows the camera. I used original photo size as a rough stand-in for the camera, since EyePACS has no camera field. At 3504x2336 only 17% of validation photos are Good, and at 4928x3264 it's 96%. So some of the score may just be the model recognising a camera, and that wouldn't carry over to a new one. I tried dropping feature groups, and no single group explains it (macro-F1 only moves between 0.816 and 0.858, and removing colour hurts most).

DeepDRiD has the same problem, more sharply. The 1976x1984 photos are 73.9% referable and the 1736x1824 ones are 29.1%. My features guess the photo size correctly 93.5% of the time. Lesion features alone gave a referable AUROC of 0.63, and adding quality features jumped it to 0.81, which is mostly not disease detection. Inside one photo size the AUROC drops to 0.776 and 0.685. Any CNN on DeepDRiD can take the same shortcut, so from Phase 2 on I report DeepDRiD disease scores per photo size too.

## Numbers

All on validation, from `results/phase1_quality.json`. Intervals resample patients.

| Target | Needed | Got | |
|---|---|---|---|
| EyeQ quality accuracy | at least 0.80 | 0.924 (0.910 to 0.936) | pass, see above |
| EyeQ quality macro-F1 | at least 0.65 | 0.867 (0.845 to 0.886) | pass, see above |
| DeepDRiD binary quality accuracy | at least 0.60 | 0.755, AUROC 0.824 | pass |
| Reason rules macro-F1 | at least 0.70 | 0.738 | pass |
| Not-a-fundus, frames rejected | at least 95% | 99.6% of 500 photos, 100 of 100 plain frames | pass |
| Not-a-fundus, real photos lost | at most 1% | 0.2% | pass |
| Feature time at 512 px, one thread | at most 30 ms | 22.5 ms median, 30.0 ms p95 | pass for quality features only |

The time target has a catch. The best model (macro-F1 0.867) also uses lesion features, which add 16.4 ms, so about 39 ms unless the two share their mask. The quality-only boosted model gets 0.858 at 22.5 ms and sits inside the same interval, so that's the one I'd keep as the baseline on a phone.

Classical disease detection is poor, as expected: referable AUROC is 0.696 on EyeQ Good + Usable and 0.682 on all EyeQ. Lesion features alone only reach about 0.58 to 0.59.

## What I tried

For quality, ten model and feature combinations are in `results/phase1_attempts.csv`. Sharpness alone gets 0.759 macro-F1. Logistic regression on all quality features gets 0.829, and random forest and boosting land between 0.853 and 0.867. Past that point the differences sit inside the confidence interval, so I stopped. Picking the best of ten on the same validation set also flatters it a little.

For the reason rules, three thresholds with blur from the Laplacian get 0.738, and using my normalised `focus` is the same (0.738). FFT high-frequency energy is much worse at 0.573, which surprised me. Mild faults are the hard part: 0.58 macro-F1 at the lowest strength and 0.80 at the highest. A depth-6 decision tree reaches 0.844, though it stops being something I'd call a rule.

My first not-a-fundus check used only colour and caught 76% of ordinary photos, which missed the target. Adding the brightness of the image corners took it to 99.6%, because a retina sits on black and most other things don't. My first tuning run found no valid setting at all, because the tuning set included nearly black Reject photos. It now tunes on readable photos only, and 12.8% of EyeQ Reject photos get flagged as not a fundus, which is an acceptable outcome for them since they get rejected anyway.

The first feature code took 33.4 ms and the mask alone was about 10 ms. Computing the mask at quarter size brought it to 22.5 ms.

## Open

The corner threshold is strict, at 15 out of 255, and was tuned on desk cameras. A handheld camera with stray light in the corners could get rejected as not a fundus, and I have no handheld photos to check that. A red disc on a black background would also pass.

The non-fundus photos are Imagenette, tuned on one set of 500 and tested on another. None are close-ups of skin or a red object on black, so 99.6% is likely optimistic for real camera frames.

Reason scores are on faults I added myself. Real blur from a moving hand doesn't look exactly like a Gaussian, so I expect real photos to score lower. I haven't measured it.

I haven't looked at how much DeepDRiD quality follows the camera. I only did that for disease.

Two later targets now look wrong. Phase 2 asked for accuracy of at least 0.87 and macro-F1 0.05 above this baseline, which means about 0.92. The first is already below the classical model, and the second is above anything in the EyeQ paper. Phase 3 asked for reason macro-F1 10 points above the Phase 1 rules, and whether that means 0.84 or 0.94 depends on whether the tree counts as a rule.
