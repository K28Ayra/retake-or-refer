# The one test run

Everything was frozen at commit `95a4abd` before I downloaded a single test photo. The freeze file holds the hash of every model, config, threshold file and test list, and `91_final_test.py` refuses to start if any of them has changed. The test sets were the 4,000-photo EyeQ test subset, the 400 DeepDRiD Online-Challenge photos and all 256 DeepDRiD ultra-widefield photos. None of them was used for training or tuning.

Two of the four must-beats against B2 pass cleanly and one passes with a caveat. The fourth, AUROC non-inferiority, misses by 0.0017. One EyeQ target missed too: sensitivity at the frozen threshold. The ultra-widefield drop came out just above the line I wrote down in advance for "not field-ready".

## Scorecard

Must-beat, full system vs B2 (two separate CNNs and the standard gate), EyeQ test:

| Number | Needed | Got | |
|---|---|---|---|
| Expected cost per photo | at least 10% lower, CI excludes zero | 11.4% lower, CI 3.7 to 18.1% | pass |
| Referable AUROC, Good + Usable | non-inferior within 0.01 | 0.896 vs 0.897, gap CI −0.0117 to +0.0105 | miss, by 0.0017 |
| Smallest file within a 2 point drop | at least 30% smaller | 62.8 KB; B2 passed at no setting tried | pass |
| Model size and CPU time | at most 65% of B2 | 50% and 50%; whole path about 90% | pass for the model only |

Targets, EyeQ test:

| Number | Needed | Got | |
|---|---|---|---|
| Quality accuracy | at least 0.87 | 0.891 (0.882 to 0.900) | pass |
| Quality macro-F1 | at least 0.78 | 0.876 (0.866 to 0.887) | pass |
| Referable AUROC | at least 0.88 | 0.896 (0.880 to 0.911) | pass |
| Sensitivity at the frozen threshold | at least 0.85 | 0.812 now, 0.892 if retakes come back Good | miss |
| Specificity | report | 0.617 | |

Outside sets:

| Number | Needed | Got | |
|---|---|---|---|
| DeepDRiD AUROC | at least 0.85 | 0.967 (0.940 to 0.985) | pass |
| DeepDRiD binary quality accuracy | report, loose context 0.65 to 0.70 | 0.658 | |
| DeepDRiD grade kappa | report | 0.819 | |
| Ultra-widefield AUROC drop | expected 0.05 to 0.15, above 0.15 means not field-ready | 0.896 to 0.743, a drop of 0.153 | flagged |
| Handheld camera | | not tested | not delivered |

## The cost saving is all in the rule

The full system costs 0.481 per photo and B2 costs 0.543. My first guess would have been that some of that comes from the joint model. It doesn't. The same joint model with the standard gate costs 0.544, which is B2's number. So the joint backbone matches two separate networks on AUROC and quality while being half the size and half the CPU time, but on its own it saves nothing.

The asymmetric rule does the work. Taking it away costs 13.2% more. Using a single disease threshold for every quality grade, instead of one per grade, costs 8.8% more. In Phase 7 the held-out saving over the standard gate was 6.4% and missed the 10% target. Here it's 11.7% on the same model, but that's one test set, and the CI against B2 still reaches down to 3.7%.

The cost weights are my guess (a miss is 10, an unnecessary referral 1, a retake 0.2). The plan said I'd also report other miss costs. The test script didn't, and I only noticed after the run, so `92_cost_weights.py` re-scores the same frozen decisions with each miss cost. No threshold moves. Its row for a miss cost of 10 matches the final run exactly.

| Miss cost | Full | B2 | Change | CI |
|---|---|---|---|---|
| 5 | 0.381 | 0.398 | −4.3% | −9.7 to +1.7% |
| 10 | 0.481 | 0.543 | −11.4% | −18.1 to −3.7% |
| 20 | 0.681 | 0.832 | −18.1% | −26.2 to −8.7% |
| 50 | 1.280 | 1.699 | −24.6% | −34.1 to −13.5% |

The dearer a miss, the more the rule's early referrals of Reject photos are worth. At a miss cost of 5 the saving isn't clear of zero. At 50 the single-threshold version (1.169) beats the shipped rule, because it refers more. With a miss cost that high I'd retune rather than ship these thresholds.

## Sensitivity: the rule trades it for retakes

0.812 of referable Good + Usable photos are referred straight away, short of 0.85. The rule sends borderline Usable photos for a retake rather than referring them, and retakes are cheap in the cost model. If a retaken photo comes back Good and goes through the Good threshold, sensitivity reaches 0.892. That's an assumption about what happens next, not something I measured. The standard gate on the same model refers 0.862 immediately, B2 refers 0.831, and both cost more. If immediate sensitivity is what matters in the field, the retake cost of 0.2 is the number to argue with.

## AUROC: same point, slightly too wide

The two systems sit at 0.896 and 0.897, a gap of 0.001. The paired bootstrap interval runs from −0.0117 to +0.0105, so I can't rule out the full system being up to 0.012 worse, which is outside the 0.01 margin. On validation the interval was −0.017 to +0.012 and missed as well. With 3,207 readable photos, 644 of them referable, the interval probably can't get narrower than this. More test photos would settle it.

## Sending photos

Each system kept its frozen thresholds. A setting passes if the share referred among sick photos and the share cleared among well photos both drop by less than 2 points from the original file, and fewer than 5% of photos change quality grade.

| System | Smallest passing setting | Mean file |
|---|---|---|
| Full, retina crop + 4:4:4 JPEG | 512 px, quality 95 | 62.8 KB |
| B2, plain JPEG | none of the 15 tried | original, 1,068 KB on average |

Every B2 setting, starting at 1024 px quality 95 (104 KB), cleared too few well photos. Specificity fell by 2.7 points even at the top setting. So B2 has to send the original, and the full system's file is 94% smaller than that. Against the biggest B2 file I tried, which still failed, it's 40% smaller.

This isn't the file Phase 5 picked on validation. There, 384 px at quality 95 (about 40 KB) passed. On test it dropped specificity by 2.5 points and failed. Sensitivity stayed within 2 points at every full setting, even 384 px quality 60. It's the clear decisions that move. The whole grid is in `results/final_sending.csv`.

## Size and speed

Measured before the test, on one pinned server core, in `results/phase9_bench.csv`:

| | Full | B2 |
|---|---|---|
| Model files | 1.66 MB | 3.29 MB (two files) |
| Model time per photo | 6.9 ms | 13.8 ms |
| Read, crop, resize, model | 66 ms | about 73 ms |

The model part is 50% of B2. The whole path is about 90%, because reading a 10-megapixel JPEG and shrinking it takes 58 ms whichever model follows. On a phone the photo comes from the camera at whatever size it's set to, so the 50% is the number that depends on my design. The 90% is what a user would feel.

## Other cameras

DeepDRiD is a different desk camera from EyePACS, and the disease part transfers well: AUROC 0.967, kappa 0.82 on grades. Quality doesn't transfer. The model calls 44% of DeepDRiD photos Reject, and its binary good-or-not agrees with DeepDRiD's label 66% of the time. The labels were made to different standards, so part of that is the labels, but a clinic using this camera would see a lot of retake requests. The asymmetric rule helps here more than anywhere. With so many photos called Reject, the standard gate refers only 45% of referable photos straight away, while the rule refers 90%.

Ultra-widefield is a different kind of image altogether. Every one of the 256 photos is called Reject. The disease score still ranks them a bit (AUROC 0.743, 0.770 per patient), and because a Reject photo with a high enough score refers rather than retakes, 69% of referable photos get referred. Nothing is ever cleared, so specificity is 0. The drop from EyeQ is 0.153, above the 0.15 I wrote down before the test, so by my own rule this is not field-ready on another camera type. B2 drops less on AUROC (0.770), but with the standard gate it asks for a retake on every photo.

None of this is a handheld camera. I dropped mBRSET early, and nothing here says how the system does on handheld photos. DDR, the optional second outside set, wasn't downloaded.

## Ablations

| Taken away | Cost | Quality F1 | AUROC | What it says |
|---|---|---|---|---|
| nothing (full) | 0.481 | 0.876 | 0.896 | |
| joint backbone (= B2 with the standard gate) | 0.543 | 0.871 | 0.897 | same accuracy at twice the size |
| reason head | 0.484 | 0.861 | 0.896 | small quality gain from the reason head |
| asymmetric rule | 0.544 | 0.876 | 0.896 | this is the saving |
| per-grade thresholds | 0.523 | 0.876 | 0.896 | about half the saving |
| enhancement into the model: routed | | 0.840 | 0.900 | quality gets worse |
| enhancement into the model: always CLAHE | | 0.810 | 0.877 | worse on both |

The reason head result flips on other cameras: without it, DeepDRiD AUROC is 0.981 against 0.967, while ultra-widefield is 0.675 against 0.743. I don't read much into one run each.

Two ablations from the plan don't apply. Compression training and the consistency loss aren't in the shipped model, because neither helped in Phase 5. Enhancement only goes to the screen, as decided in Phase 4, and this run backs that up. Feeding the enhanced picture to the model lowered quality F1 both ways.

Baselines for quality alone: the classical model (B1) gets 0.842 accuracy and 0.814 F1, and ResNet18 (B3) gets 0.881 and 0.864. Both are below the joint MobileNet. These numbers are on a 4,000-photo stratified subset, so they don't compare directly with the EyeQ paper's full test set.

## Validation against test

Everything got somewhat worse on test, as expected. Quality accuracy went from 0.949 to 0.891, AUROC from 0.915 to 0.896, cost per photo from 0.388 to 0.481. The saving over B2 shrank from 15.3% to 11.4%.

## What changed around the run

- The first attempt stopped at the ultra-widefield step. Patient 121's two training photos are saved as `r2.jpg` and `l3.jpg`, without the patient number in front, and the loader expected `121_r2.jpg`. The dry run skips that set, so this code had never run. I made the loader also try the short name and ran everything again. No model, threshold or config changed, and the freeze check passed again. The EyeQ comparison printed before the stop and the one from the rerun are identical. The first log is kept as `logs/phase9_final_stopped_at_uwf.log`.
- Before the freeze, I refit the B1 classical quality model. The Phase 1 feature csv was made by an earlier version of the feature code, and the saved model read the current features wrongly. It's the same model type and settings, trained on the same photos with today's code.
- Also before the freeze, the enhancement step crashed DataLoader workers once OpenCV had run in the main process. The enhancement now runs inside the workers.

## What I'd say about this system

On EyePACS-style photos it grades quality well, ranks disease about as well as two separate networks at half the size, and the rule makes it measurably cheaper than the usual gate under my costs. It fails the strict AUROC margin and the immediate-sensitivity target. On another desk camera the disease score holds up but quality doesn't. On ultra-widefield it rejects everything. It has never seen a handheld photo. That's the gap between this and a field tool.
