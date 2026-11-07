# Single-task CNNs: the smallest model was enough

Correction, added during Phase 3. Every run in this doc had a bug: each photo got at most 8 different augmented versions, not a new one each epoch. After I fixed it, the chosen models improved to macro-F1 0.905 (0.904 to 0.906) and AUROC 0.917 (0.913 to 0.922), with accuracy 0.945 and Usable recall 0.882, all three seeds. One seed now reaches the 0.92 stretch goal, and the mean doesn't. Those runs have `_augfixed` in their names in `results/phase2_runs.csv`. I didn't rerun the other backbones or sizes, so everything below is the old numbers, and the comparisons still compare like with like. More in [phase3.md](phase3.md).

I'm taking MobileNetV3 small at 384 px into Phase 3. It's 6.1 MB in float32 and takes 12.9 ms on one server CPU thread, and I couldn't show that anything bigger is better. ResNet18 is seven times the size and scored about the same.

I trained 42 runs with timm 1.0.26 and torch 2.4.1, plus 18 that I threw away (more on those below). Everything here is on validation. The test set hasn't been touched.

| Target | Needed | MobileNetV3 small, 384 px, 3 seeds | |
|---|---|---|---|
| Quality accuracy | at least 0.90 | 0.942, 0.942, 0.945 | pass |
| Quality macro-F1 above classical | paired CI above zero | 0.901, 0.899, 0.902 against 0.867. Lowest CI on the gap starts at +0.011 | pass |
| Usable recall | at least 0.75 | 0.860, 0.848, 0.844 | pass |
| Referable AUROC, EyeQ Good + Usable | at least 0.88 | 0.915, 0.904, 0.902 | pass |
| Same, stretch | 0.92 | best single run 0.916 | missed |
| DeepDRiD disease AUROC | at least 0.85 | 0.966, 0.965, 0.967 | pass, but read the DeepDRiD section |
| Match ResNet18 with a phone-sized model (stretch) | | 0.901 mean against 0.900 from one ResNet18 seed at 320 px | met on validation only |

The stretch AUROC is the one real miss. I tried 448 px as a last attempt and got 0.916 against 0.915 on the same seed, which is not worth 17.5 ms instead of 12.9 ms, so I stopped.

## My first batch of runs was unfair

Quality on the small model at 320 px gave 0.876, 0.904 and 0.908 across three seeds. A gap that size looked wrong. The first seed had its best epoch at 2 and stopped at 7, while the other two peaked at 16 and 17.

The cause was my early stopping. I used patience 5 with a one-cycle learning rate that peaks at about epoch 2, and validation scores jump around badly in that stretch. One run went 0.867, 0.499, 0.875. So runs could quit before the rate came down, and 18 of the first 30 stopped early. Most of my input size and class-balance comparisons were affected. Small at 384 px looked like 0.868 and was really 0.901.

I removed early stopping and reran those 18. The old numbers are in `results/phase2_runs_patience5.csv`.

## The choices

| Quality at 320 px | MB | ms, one thread | Macro-F1 | Usable recall |
|---|---|---|---|---|
| MobileNetV3 small, 3 seeds | 6.1 | 9.9 | 0.902 mean, 0.895 to 0.908 | 0.751 to 0.883 |
| MobileNetV3 large, 3 seeds | 16.8 | 26.8 | 0.903 mean, 0.898 to 0.907 | 0.809 to 0.817 |
| ResNet18, 1 seed | 44.7 | 85.8 | 0.900 | 0.848 |
| EfficientNet-Lite0, 1 seed | 13.5 | 39.5 | 0.884 | 0.759 |

Lite0's gap over classical has a CI that crosses zero, so it fails the target outright.

Input size matters for disease and barely for quality. Without Reject photos, referable AUROC on the small model is 0.843 at 224 px, 0.890 at 320 px (three seeds) and 0.906 at 384 px. Quality goes 0.898, 0.902, 0.901. Seeds also agree much better on quality at 384 px, where macro-F1 stays between 0.899 and 0.902 and Usable recall between 0.844 and 0.860. At 320 px Usable recall swings from 0.751 to 0.883, so 384 px it is. The joint model in Phase 3 needs one size anyway.

One result went against what I expected. I'd left EyeQ Reject photos out of disease training because their labels looked shaky. Putting them back helped: on the small model at 320 px the mean goes from 0.890 to 0.897, and on all EyeQ photos from 0.874 to 0.890. No seed got worse. The large model gained 0.013. I don't know why, maybe it's just 20% more data or maybe it copes better with poor photos. I kept them in.

Class balancing didn't matter much. Without it the small model gets macro-F1 0.896 and the large one 0.901 (one seed each), about the same as with it. Nothing argued for changing it, so balancing stayed.

## DeepDRiD's 0.966 is not the real performance

It went above 0.92 after a single epoch, so I checked. Here's what I found, using the small model at 320 px with one seed:

- Trained on EyeQ only, so it has never seen a DeepDRiD photo, it still scores 0.939. Adding DeepDRiD to training brings 0.966, so 0.027 comes from training on the same dataset.
- DeepDRiD validation is 45% referable and 22% grades 3 and 4. EyeQ is 17% and 2.6%. Severe cases are easy, and that explains part of the gap. EyeQ reweighted to DeepDRiD's grade mix goes from 0.898 to 0.920.
- The mix doesn't explain all of it. Grade 2 against non-referable photos gets 0.923 on DeepDRiD and 0.884 on EyeQ with the same EyeQ-only model.
- 131 of 400 validation photos correlate above 0.98 with some train photo at 32x32. Without them the EyeQ-only model drops to 0.911. It never trained on those train photos, so this can't be leakage. Those 131 are just easier. Fundus photos look alike at that scale anyway, and the median is 0.974.

The camera shortcut I was worried about after Phase 1 doesn't seem to drive this. Inside one photo size the chosen model gets 0.963 to 0.968 at 1736x1824 and 0.896 to 0.943 at 1976x1984. The classical model got 0.776 and 0.685 there. That second size is also where seeds disagree most, and the single-seed EyeQ-only model gets only 0.848 on it.

I can't fully explain what's left. I'd trust the EyeQ number more than this one, and DeepDRiD won't count as evidence of external performance later.

## Photo size on quality

Accuracy for the three seeds against the classical model:

| Photo size | Photos | Classical | CNN |
|---|---|---|---|
| 2560x1920 | 133 | 0.865 | 0.902 to 0.932 |
| 2592x1944 | 257 | 0.942 | 0.949 to 0.961 |
| 3008x2000 | 61 | 1.000 | 0.984 to 1.000 |
| 3504x2336 | 157 | 0.803 | 0.847 to 0.873 |
| 3888x2592 | 539 | 0.930 | 0.954 to 0.963 |
| 4752x3168 | 208 | 0.947 | 0.918 to 0.938 |
| 4928x3264 | 166 | 0.994 | 0.982 to 0.994 |

The CNN helps most on the hardest size, 3504x2336, but it's worse than classical at 4752x3168. It doesn't win everywhere.

## How much to trust the numbers

I pick the best epoch on the same validation set I report, which flatters results. The damage is small at 384 px, where last-epoch macro-F1 is 0.899, 0.896 and 0.901. It's bigger at 320 px, where the first seed goes from 0.895 to 0.877.

Reweighted to the EyeQ test class mix, quality comes to about 0.925 accuracy and 0.905 macro-F1. That's above every number in the EyeQ paper, including 0.9175 and 0.8551 for DenseNet121 MCF, and I don't believe a 6 MB model beats it. The reweighting only corrects class shares, and my validation comes from EyeQ's training photos. We'll find out in Phase 9.

The 12.9 ms is float32 on a server CPU. Use it to rank models, not to predict phone speed. Phase 6 measures the real export.

## Open

The size and variant comparisons have one seed in many cells. Seeds differ by up to 0.014 on disease, so small gaps there mean nothing. The EyeQ-only result is one seed too.

Disease is scored on EyeQ labels I haven't checked for grader noise. It might be why EyeQ scores lower than DeepDRiD, and I have no way to test that here.

None of this is on handheld photos.

Phase 3 now has a clear bar: within 0.01 of macro-F1 0.901 and AUROC 0.907, both means of three seeds at 384 px.
