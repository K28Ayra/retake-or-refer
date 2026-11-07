# One network instead of two

Can a single MobileNetV3 small do quality, disease and fault reason at once, without getting worse? On means over three seeds, yes. It's 6.1 MB against 12.2 MB for two models, and 13.0 ms against 26.9 ms on one server CPU thread. I can't claim much more than that, because the intervals on the comparison are wide.

The model has a shared backbone at 384 px and small output layers for three-class quality, referable disease, three fault reasons and the five DR grades. Each photo teaches only the outputs it has a label for. Everything is on validation.

| Target | Needed | Got, 3 seeds | |
|---|---|---|---|
| Quality macro-F1 | within 0.01 of 0.905 | 0.909 mean (0.895 to 0.920) | pass on the mean |
| Referable AUROC, EyeQ Good + Usable | within 0.01 of 0.917 | 0.913 mean (0.909 to 0.916) | pass on the mean |
| Size | at most 65% of two models | 50% | pass |
| Time | at most 65% of two models | 48% | pass |
| Reason macro-F1 on added faults | at least 0.87, CI above 0.844 | 0.978, 0.982, 0.980. Lowest CI starts at 0.973 | pass |
| Blur score against DeepDRiD Clarity | Spearman at least 0.40 in size | -0.53, -0.47, -0.48 | pass on the point value |
| Joint beats separate models on AUROC (stretch) | CI above zero | no seed comes close | missed |

Two of those passes are weaker than they look.

The non-inferiority one is the mean. Macro-F1 is 0.004 above the separate models and AUROC is 0.003 below. On the same photos, seed by seed, the paired intervals are far wider than my 0.01 margin. For macro-F1 the gap runs from -0.009 to +0.022, -0.001 to +0.029, and -0.026 to +0.007. For AUROC it's -0.031 to +0.004, -0.012 to +0.017, and -0.015 to +0.017. All three cross zero, and they also go past -0.01 in most cases. So I found no sign of harm, but this validation set is too small to prove there's none.

The Clarity intervals go down to 0.43, 0.36 and 0.39 in size, so two of the three dip under 0.40.

## The bar moved because of a bug in my own code

While writing the data loader I noticed that `persistent_workers=True` doesn't give workers a new seed each epoch, which my comment in `train_utils.py` said it did. I counted distinct versions per photo over 20 epochs and got at most 8, one per worker, so Phase 2 trained on much weaker augmentation than I thought.

I fixed that, then retrained the six chosen Phase 2 models. Otherwise the joint model would have looked better just for having the fix. It helped more than expected: quality went from 0.901 to 0.905, and AUROC from 0.907 to 0.917. Those new numbers are the bar above.

## What I tried

One thing at a time, seed 0 unless noted. The target for both columns is at least 0.895 and 0.907.

| Outputs | Macro-F1 | AUROC | Reason F1 |
|---|---|---|---|
| Quality + disease | 0.912 | 0.912 | |
| + reason | 0.903 (3 seeds) | 0.913 (3 seeds) | 0.984 |
| + reason, 30% as many faulty photos | 0.908 (3 seeds) | 0.913 (3 seeds) | 0.969 |
| + reason, loss weight 0.3 | 0.900 | 0.900 | 0.967 |
| + reason + 5 grades | 0.909 (3 seeds) | 0.913 (3 seeds) | 0.980 |
| + reason + DeepDRiD quality | 0.893 (2 seeds) | 0.904 (2 seeds) | 0.979 |
| + reason + DeepDRiD quality + 5 grades | 0.897 | 0.917 | 0.982 |

Before running the extra seeds I wrote down how I'd choose: keep the plain reason model unless something beats it by more than 0.005 on one metric without being worse on the other. The grade version beat it by 0.0055 on macro-F1 and 0.0004 on AUROC, so it won. That's a rule being applied, not evidence that grades help. Seeds differ by 0.025 on macro-F1 there, and the lowest seed has Usable recall of 0.829, against 0.875 and 0.895.

The DeepDRiD quality output failed both margins on two seeds, so I removed it. It also wasn't any good: AUROC 0.795 to 0.800, below the 0.824 the classical model got in Phase 1. I don't know why adding it hurts the other outputs, and I didn't test any explanation. The app won't report a DeepDRiD-style quality score.

Lowering the reason weight to 0.3 was simply bad, with AUROC at 0.900 on the one seed I ran. I tried it because last-epoch macro-F1 on seed 0 was 0.889 with reason and 0.912 without. That looked like a real cost, but I only have one seed without reason, so I can't say how big it is.

## Reason on real photos

This one worries me more than any number above. On faults I added, reason macro-F1 is about 0.98, and 0.96 on the mildest. Here's what the model says about real EyeQ validation photos, as the range across seeds:

| Real grade | No fault named | Blur | Uneven light | Low contrast |
|---|---|---|---|---|
| Good | 99.1 to 99.6% | 0% | 0.3 to 0.8% | 0.1 to 0.3% |
| Usable | 84.4 to 85.6% | 5.1 to 6.6% | 5.4 to 6.6% | 1.9 to 3.9% |
| Reject | 41.8 to 52.5% | 26.4 to 33.8% | 13.1 to 23.7% | 3.9 to 8.0% |

The direction is right, since worse grades get more faults named. But for about half of real Rejects the model says reject and names no reason. The brief wants a reason, so this is a gap in the output. Some of those photos may have faults I never simulated, like cut-off fields, dark frames or artefacts. I can't tell because no dataset labels this. Real blur also isn't a Gaussian. The 0.98 tells you the model recognises my faults and not much about real ones.

## Also measured

Grade agreement (quadratic kappa) is 0.64 to 0.68 on EyeQ Good + Usable and 0.80 to 0.83 on DeepDRiD. The per-photo-size DeepDRiD AUROC is 0.962 to 0.973 at 1736x1824, and 0.916 to 0.923 at 1976x1984. I said in Phase 2 not to read much into DeepDRiD and that still holds.

The 4.5 hours of training for the 14 runs here was much less than I'd planned for, and the six Phase 2 retrains added 50 minutes.

## Open

I still want 100 real Reject photos labelled by hand with why they're bad, so reasons have some real ground truth. Without it that half with no reason stays unexplained.

Best-epoch selection uses the same validation set as the report, and now it's on the mean of three scores. Last-epoch means are 0.903 and 0.913, against 0.900 and 0.914 for the separate models, so this time the effect is small.

Nothing here is on a test set, a handheld photo or a phone yet.
