# Refer, retake or clear

My rule costs 6.4% less per photo than the standard gate on patients it wasn't tuned on (95% CI 3.5 to 9.5%). The target was 10%, so the main target is missed. The direction holds at every miss cost I tried, but the size only reaches 10% when a miss costs 5.

The rule is "a bad photo can refer, but it can never clear". A photo the model grades Reject is referred straight away if its disease probability is at least 0.48, and otherwise retaken. A Usable photo is referred at 0.40 or more, cleared under 0.063, and retaken in between. A Good photo is referred at 0.063 or more. The standard gate retakes every Reject and uses one threshold for the rest.

Everything here uses the model that ships: seed 1 with int8 weights, run on the 1,784 EyeQ validation photos read from the original files the way the phone reads them. 337 of them (18.9%) are labelled Reject.

## The costs, and what I assumed

A missed referable eye costs 10, an unnecessary referral 1, and a retake 0.2, as in the plan. A retake also needs an outcome, and no dataset gives one. I assumed the second photo is a typical Good photo of the same eye, decided with the Good threshold, so a retake costs 0.2 plus the miss or over-referral rate on Good photos. That's kind to retakes. A handheld camera that failed once will often fail again, and then referring now would save more than I count.

## Finding a rule that holds up

Tuning thresholds and scoring them on the same photos flatters the rule, so the main number is held out. I split patients in half, tuned on one half and scored the other, swapped, then repeated that over 20 random splits and averaged each photo's cost. The standard gate is tuned the same way, on the same costs.

| Rule | Thresholds | Same photos | Held out | Held-out 95% CI |
|---|---|---|---|---|
| Standard gate + "a Reject can refer" | 2 | -5.6% | -3.6% | -6.8 to -0.6% |
| Good and Usable-clear share one threshold | 3 | -9.3% | -6.4% | -9.5 to -3.5% |
| All four separate | 4 | -9.3% | -3.8% | -7.3 to -0.3% |
| Three thresholds + a tuned Reject probability | 4 | -9.1% | -8.3% | -11.1 to -5.8% |

The four-threshold rule matched the three-threshold one on the same photos and lost on held-out ones, which is overfitting. Before running these I wrote down how I'd choose: keep the simpler rule unless a bigger one lowers the held-out cost by more than 0.005 per photo. The last row looks best as a percentage, but only because the standard gate got worse on held-out patients when given the same extra freedom. In absolute terms my rule went from 0.4262 to 0.4259. That's the second try in a row gaining less than 0.005, so I stopped there.

My first run used 5 splits and gave -1.0% held out at miss cost 10, against -11.5% at 5 and -10.4% at 20. Five splits were too few for an answer that jumpy, so I went to 20.

## Against the targets

Miss cost 10, three-threshold rule.

| Target | Needed | Got | |
|---|---|---|---|
| Expected cost against the standard gate | at least 10% lower, CI excluding zero | 6.4% lower held out (CI 3.5 to 9.5%), 9.3% on the tuning photos | missed |
| Sensitivity with cost-optimal thresholds | at least 0.90 | 0.935 counting retakes, 0.797 referred on the first photo | pass |
| A Reject photo never clears | unit test | `test_a_reject_photo_is_never_cleared` | pass |
| Referable photos the gate retakes, referred now | at least 25%, at least 70% right | 52 of 84 (62%), and 52 of the 64 early referrals are referable (81%) | pass |
| Retake rate | at most the Reject share + 5 points | 21.0% against 18.9% | pass |
| ECE after calibration | at most 0.05 | 0.005 for disease, 0.009 for quality | pass, see below |
| 80% sensitivity at 95% specificity (stretch) | | 86.3% specificity at 80% sensitivity, 68.4% sensitivity at 95% | missed |

Sensitivity is on the tuning photos, since that's how the target is worded.

## Cost weights

| Miss cost | Standard gate, held out | Mine, held out | Change | 95% CI |
|---|---|---|---|---|
| 5 | 0.357 | 0.321 | -10.1% | -14.2 to -6.3% |
| 10 | 0.455 | 0.426 | -6.4% | -9.5 to -3.5% |
| 20 | 0.598 | 0.543 | -9.1% | -13.0 to -5.6% |
| 50 | 0.740 | 0.711 | -3.9% | -9.1 to +1.4% |

At a miss cost of 50 nearly everyone uncertain gets referred under either rule, so there's little left to separate them.

## Calibration nearly fooled me

The plan said temperature scaling. For quality that works: one temperature of 1.27 takes top-class ECE from 0.015 to 0.009. For disease, my first ECE measured confidence in whichever answer the model leaned to, and it said 0.009. Mostly that was easy negatives. Among referred photos the model claimed 0.58 on average and was right 0.47 of the time. In the middle, photos given about 0.79 were referable half the time.

The disease output was trained with a weight on positives, which shifts every score up, and a temperature can only stretch scores, not move them. Measured properly, as "does 0.3 mean yes 30% of the time", ECE is 0.076 before and 0.079 with a temperature. Platt scaling, which fits a scale and a shift, brings it to 0.005. Referred photos now claim 0.44 and are right 0.47 of the time, and cleared ones claim 0.976 and are right 0.978. Both keep the order of the scores, so no decision changed. The thresholds above are on the Platt scale.

The confidence the app shows is that calibrated probability: of disease for a referral, of no disease for a clear, of Reject for a retake. A retake on a Usable photo is labelled as a borderline disease score rather than a bad photo. I didn't add the margin from the plan, because the probability already says how close a call was.

## What the rule actually does

| True quality | Cleared | Referred | Retaken |
|---|---|---|---|
| Good (1,190) | 721 | 461 | 8 |
| Usable (257) | 105 | 45 | 107 |
| Reject (337) | 10 | 68 | 259 |

The rule refers 32% of everyone. With a miss costing 10 times an unnecessary referral, the cheapest Good threshold is 0.063, so anything with a 6% chance of disease goes. That's what these costs ask for, and it leaves specificity at 0.56. A real programme refers far fewer people, and it would need costs from a clinician, not me.

The 10 true Rejects that were cleared were graded Good or Usable by the model. A photo the model calls Reject never clears, which is the rule. A photo it wrongly calls fine can still clear, as it could under the standard gate.

## Two other checks

On the 40 KB photos from Phase 5, with thresholds unchanged, the cost is 0.393 against 0.388 on originals, and 88.7% of decisions stay the same. Of the 200 that change, half started with a disease probability between 0.054 and 0.118, around the 0.063 threshold, and only 17 changed quality grade. The phone decides on the original photo, so this matters only if a decision is ever made from the sent file.

On DeepDRiD validation (400 photos), the model calls 42% of photos Reject, a sign the quality grader doesn't transfer well to that camera. My rule costs 0.224 there against 0.397 for the gate, because so many of those "Reject" photos show clear disease and get referred instead of retaken. That gap says more about the quality grader on a new camera than about the rule. DeepDRiD has no three-class quality labels, so retakes there use EyeQ's follow-up rates.

## Open

The costs and the retake outcome are my assumptions. The main result changes with them, from -3.9% to -10.1% across miss costs.

The held-out numbers come from splitting the same validation photos, not from new photos. Phase 9's test set will check them.

The tuned values are in `configs/decision.yaml`.
