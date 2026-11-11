# Starting from nothing

The last check was whether someone else can get this running from the GitHub page alone. I cloned the public repo into an empty folder and ran `sh setup_tools.sh`. That built the env from `environment.yml` in 20 minutes, most of it downloading torch. All 85 tests passed in 6 seconds.

Then I gave the clone the model file and the photos, and ran dataset mode on the same 200 photos as Phase 8:

```bash
python run.py --input check_photos.csv --out outputs/run
```

I compared it with the same command run from my working copy. Every value in all 200 rows matched exactly, including quality, reason, probabilities, decision, confidence and the never-clear flag. Every enhanced image matched byte for byte. The only column that differed was the folder the enhanced pictures were written to.

| Check | Got | |
|---|---|---|
| Fresh env from `environment.yml` | built, 20 min | pass |
| `pytest` in the fresh clone | 85 passed | pass |
| Phase 8 dataset run reproduced | 200 of 200 identical | pass |
| CI on the last push | green | pass |
| `check_signs.py` over docs and README | no em-dashes or emoji; the few flagged words are false hits, like "there's" matching "here's" | pass |
| Docs share a skeleton | none repeated | pass |
| Model card | [model_card.md](model_card.md) | done |

The fresh clone needed two things the repo can't hold: the photos, which you download yourself, and `int8_weights.onnx`. You build that file by training (Phase 3) and exporting (Phase 6). It isn't published because I haven't confirmed that the EyePACS rules allow sharing weights trained on their photos. So "fresh clone reproduces the run" here means given the same model file, not from scratch. I didn't repeat the training.

## Still open at the end

- No handheld camera, ever. This is the biggest gap.
- No phone timing. The laptop browser numbers from Phase 6 are the closest I have.
- Live mode has only run on replayed clips, never with a real camera on a screen.
- `run.py` doesn't use the not-a-fundus check from Phase 1. A photo of something else would still get a decision.
- The costs behind every threshold are mine, not a clinician's.
