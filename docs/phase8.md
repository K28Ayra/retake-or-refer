# One command, two modes

`run.py` is the app. With `realtime: false` in `configs/default.yaml` it reads a folder or a csv and writes one row and one enhanced picture per photo. With `--realtime on` it reads a camera, a stream address or a video file, checks every frame, keeps the best recent one, and runs the full model when you capture. Both modes go through the same `Screener` in `retake/pipeline.py`, which loads the 1.66 MB ONNX file from Phase 6 and the thresholds from Phase 7.

Neither mode needs torch. I moved `prepare_image` into `images.py` and `reason_answers` into `decide.py`, and made the U-Net load only when asked for. The old import paths still work. So a laptop can run the app from `requirements-live.txt` and the model file alone.

| Target | Needed | Got | |
|---|---|---|---|
| Dataset mode, all four outputs | every photo | 200 of 200 | pass |
| Same decisions as Phase 7 | 200 photos | 200 of 200 | pass |
| `realtime: false` never opens a camera | a test | `test_dataset_mode_never_opens_a_camera` | pass |
| Replayed clips, sharp frame picked | at least 90% | 46 of 50 (92%) | pass |
| Frame checks | at least 10 per second | 88 per second | pass, server core |
| Capture to decision | at most 1 s | 23 ms median, 46 ms slowest | pass, server core |
| Real handheld fundus camera (stretch) | | not done | open |

Speeds are on one pinned core of the Xeon from Phase 6, not a laptop. There's a lot of room, but a laptop number still needs someone with a laptop.

## What each photo gets

Quality (good, usable or reject) with its probability. The fault reason. The calibrated chance of referable DR. A decision with its confidence and, for a retake, whether it's because of a bad photo or a borderline score on a usable one. Whether the never-clear rule applied. And the enhanced picture, which only a person looks at. The model always reads the original, as Phase 4 decided.

The test that no camera is opened replaces `cv2.VideoCapture` with something that fails if it's ever called, runs dataset mode on a photo, and checks that `retake.live` never got imported. It uses a stand-in screener, so it runs in CI without the model file.

## Picking the sharp frame

Each frame is shrunk, the retina is found with the Phase 1 mask, and three numbers come out: how much of the frame is retina, how bright the retina is, and the variance of the Laplacian on green, inside the retina and away from its rim. A frame with too little retina or too little light is unusable. The window keeps the last 60 frames, about 2 seconds at 30 fps, and on capture it takes the sharpest usable one.

I had no fundus videos, so I made 50 clips from DeepDRiD validation photos at 1280 px wide. In each 60-frame clip one random frame is sharp, blur grows by 0.3 sigma per frame away from it, and every frame gets a small random shift, a little extra blur and some noise. They're written as mp4 and replayed through `--source`, so the video decoding is real too. A clip passes if the chosen frame has blur sigma of 1.0 or less.

| Width the check runs at | Sharp frame picked | Checks per second |
|---|---|---|
| 320 px | 40 of 50 | 184 |
| 640 px | 46 of 50 | 88 |
| 1280 px | 44 of 50 | 23 |

At 320 px the mild blur near the sharp moment barely shows after shrinking, and the chosen frame was 2 frames off on average. At full width the added noise counts as sharpness too. 640 px won, so that's the setting. All three runs used the same 50 clips, so 640 was picked on the same clips it's scored on. The four misses chose frames with blur 1.0 to 1.7, so close calls rather than badly blurred frames.

These clips test the software loop and the frame choice. Blur, shake and noise from a real handheld camera won't look like my Gaussian versions, and none of this says the model reads a live fundus image correctly.

## Running it

Dataset mode, on any folder of photos:

```bash
python run.py --input photos/ --out outputs/run
```

Live mode on a laptop, with only the live requirements and the model file:

```bash
pip install -r requirements-live.txt
python run.py --realtime on --source 0 --model int8_weights.onnx
```

Space captures and q quits. `--source` also takes a phone camera stream, from an app that serves the camera over Wi-Fi, or a video file. `--no-window` captures the best frame without showing anything, which is how the replay check runs.

## Open

The live window has never been on a screen. The server has no display, so the replay check runs without one, and the drawing code only has a test that it writes text onto the frame. The same goes for a phone camera stream by address.

A webcam can't image a retina. Until someone points a fundus camera at an eye, live mode only shows that the loop works.

The decision rule underneath is Phase 7's, with its cost assumptions.
