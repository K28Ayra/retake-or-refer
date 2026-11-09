# A 1.7 MB model, and a page that runs it on a phone

The phone gets one ONNX file of 1.66 MB that takes plain 384 x 384 RGB pixels and returns the four outputs. Its weights are int8 and its activations stay float. On one pinned server core, the whole path from a 10 MP JPEG to the answer takes 67 ms median and 115 ms at p95. The `phone/` page runs the same file in a browser with onnxruntime-web.

I haven't run it on a real phone yet. That part needs you, and it's described at the end.

| Target | Needed | Got | |
|---|---|---|---|
| Size | at most 10 MB, stretch 5 MB | 1.66 MB | pass, stretch too |
| AUROC drop from quantising | at most 0.01 | 0.0028 mean over 3 seeds, worst seed 0.0078 | pass |
| Macro-F1 drop | at most 0.01 | 0.0039 mean, worst seed 0.0064 | pass |
| Time, one thread, with preprocessing | at most 300 ms median | 67 ms median, 115 ms p95 | pass |
| ONNX against PyTorch before quantising | within 1e-3 on 100 photos | largest gap 0.00008 to 0.00018 | pass |
| Real phone (stretch) | measured | not done | open |

That's a weaker int8 than the plan meant. The plan was full static int8, with int8 maths, and that failed badly on this model.

## Full int8 ruined the model

I used ONNX Runtime 1.19.2 static quantisation in QDQ format, with weights per channel, calibrated on 300 training photos. Referable AUROC went from 0.909 to 0.555 on seed 0, and quality macro-F1 from 0.911 to 0.306. Nothing I tried on seed 0 brought it back:

| Tried | MB | Macro-F1 | AUROC |
|---|---|---|---|
| Float32 | 6.14 | 0.911 | 0.909 |
| Full int8, min-max ranges | 1.90 | 0.306 | 0.555 |
| Only Conv and Gemm quantised | 1.89 | 0.253 | 0.551 |
| Squeeze-excite left in float | 3.19 | 0.225 | 0.579 |
| Squeeze-excite, stem and blocks 0 and 1 in float | 3.20 | 0.268 | 0.569 |
| 99.99 percentile ranges, set by hand | 1.90 | 0.453 | 0.543 |
| Same, plus biases in float | 1.80 | 0.661 | 0.611 |
| Int8 weights, float activations | 1.66 | 0.905 | 0.901 |

To find out why, I quantised one layer at a time and left the rest in float. With only the first convolution in int8, the referable score correlated 0.27 with the float model. With only the first depthwise convolution in int8, 0.24. Most later layers on their own stayed between 0.91 and 1.0.

Batch norm folded into those early layers leaves channels whose ranges differ by hundreds of times. In the first depthwise layer, per-channel weight maxima run from 0.16 to 104. Weights get one scale per channel, so they're fine. Activations get one scale for the whole tensor, so the small channels round to almost nothing. One tensor I checked had a calibrated range of -191 to +55 against a typical value of about 1.2. That puts each int8 step at 0.97.

Two things about ONNX Runtime cost me time. `quantize_static` in 1.19.2 doesn't pass a percentile through to the calibrator, so my first percentile runs used 99.999 whatever I asked for. Asking for entropy calibration gave a file byte-identical to min-max, and I didn't find out why. In the end I ran the percentile calibrator myself and handed its ranges over as `TensorQuantOverrides`. It still failed.

What does work is storing only the weights as int8, one scale per output channel, with a `DequantizeLinear` in front of each Conv and Gemm (`retake/export.py`). At first that ran slower than float, 16.6 ms against 6.5, because it dequantised on every photo. Setting `session.disable_quant_qdq` makes ONNX Runtime do it once at load, and then it runs as fast as float32. The page sets the same option.

I didn't try quantisation-aware training. Full int8 would mostly buy speed, and the model already takes 7 ms of the 67. The weights-only file meets every target.

## Seed 1 goes on the phone

| | Macro-F1 | AUROC, Good + Usable | DeepDRiD AUROC | Reason F1 |
|---|---|---|---|---|
| Float32 ONNX | 0.917 | 0.916 | 0.972 | 0.982 |
| Int8 weights | 0.911 | 0.915 | 0.972 | 0.982 |

Seed 1 had the best Phase 3 macro-F1 and nearly the best AUROC. That's picked on validation, so it flatters seed 1 a bit. The other two seeds lose at most 0.008 AUROC after quantising.

Float32 ONNX on CPU gives a mean macro-F1 of 0.908, against 0.909 in Phase 3. That's GPU half precision against CPU float32, not the export.

## The phone has to see what Python sees

Everything before the network has to run on the phone in JavaScript. That's the crop, the shrink to 512, the pad and the resize to 384, in `phone/preprocess.js`. My first version gave every pixel within 1 grey level of OpenCV, but 4.2% of pixels on real photos were off by 1. That moved the model's raw scores by up to 0.22.

There were two causes. OpenCV's area resize rounds halves up when it shrinks by a whole number and to even otherwise, and the 512 to 384 step lands on exact halves a lot. It also adds up in 32-bit floats, so `Math.fround` copies that. After both fixes, 50 real validation photos give identical model inputs and identical outputs. A pytest runs the JS through node on synthetic photos.

Speed in JavaScript was the other problem. In onnxruntime-web 1.19.2 under node on one server core, the model took 59 ms, about 9 times slower than native. My preprocessing took 579 ms on a 15 MP photo. Flat typed arrays and a row-at-a-time vertical pass brought it to about 250 ms, and the match stayed exact. A phone will be slower, and only a real phone can say how much.

## Timing details

500 EyeQ validation originals, median 10.1 MP, ONNX Runtime and OpenCV on one thread, pinned with `taskset`:

| Step | Median | p95 |
|---|---|---|
| Reading the JPEG | 34 ms | 70 ms |
| Crop and resize | 25 ms | 41 ms |
| Model | 6.9 ms | 7.1 ms |
| Total | 67 ms | 115 ms |

The process peaked at 641 MB, but 399 MB of that is Python, pandas and the libraries before the model loads. This is a Xeon Gold 5317 at 3 GHz, so treat it as an upper bound on how fast a phone can go, not a phone number.

## Trying it on your phone

The page loads onnxruntime-web 1.19.2 from jsDelivr, and the model file sits next to the page. It isn't in the repo.

```bash
python scripts/63_serve_phone.py outputs/phase6/s1/int8_weights.onnx
```

Open the printed address on a phone on the same Wi-Fi, take or pick a photo, and the page shows the outputs and how long each step took. "Time 20 runs" gives the model time on its own. It runs with one thread so it compares with the server number.

## Open

No real phone number yet. That's the stretch target, and it isn't done until someone runs the page on a phone.

The model is touchy about early-layer noise. A 1-level difference on 4% of the pixels moved scores by up to 0.22, and quantising one layer wrecked it. I don't know whether that also makes it fragile to real phone camera noise, and Phase 5's compression results don't settle it.

The phone has to apply the JPEG orientation correctly, and the browser does that when it decodes. I haven't checked colour profiles on a phone.
