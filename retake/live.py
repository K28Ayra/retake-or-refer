"""Live mode: quick checks on every frame, keep the best recent one, run the full model on capture.

Only run.py imports this, and only when realtime is on.
"""
import time
from collections import deque

import cv2
import numpy as np

from retake.quality_features import fundus_mask

CHECK_WIDTH = 640
MIN_COVERAGE = 0.15
MIN_BRIGHTNESS = 25
WINDOW = 60


def frame_check(frame):
    """Cheap enough for every frame: how much retina, how bright, how sharp."""
    height, width = frame.shape[:2]
    scale = CHECK_WIDTH / width
    small = cv2.resize(frame, (CHECK_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    mask = fundus_mask(small)
    coverage = float(mask.mean())
    green = small[:, :, 1]
    if coverage == 0:
        return {"coverage": 0.0, "brightness": 0.0, "sharpness": 0.0, "message": "no retina in view"}

    brightness = float(green[mask].mean())
    edges = cv2.Laplacian(green.astype(np.float32), cv2.CV_32F)
    # stay off the rim, where the edge of the retina would count as sharpness
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    sharpness = float(edges[inner].var()) if inner.any() else 0.0

    message = "ok"
    if coverage < MIN_COVERAGE:
        message = "no retina in view"
    elif brightness < MIN_BRIGHTNESS:
        message = "too dark"
    return {"coverage": coverage, "brightness": brightness, "sharpness": sharpness, "message": message}


class BestFrame:
    """The last few frames and their checks. Usable frames beat unusable ones, then the sharpest wins."""

    def __init__(self, size=WINDOW):
        self.frames = deque(maxlen=size)

    def add(self, index, frame, check):
        self.frames.append((index, frame, check))

    def best(self):
        if not self.frames:
            return None
        usable = [item for item in self.frames if item[2]["message"] == "ok"]
        pool = usable if usable else list(self.frames)
        return max(pool, key=lambda item: item[2]["sharpness"])


def open_source(source):
    """A camera number, a stream address such as a phone camera app, or a video file to replay."""
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(str(source))


def draw(frame, check, best_sharpness, answer):
    shown = frame.copy()
    lines = [f"{check['message']}  sharpness {check['sharpness']:.0f} (best {best_sharpness:.0f})"]
    if answer:
        lines.append(f"{answer['decision'].upper()}  confidence {answer['confidence']:.2f}")
        lines.append(f"quality {answer['quality']}, fault {answer['reason']}")
        lines.append(f"referable DR {answer['referable_probability']:.2f}")
    for row, text in enumerate(lines):
        position = (10, 30 + row * 30)
        cv2.putText(shown, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(shown, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def run_live(screener, source, show=True, window=WINDOW):
    """Space captures, q quits. Without a window, or when a replayed video ends, the best frame is captured."""
    capture = open_source(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")

    best = BestFrame(window)
    answer = None
    check_times = []
    index = 0
    result = None
    while True:
        ok, frame = capture.read()
        wants_capture = not ok
        if ok:
            start = time.perf_counter()
            check = frame_check(frame)
            check_times.append(time.perf_counter() - start)
            best.add(index, frame, check)
            index += 1

        if show and ok:
            chosen = best.best()
            cv2.imshow("retake or refer", draw(frame, check, chosen[2]["sharpness"], answer))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            wants_capture = key == ord(" ")

        if wants_capture:
            chosen = best.best()
            if chosen is None:
                break
            start = time.perf_counter()
            answer = screener.screen(chosen[1])
            decide_ms = (time.perf_counter() - start) * 1000
            result = {"frame": chosen[0], "check": chosen[2], "answer": answer, "decide_ms": decide_ms}
            if not ok or not show:
                break

    capture.release()
    if show:
        cv2.destroyAllWindows()
    if result is not None:
        result["frames"] = index
        result["checks_per_second"] = len(check_times) / max(sum(check_times), 1e-9)
    return result
