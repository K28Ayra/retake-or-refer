"""Screen retinal photos: refer, retake or clear.

Dataset mode (realtime: false in configs/default.yaml):
    python run.py --input photos/ --out outputs/run

Live mode, from a webcam, a phone camera stream, or a video replayed as if it were a camera:
    python run.py --realtime on --source 0
    python run.py --realtime on --source http://192.168.1.20:8080/video
    python run.py --realtime on --source clip.mp4 --no-window
"""
import argparse
import json

from retake.config import load_config
from retake.pipeline import Screener, run_dataset

MODEL = "outputs/phase6/s1/int8_weights.onnx"


def on_or_off(text):
    if text not in ("on", "off"):
        raise argparse.ArgumentTypeError("use on or off")
    return text == "on"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refer, retake or clear for retinal photos.")
    parser.add_argument("--realtime", type=on_or_off, help="on or off, overrides the config")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--input", help="dataset mode: a folder of photos or a csv with a path column")
    parser.add_argument("--out", default="outputs/run")
    parser.add_argument("--source", help="live mode: camera number, stream address or video file")
    parser.add_argument("--no-window", action="store_true", help="live mode: no display, capture the best frame")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.realtime)
    screener = Screener(args.model)

    if not config["realtime"]:
        if args.input is None:
            parser.error("dataset mode needs --input")
        table = run_dataset(screener, args.input, args.out)
        print(f"{len(table)} photos, results in {args.out}/results.csv")
        print(table["decision"].value_counts().to_string())
        return table

    # only now, so dataset mode never loads any camera code
    from retake.live import run_live

    source = args.source if args.source is not None else config["camera_source"]
    result = run_live(screener, source, show=not args.no_window)
    if result is None:
        print("nothing captured")
        return None
    answer = dict(result["answer"])
    answer.pop("enhanced")
    speed = {
        "frames_seen": result["frames"],
        "checks_per_second": round(result["checks_per_second"], 1),
        "captured_frame": result["frame"],
        "decide_ms": round(result["decide_ms"], 1),
    }
    print(json.dumps({**speed, **answer}, indent=1))
    return result


if __name__ == "__main__":
    main()
