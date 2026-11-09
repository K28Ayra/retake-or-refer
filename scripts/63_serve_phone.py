"""Serve the phone page on your local network.

    python scripts/63_serve_phone.py outputs/phase6/s1/int8_weights.onnx

Then open the printed address on a phone connected to the same Wi-Fi.
"""
import functools
import http.server
import shutil
import socket
import sys

PORT = 8000


def local_address():
    # no packet is sent, this only asks which network card would be used
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def main():
    shutil.copy(sys.argv[1], "phone/model.onnx")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="phone")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler)
    print(f"open http://{local_address()}:{PORT} on the phone, ctrl+c to stop", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
