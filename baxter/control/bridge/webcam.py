#!/usr/bin/env python3
"""Overhead USB webcam frame source for perception.

Opens the camera once, warms it up (early frames wash out while it auto-exposes
and initialises -- see memory), and serves fresh BGR frames. Keep one instance
open for a session rather than reopening (reopen = another warm-up).

    from webcam import Webcam
    cam = Webcam(0)
    img = cam.frame()        # HxWx3 BGR, or None
    cam.close()
"""

import threading
import time

import cv2


class Webcam(object):
    def __init__(self, index=0, warmup_s=2.5):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "cannot open webcam %s"
                % (index if isinstance(index, str) else "/dev/video%d" % index)
            )
        # discard frames while the camera initialises (first ~seconds wash out)
        t = time.time()
        while time.time() - t < warmup_s:
            self.cap.read()

    def frame(self):
        """Return a current BGR frame (a few reads to flush any buffer)."""
        ok, f = False, None
        for _ in range(5):
            ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        self.cap.release()


class WebcamStream(Webcam):
    """Webcam read continuously on a thread; latest() never blocks the caller.

    For control loops: frame() takes ~5 reads per call, too slow per tick.
    """

    def __init__(self, index=0, warmup_s=2.5):
        Webcam.__init__(self, index, warmup_s)
        self._img, self._t, self._run = None, None, True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                self._img, self._t = f, time.monotonic()

    def latest(self):
        """(bgr ndarray, monotonic capture time) or (None, None)."""
        return self._img, self._t

    def close(self):
        self._run = False
        self._thread.join(timeout=1.0)
        Webcam.close(self)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from perception import detect_blocks, annotate

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--save", default="usb_cam.jpg")
    args = ap.parse_args()
    cam = Webcam(args.index)
    img = cam.frame()
    cam.close()
    if img is None:
        print("no frame")
        sys.exit(1)
    objs = detect_blocks(img)
    cv2.imwrite(args.save, img)
    cv2.imwrite(args.save.replace(".jpg", "_annotated.jpg"), annotate(img, objs))
    print("frame %s -> %d blob(s):" % (img.shape, len(objs)))
    for o in objs:
        print(
            "  %-6s pixel=(%.0f,%.0f) area=%.0f"
            % (o["color"], o["pixel"][0], o["pixel"][1], o["area"])
        )
