"""
beacon_detect.py

Finds the blinking LED beacon in the camera feed and streams the video
with a red crosshair on it.

Run on the Pi:   python3 beacon_detect.py
Then open:       http://192.168.1.218:8000/video   in your laptop's browser

How it works, in one sentence: keep the last second of frames, and for every
pixel ask "does this pixel turn on and off exactly 5 times per second?"
The beacon does. Room lights, people, and screens don't.
"""

import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn


# Settings: change these numbers to tune the detector

CAMERA_INDEX = 0         # /dev/video0 is the Brio
WIDTH, HEIGHT = 640, 480
FPS = 30
EXPOSURE = 83            # units of 100 microseconds, so 83 = 8.3 ms (cancels light flicker)

BLINK_HZ = 5.0           # MUST match BLINK_HZ in the beacon firmware
WINDOW_SECONDS = 1.0     # how much history to look at (1 s = 30 frames = 5 blinks)
SMALL_WIDTH = 160        # analyze a shrunk copy of each frame, for speed
MIN_SCORE = 0.5          # how "pure" the 5 Hz flicker must be, 0 to 1 (clean LED is about 0.9)
MIN_AMPLITUDE = 15       # how strong the flicker must be, in brightness levels (0 to 255)

PORT = 8000


# Camera setup

def lock_camera_settings():
    """Turn off everything automatic, so the ONLY thing changing between
    frames is the beacon. Same commands you typed by hand earlier."""
    device = f"/dev/video{CAMERA_INDEX}"
    settings = [
        "auto_exposure=1",              # manual exposure
        "exposure_dynamic_framerate=0", # never drop below 30 fps
        f"exposure_time_absolute={EXPOSURE}",
        "white_balance_automatic=0",
        "backlight_compensation=0",
    ]
    for setting in settings:
        subprocess.run(["v4l2-ctl", "-d", device, "-c", setting], check=False)


def open_camera():
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))  # uncompressed, no JPEG noise
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, FPS)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always give us the newest frame
    if not cam.isOpened():
        raise RuntimeError("Could not open camera. Is the Brio plugged in?")
    lock_camera_settings()  # do this AFTER opening, since opening can reset them
    return cam


# The beacon finder

class BeaconFinder:
    def __init__(self):
        self.frames = deque()  # last second of grayscale frames
        self.times = deque()   # the exact time each one was captured

    def update(self, frame, t):
        """Feed in one new frame. Returns the beacon's (x, y) position
        in full size pixels, or None if no beacon is found."""

        # 1. Shrink and convert to grayscale. Color doesn't help us, and a
        #    small image is much faster to analyze.
        h, w = frame.shape[:2]
        small_h = int(h * SMALL_WIDTH / w)
        small = cv2.resize(frame, (SMALL_WIDTH, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2. Add it to the history, and drop anything older than WINDOW_SECONDS.
        self.frames.append(gray)
        self.times.append(t)
        while t - self.times[0] > WINDOW_SECONDS:
            self.frames.popleft()
            self.times.popleft()

        # Wait until we have (almost) a full second of history.
        times = np.array(self.times)
        if times[-1] - times[0] < 0.8 * WINDOW_SECONDS:
            return None

        # 3. Stack frames into one 3D block: (number of frames, height, width).
        #    Then subtract each pixel's average brightness, so we only see CHANGE.
        stack = np.stack(self.frames)
        change = stack - stack.mean(axis=0)

        # 4. For every pixel at once, measure how much of its change happens
        #    at exactly BLINK_HZ. We compare each pixel's brightness over time
        #    against a perfect 5 Hz wave. If they line up, the result is big.
        #    (This is a single frequency Fourier transform. Using the real
        #    capture times means slightly uneven frame timing doesn't hurt.)
        phase = 2 * np.pi * BLINK_HZ * (times - times[0])
        cos_part = np.tensordot(np.cos(phase), change, axes=1)
        sin_part = np.tensordot(np.sin(phase), change, axes=1)
        n = len(times)

        # amplitude: how strongly the pixel flickers at 5 Hz
        amplitude = (2.0 / n) * np.sqrt(cos_part ** 2 + sin_part ** 2)
        # total: how much the pixel changes overall, at ANY rhythm
        total = np.sqrt(2.0 * (change ** 2).sum(axis=0) / n)
        # score: what fraction of its change is the 5 Hz beacon rhythm.
        # Beacon is about 0.9. A person walking by is low, since they change
        # but not in a steady 5 Hz pattern.
        score = amplitude / (total + 1e-6)

        # 5. Keep pixels that are both pure (high score) and strong (high amplitude).
        hits = (score > MIN_SCORE) & (amplitude > MIN_AMPLITUDE)
        if not hits.any():
            return None

        # 6. Group neighboring hit pixels into blobs, and pick the strongest blob.
        count, labels, _, _ = cv2.connectedComponentsWithStats(hits.astype(np.uint8))
        strength = np.bincount(labels.ravel(), weights=(amplitude * hits).ravel(), minlength=count)
        strength[0] = 0  # label 0 is the background, ignore it
        best = int(np.argmax(strength))

        # 7. The beacon's center is the brightness weighted center of that blob.
        ys, xs = np.nonzero(labels == best)
        weights = amplitude[ys, xs]
        cx = np.average(xs, weights=weights)
        cy = np.average(ys, weights=weights)

        # Scale back up from the small image to full size pixels.
        return (cx + 0.5) * (w / SMALL_WIDTH), (cy + 0.5) * (h / small_h)


# Main loop: runs in the background, always grabbing and analyzing frames

latest_jpeg = None
jpeg_lock = threading.Lock()


def camera_loop():
    global latest_jpeg
    cam = open_camera()
    finder = BeaconFinder()
    frame_times = deque(maxlen=30)

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.monotonic()
        frame_times.append(now)
        target = finder.update(frame, now)

        # Draw a small white cross at the center of the frame. Later, the
        # gimbal's job will be to move the red crosshair onto this.
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        if target is not None:
            x, y = int(target[0]), int(target[1])
            cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
            cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            status = f"BEACON at ({x}, {y})"
        else:
            status = "searching"

        # Show the real frame rate, so we can confirm we're holding 30 fps.
        fps = 0.0
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        cv2.putText(frame, f"{status}   {fps:.0f} fps", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Compress to JPEG for the browser stream.
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with jpeg_lock:
                latest_jpeg = jpg.tobytes()


# Web stream, same idea as Argus's /video endpoint

app = FastAPI()


def mjpeg_stream():
    while True:
        with jpeg_lock:
            jpg = latest_jpeg
        if jpg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1 / 30)


@app.get("/video")
def video():
    return StreamingResponse(mjpeg_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)