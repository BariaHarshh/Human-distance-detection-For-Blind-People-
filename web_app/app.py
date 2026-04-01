"""
app.py  –  Flask backend for the Crowd Monitor web app
-------------------------------------------------------
Routes:
  GET  /              → main page (index.html)
  GET  /video_feed    → MJPEG stream (live annotated camera feed)
  GET  /state         → JSON: latest detection data
  POST /toggle_audio  → flip the audio-enabled flag

Architecture:
  A background thread continuously reads camera frames and runs YOLO.
  The latest annotated frame is stored in a shared variable.
  /video_feed serves that frame to the browser as a MJPEG stream.
  /state returns JSON so the browser can update the info panel via JS.
"""

import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

from detector import MultiObjectDetector

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
# Model sits one level up from web_app/; adjust if you move the .pt file.
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Detector ─────────────────────────────────────────────────────────────────
# Multi-object: detects person, car, bus, chair, table, bottle, etc.
detector = MultiObjectDetector(
    model_path = str(MODEL_PATH),
    conf       = 0.35,
    smooth_n   = 5,
)

# ── Camera ───────────────────────────────────────────────────────────────────
# cv2.CAP_DSHOW is the best backend on Windows (avoids 5-second init delay)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS,          30)

# ── Shared state (thread-safe via locks) ─────────────────────────────────────
_frame_lock  = threading.Lock()   # guards the latest annotated frame
_state_lock  = threading.Lock()   # guards the latest detection dict

_latest_frame = None              # bytes: JPEG-encoded annotated frame
_app_state    = {
    "detected":  False,
    "count":     0,
    "top_name":  None,            # name of closest/most important object
    "zone":      None,
    "direction": None,
    "alert":     "",              # natural-language alert string for TTS
    "fps":       0.0,
    "audio":     True,            # toggled by /toggle_audio
}


# ── Background capture thread ─────────────────────────────────────────────────
def _capture_loop():
    """
    Runs forever in a daemon thread.
    Reads camera frames → runs YOLO → stores the encoded JPEG + detection info.
    """
    global _latest_frame, _app_state

    while True:
        ok, frame = camera.read()
        if not ok:
            # Camera hiccup – wait a moment and retry
            time.sleep(0.05)
            continue

        # Run detection
        vis, info = detector.process(frame)

        # Encode to JPEG (quality 82 = good trade-off between size and clarity)
        ret, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue

        with _frame_lock:
            _latest_frame = buf.tobytes()

        with _state_lock:
            _app_state.update(info)   # update zone / direction / fps / count


# Start the capture thread before the first request
_thread = threading.Thread(target=_capture_loop, daemon=True)
_thread.start()


# ── MJPEG stream generator ────────────────────────────────────────────────────
def _gen_frames():
    """
    Generator that yields MJPEG chunks from the latest processed frame.
    Flask's Response wraps this for the browser.
    """
    while True:
        with _frame_lock:
            frame_bytes = _latest_frame

        if frame_bytes is None:
            # Nothing ready yet – tiny sleep to avoid busy-loop
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )
        # Cap stream at ~30 FPS to the browser (camera may run faster)
        time.sleep(0.033)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Live annotated MJPEG camera stream."""
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def state():
    """Return current detection data as JSON. Polled by the browser JS."""
    with _state_lock:
        return jsonify(dict(_app_state))


@app.route("/toggle_audio", methods=["POST"])
def toggle_audio():
    """Toggle the audio flag. Browser JS checks this before speaking."""
    with _state_lock:
        _app_state["audio"] = not _app_state["audio"]
        return jsonify({"audio": _app_state["audio"]})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Crowd Monitor  –  starting on  http://localhost:5000")
    print("=" * 55 + "\n")
    # threaded=True lets Flask handle /video_feed and /state concurrently
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
