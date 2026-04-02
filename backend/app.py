"""
app.py - Flask API for Real-Time Object Detection & Navigation
================================================================
Endpoints:
  GET  /           → Health check
  POST /predict    → Accept image (file upload OR base64), return detections

Distance calculation uses bbox HEIGHT / frame HEIGHT (more accurate).
Designed for deployment on Render.
"""

import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

# ── Model ────────────────────────────────────────────────────────────────────
print("[API] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("[API] Model loaded.")

# ── Target classes ───────────────────────────────────────────────────────────
TARGET_CLASSES = {
    0:  ("person",     1),
    1:  ("bicycle",    1),
    2:  ("car",        1),
    3:  ("motorcycle", 1),
    5:  ("bus",        1),
    7:  ("truck",      1),
    56: ("chair",      2),
    57: ("couch",      2),
    60: ("table",      2),
    39: ("bottle",     3),
    63: ("laptop",     3),
    67: ("phone",      3),
}
CLASS_IDS = list(TARGET_CLASSES.keys())

# ── Distance zones (bbox_height / frame_height) ─────────────────────────────
# height_ratio < 0.15  → far        (~3+ metres)
# 0.15 – 0.35          → medium     (~1.5–3 metres)
# 0.35 – 0.60          → near       (~0.5–1.5 metres)
# 0.60+                → very close (< 0.5 metres)

ZONE_THRESHOLDS = [
    (0.15, "far"),
    (0.35, "medium"),
    (0.60, "near"),
    (1.01, "very close"),
]


def get_zone(box_height: int, frame_height: int) -> str:
    ratio = box_height / max(frame_height, 1)
    for threshold, zone in ZONE_THRESHOLDS:
        if ratio < threshold:
            return zone
    return "very close"


def get_direction(cx: int, frame_w: int) -> str:
    if cx < frame_w // 3:
        return "left"
    if cx > (2 * frame_w) // 3:
        return "right"
    return "center"


def detect_objects(image: np.ndarray) -> dict:
    """Run YOLOv8 on image, return structured detection results."""
    h, w = image.shape[:2]

    results = model(
        image,
        imgsz=640,
        conf=0.25,
        classes=CLASS_IDS,
        verbose=False,
    )[0]

    detections = []
    person_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue

        name, priority = TARGET_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_h = max(y2 - y1, 1)
        cx = (x1 + x2) // 2
        conf = float(box.conf[0])

        if cls_id == 0:
            person_count += 1

        detections.append({
            "label":      name,
            "priority":   priority,
            "distance":   get_zone(box_h, h),
            "direction":  get_direction(cx, w),
            "confidence": round(conf, 3),
            "box_h":      box_h,
            "bbox":       [x1, y1, x2, y2],
        })

    # Sort: highest priority first, then closest (tallest bbox)
    detections.sort(key=lambda d: (d["priority"], -d["box_h"]))

    if not detections:
        return {
            "detected":     False,
            "label":        None,
            "distance":     None,
            "direction":    None,
            "confidence":   None,
            "object_count": 0,
            "person_count": 0,
            "is_crowd":     False,
            "all_objects":  [],
        }

    closest = detections[0]

    all_objects = []
    for d in detections:
        all_objects.append({
            "label":      d["label"],
            "distance":   d["distance"],
            "direction":  d["direction"],
            "confidence": d["confidence"],
            "bbox":       d["bbox"],
        })

    return {
        "detected":     True,
        "label":        closest["label"],
        "distance":     closest["distance"],
        "direction":    closest["direction"],
        "confidence":   closest["confidence"],
        "object_count": len(detections),
        "person_count": person_count,
        "is_crowd":     person_count >= 3,
        "all_objects":  all_objects,
    }


def decode_image(request_obj) -> np.ndarray:
    """
    Decode image from either:
      1. multipart/form-data  (field: "image")
      2. JSON body            (field: "image" as base64 data-URL or raw base64)
    Returns numpy BGR image or None.
    """
    # Method 1: File upload
    if "image" in request_obj.files:
        file = request_obj.files["image"]
        if file.filename != "":
            img_bytes = file.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Method 2: Base64 in JSON (used by live camera mode)
    data = request_obj.get_json(silent=True)
    if data and "image" in data:
        b64 = data["image"]
        # Strip data URL prefix if present: "data:image/jpeg;base64,..."
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Real-Time Object Detection API is running",
        "endpoints": {
            "POST /predict": "Send image (file or base64) to detect objects"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept image via file upload OR base64 JSON.
    Returns detection results as JSON.
    """
    try:
        image = decode_image(request)

        if image is None:
            return jsonify({
                "error": "No valid image provided. Send as file (key: 'image') or JSON {\"image\": \"base64...\"}."
            }), 400

        result = detect_objects(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
