"""
app.py - Flask API for Crowd Monitoring & Intruder Detection
-------------------------------------------------------------
Endpoint:
  POST /predict  - Accept an image, return closest object + distance

Designed for deployment on Render (no webcam, no GUI, no pyttsx3).
"""

import io
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow frontend on Vercel to call this API

# ── Load YOLOv8 model ────────────────────────────────────────────────────────
# On Render, the model file must be in the same directory or downloaded at build
print("[API] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("[API] Model loaded successfully.")

# ── Target classes (COCO IDs) ────────────────────────────────────────────────
# Priority 1 = person/vehicles, 2 = furniture, 3 = small objects
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

# ── Distance zone thresholds ────────────────────────────────────────────────
# ratio = bounding_box_area / frame_area
ZONE_THRESHOLDS = [
    (0.02, "far"),         # < 2%  → far (~3+ metres)
    (0.08, "medium"),      # 2-8%  → medium (~1.5-3 metres)
    (0.20, "near"),        # 8-20% → near (~0.5-1.5 metres)
    (1.01, "very close"),  # 20%+  → very close (< 0.5 metres)
]


def get_distance_zone(box_area: int, frame_area: int) -> str:
    """Convert bounding-box/frame area ratio into a distance zone."""
    ratio = box_area / max(frame_area, 1)
    for threshold, zone_name in ZONE_THRESHOLDS:
        if ratio < threshold:
            return zone_name
    return "very close"


def get_direction(cx: int, frame_w: int) -> str:
    """Determine if object is left, ahead, or right."""
    if cx < frame_w // 3:
        return "left"
    if cx > (2 * frame_w) // 3:
        return "right"
    return "ahead"


def detect_objects(image: np.ndarray) -> dict:
    """
    Run YOLOv8 on the image and return detection results.

    Returns a dict with:
      - detected: bool
      - label: str (name of closest object)
      - distance: str (zone)
      - direction: str
      - confidence: float
      - object_count: int
      - person_count: int
      - is_crowd: bool
      - all_objects: list of all detections
    """
    h, w = image.shape[:2]
    frame_area = h * w

    # Run YOLO inference
    results = model(
        image,
        imgsz=640,
        conf=0.35,
        classes=CLASS_IDS,
        verbose=False,
    )[0]

    # Parse detections
    detections = []
    person_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue

        name, priority = TARGET_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = max((x2 - x1) * (y2 - y1), 1)
        cx = (x1 + x2) // 2
        conf = float(box.conf[0])

        if cls_id == 0:
            person_count += 1

        zone = get_distance_zone(area, frame_area)
        direction = get_direction(cx, w)

        detections.append({
            "label": name,
            "priority": priority,
            "distance": zone,
            "direction": direction,
            "confidence": round(conf, 3),
            "area": area,
        })

    # Sort: highest priority first, then closest (largest area)
    detections.sort(key=lambda d: (d["priority"], -d["area"]))

    if not detections:
        return {
            "detected": False,
            "label": None,
            "distance": None,
            "direction": None,
            "confidence": None,
            "object_count": 0,
            "person_count": 0,
            "is_crowd": False,
            "all_objects": [],
        }

    closest = detections[0]

    # Build all_objects list (without internal fields)
    all_objects = []
    for d in detections:
        all_objects.append({
            "label": d["label"],
            "distance": d["distance"],
            "direction": d["direction"],
            "confidence": d["confidence"],
        })

    return {
        "detected": True,
        "label": closest["label"],
        "distance": closest["distance"],
        "direction": closest["direction"],
        "confidence": closest["confidence"],
        "object_count": len(detections),
        "person_count": person_count,
        "is_crowd": person_count >= 3,
        "all_objects": all_objects,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Crowd Monitor API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an image and return detection results.

    Expects: multipart/form-data with field "image" containing an image file.
    Returns: JSON with detection results.
    """
    # Check if image was sent
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send an image file with key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select an image."}), 400

    try:
        # Read image bytes and decode with OpenCV
        image_bytes = file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image. Please send a valid image file (JPG/PNG)."}), 400

        # Run detection
        result = detect_objects(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
