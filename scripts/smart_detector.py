"""
smart_detector.py
================================================================================
Real-Time Object Detection & Navigation System for Blind People
--------------------------------------------------------------------------------
Features:
  - Detects 12 object classes (person, car, bus, chair, table, etc.)
  - Priority-based selection: person/vehicle > furniture > small objects
  - Distance zones using bbox HEIGHT ratio (more accurate than area):
      far / medium / near / very close
  - Direction: left / center / right
  - 3-frame smoothing to prevent flickering
  - Smart TTS with cooldown (non-blocking background thread)
  - Beep alerts (frequency rises as object gets closer)
  - Crowd warning when 3+ persons visible
  - FPS + HUD overlay

HOW TO RUN:
  pip install ultralytics opencv-python pyttsx3
  python smart_detector.py

Press Q to quit.
================================================================================
"""

import queue
import threading
import time
from collections import deque

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = r"D:\object-detection(project)\yolov8n.pt"

CAMERA_IDX   = 0
CONF         = 0.35
IMG_SIZE     = 640
SMOOTH_N     = 3       # frames to average for smoothing

SAME_COOLDOWN = 3.0    # seconds before repeating same alert
DIFF_COOLDOWN = 1.0    # seconds before a new/different alert


# ─────────────────────────────────────────────────────────────────────────────
#  TARGET CLASSES  (COCO IDs)
#  Priority 1 = HIGH (person/vehicles) — announced first
#  Priority 2 = MEDIUM (furniture)
#  Priority 3 = LOW (small objects)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES = {
    0:  ("person",      1, "person"),
    1:  ("bicycle",     1, "bicycle"),
    2:  ("car",         1, "car"),
    3:  ("motorcycle",  1, "motorcycle"),
    5:  ("bus",         1, "bus"),
    7:  ("truck",       1, "truck"),
    56: ("chair",       2, "chair"),
    57: ("couch",       2, "couch"),
    60: ("table",       2, "table"),
    39: ("bottle",      3, "bottle"),
    63: ("laptop",      3, "laptop"),
    67: ("phone",       3, "phone"),
}

CLASS_IDS = list(TARGET_CLASSES.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  DISTANCE ZONES — using bbox_height / frame_height
#
#  This is MORE ACCURATE than area ratio because:
#  - Height correlates directly with real-world distance (perspective)
#  - Not affected by object width variations (wide bus vs narrow person)
#
#  Calibration (640x480 webcam, approximate):
#    height_ratio < 0.15  → object is 3+ metres away      → far
#    0.15 – 0.35          → object is 1.5–3 metres away    → medium
#    0.35 – 0.60          → object is 0.5–1.5 metres away  → near
#    0.60+                → object is < 0.5 metres away    → very close
# ─────────────────────────────────────────────────────────────────────────────

ZONE_THRESHOLDS = [
    (0.15, "far"),
    (0.35, "medium"),
    (0.60, "near"),
    (1.01, "very close"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

ZONE_BGR = {
    "far":        (0, 200, 0),     # green
    "medium":     (0, 220, 255),   # yellow
    "near":       (0, 140, 255),   # orange
    "very close": (0,  50, 255),   # red
}

PRIORITY_BGR = {
    1: (0,   50, 255),
    2: (0,  165, 255),
    3: (0,  210, 100),
}


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT-TO-SPEECH — BACKGROUND THREAD
# ─────────────────────────────────────────────────────────────────────────────

_tts_q = queue.Queue()


def _tts_worker():
    """Background thread: picks up text from _tts_q and speaks it."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   155)
        engine.setProperty("volume", 1.0)
    except Exception as e:
        print(f"[TTS] Cannot initialise pyttsx3: {e}")
        print("[TTS] Audio alerts disabled.")
        return

    while True:
        text = _tts_q.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text: str):
    """Non-blocking TTS. Clears old queued messages, pushes latest."""
    while not _tts_q.empty():
        try:
            _tts_q.get_nowait()
        except queue.Empty:
            break
    _tts_q.put(text)


# ─────────────────────────────────────────────────────────────────────────────
#  BEEP SYSTEM (Windows only)
# ─────────────────────────────────────────────────────────────────────────────

BEEP_PARAMS = {
    "far":        (300,  400),
    "medium":     (500,  300),
    "near":       (750,  220),
    "very close": (1050, 150),
}


def beep(zone: str):
    """Play a warning beep. Silently skipped on non-Windows."""
    try:
        import winsound
        freq, dur = BEEP_PARAMS.get(zone, (440, 250))
        winsound.Beep(freq, dur)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_zone(box_height: int, frame_height: int) -> str:
    """Convert bbox_height / frame_height ratio into a distance zone."""
    ratio = box_height / max(frame_height, 1)
    for threshold, zone_name in ZONE_THRESHOLDS:
        if ratio < threshold:
            return zone_name
    return "very close"


def get_direction(cx: int, frame_w: int) -> str:
    """Divide frame into thirds: left / center / right."""
    if cx < frame_w // 3:
        return "left"
    if cx > (2 * frame_w) // 3:
        return "right"
    return "center"


def mode_of(history) -> str:
    """Return most common value in a deque (for smoothing)."""
    if not history:
        return None
    lst = list(history)
    return max(set(lst), key=lst.count)


# ─────────────────────────────────────────────────────────────────────────────
#  SMART DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SmartDetector:
    """
    Real-time multi-object detector with distance + direction + TTS.

    Usage:
        detector = SmartDetector()
        annotated_frame, info = detector.process(frame)
    """

    def __init__(self):
        print(f"[SmartDetector] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        self._zone_hist = deque(maxlen=SMOOTH_N)
        self._dir_hist  = deque(maxlen=SMOOTH_N)

        self._last_alert      = ""
        self._last_alert_time = 0.0

        self._t_prev = time.perf_counter()
        self.fps     = 0.0

        # Track consecutive "no detection" frames to auto-clear smoothing
        self._empty_frames = 0

        print("[SmartDetector] Ready.\n")

    def _tick_fps(self):
        now          = time.perf_counter()
        self.fps     = round(1.0 / max(now - self._t_prev, 1e-6), 1)
        self._t_prev = now

    def _build_alert(self, top: dict, person_count: int) -> str:
        spoken = top["spoken"]
        zone   = top["zone"]
        dirn   = top["direction"]

        parts = []
        if person_count >= 3:
            parts.append("Crowd detected. Be careful.")

        if zone == "very close":
            parts.append(f"Warning! {spoken} very close {dirn}.")
        else:
            parts.append(f"{spoken} {dirn}, {zone}.")

        return " ".join(parts)

    def _should_speak(self, alert_text: str) -> bool:
        now = time.perf_counter()
        if alert_text != self._last_alert:
            return (now - self._last_alert_time) >= DIFF_COOLDOWN
        return (now - self._last_alert_time) >= SAME_COOLDOWN

    def process(self, frame):
        """
        Detect objects in one frame.

        Returns:
            vis  : annotated frame (numpy array)
            info : dict with detection data
        """
        h, w = frame.shape[:2]
        self._tick_fps()

        # ── YOLO inference ───────────────────────────────────────────────────
        results = self.model(
            frame,
            imgsz   = IMG_SIZE,
            conf    = CONF,
            classes = CLASS_IDS,
            verbose = False,
        )[0]

        # ── Parse detections ─────────────────────────────────────────────────
        detections   = []
        person_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue

            name, priority, spoken = TARGET_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_h  = max(y2 - y1, 1)
            box_area = max((x2 - x1) * box_h, 1)
            cx     = (x1 + x2) // 2

            if cls_id == 0:
                person_count += 1

            detections.append({
                "cls_id":    cls_id,
                "name":      name,
                "spoken":    spoken,
                "priority":  priority,
                "conf":      float(box.conf[0]),
                "box":       (x1, y1, x2, y2),
                "box_h":     box_h,
                "area":      box_area,
                "cx":        cx,
                "zone":      get_zone(box_h, h),
                "direction": get_direction(cx, w),
            })

        # ── Rank: priority first, then closest (tallest bbox = closest) ──────
        detections.sort(key=lambda d: (d["priority"], -d["box_h"]))

        # ── Smooth zone & direction for top object ───────────────────────────
        if detections:
            self._empty_frames = 0
            top1 = detections[0]
            self._zone_hist.append(top1["zone"])
            self._dir_hist.append(top1["direction"])
            top1["zone"]      = mode_of(self._zone_hist)
            top1["direction"] = mode_of(self._dir_hist)
        else:
            self._empty_frames += 1
            # Clear smoothing history after 3 empty frames
            # This prevents "stuck in very close" bug
            if self._empty_frames >= 3:
                self._zone_hist.clear()
                self._dir_hist.clear()

        # ── Build alert + speak ──────────────────────────────────────────────
        alert_text = ""

        if detections:
            alert_text = self._build_alert(detections[0], person_count)
            if self._should_speak(alert_text):
                speak(alert_text)
                beep(detections[0]["zone"])
                self._last_alert      = alert_text
                self._last_alert_time = time.perf_counter()
        else:
            if self._last_alert != "":
                self._last_alert = ""

        # ── Draw bounding boxes ──────────────────────────────────────────────
        vis = frame.copy()

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["box"]
            zone_color = ZONE_BGR.get(d["zone"], (180, 180, 180))
            thick = 3 if i == 0 else 2
            label = f"{d['name']}  {d['zone']}  {d['direction']}  {d['conf']:.0%}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), zone_color, thick)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            ty = max(y1 - 8, th + 6)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), zone_color, -1)
            cv2.putText(vis, label, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

        # ── HUD overlay ──────────────────────────────────────────────────────
        hud_lines = [
            f"FPS: {self.fps:.0f}",
            f"Objects: {len(detections)}   Persons: {person_count}",
        ]
        if detections:
            t = detections[0]
            hud_lines.append(f"CLOSEST: {t['name']}  |  {t['zone']}  |  {t['direction']}")

        for i, line in enumerate(hud_lines):
            y_pos = 28 + i * 28
            color = (0, 230, 0) if i == 0 else (0, 225, 255)
            cv2.putText(vis, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── Alert banner at bottom ───────────────────────────────────────────
        if alert_text:
            cv2.rectangle(vis, (0, h - 38), (w, h), (0, 0, 0), -1)
            cv2.putText(vis, alert_text, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)
        else:
            cv2.rectangle(vis, (0, h - 38), (w, h), (30, 30, 30), -1)
            cv2.putText(vis, "No objects detected", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

        return vis, {
            "detected":     len(detections) > 0,
            "count":        len(detections),
            "top_name":     detections[0]["name"] if detections else None,
            "zone":         detections[0]["zone"] if detections else None,
            "direction":    detections[0]["direction"] if detections else None,
            "alert":        alert_text,
            "person_count": person_count,
            "is_crowd":     person_count >= 3,
            "fps":          self.fps,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Real-Time Object Detection & Navigation System")
    print("  for Blind People")
    print("  ─────────────────────────────────────────────")
    print("  Objects: person, car, bus, truck, bicycle, motorcycle,")
    print("           chair, couch, table, bottle, laptop, phone")
    print("  Distance: bbox height ratio (3-frame smoothing)")
    print("  Press Q to quit.")
    print("=" * 60 + "\n")

    detector = SmartDetector()

    cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_IDX}. Try 1 or 2.")
        return

    print("Camera open. Running...\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        vis, info = detector.process(frame)
        cv2.imshow("Object Detection - Navigation System (Q to quit)", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    _tts_q.put(None)
    print("\nStopped. Goodbye.")


if __name__ == "__main__":
    main()
