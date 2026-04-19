"""
detector.py  (web_app version)
──────────────────────────────────────────────────────────────────────────────
Multi-object detector for the Flask web app.
Identical logic to smart_detector.py but WITHOUT pyttsx3/beep —
audio is handled by the browser (Web Speech API + Web Audio API).
──────────────────────────────────────────────────────────────────────────────
"""

import time
from collections import deque

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  TARGET CLASSES  (COCO IDs)
#  Priority 1 = high (person / vehicle)  →  announced first
#  Priority 2 = medium (furniture)
#  Priority 3 = low (small objects)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES = {
    0:  ("person",     1, "person"),
    1:  ("bicycle",    1, "bicycle"),
    2:  ("car",        1, "car"),
    3:  ("motorcycle", 1, "motorcycle"),
    5:  ("bus",        1, "bus"),
    7:  ("truck",      1, "truck"),
    56: ("chair",      2, "chair"),
    57: ("couch",      2, "couch"),
    60: ("table",      2, "table"),
    39: ("bottle",     3, "bottle"),
    63: ("laptop",     3, "laptop"),
    67: ("phone",      3, "phone"),
}

CLASS_IDS = list(TARGET_CLASSES.keys())

# Distance zone ratio thresholds (box_area / frame_area)
ZONE_THRESHOLDS = [
    (0.03, "FAR"),
    (0.10, "MEDIUM"),
    (0.25, "NEAR"),
    (1.01, "VERY CLOSE"),
]

# Bounding box BGR colours by priority
PRIORITY_BGR = {
    1: (0,   50, 255),   # red   — person/vehicle
    2: (0,  165, 255),   # orange — furniture
    3: (0,  210, 100),   # green  — small objects
}

# Alert cooldowns (seconds) — same logic as smart_detector.py
SAME_COOLDOWN = 4.0
DIFF_COOLDOWN = 1.5


def _zone(box_area: int, frame_area: int) -> str:
    ratio = box_area / max(frame_area, 1)
    for threshold, name in ZONE_THRESHOLDS:
        if ratio < threshold:
            return name
    return "VERY CLOSE"


def _direction(cx: int, frame_w: int) -> str:
    if cx < frame_w // 3:
        return "LEFT"
    if cx > (2 * frame_w) // 3:
        return "RIGHT"
    return "AHEAD"


def _mode(hist):
    if not hist:
        return None
    lst = list(hist)
    return max(set(lst), key=lst.count)


class MultiObjectDetector:
    """
    Drop-in replacement for the old PersonDetector.
    process(frame) → (annotated_frame, info_dict)

    info_dict keys:
      detected  – bool
      count     – int  (total detected objects)
      top_name  – str | None  (display name of closest/most important object)
      zone      – str | None  (smoothed distance zone)
      direction – str | None  (smoothed direction)
      alert     – str         (natural-language alert for TTS)
      fps       – float
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf: float = 0.35, smooth_n: int = 5):
        print(f"[Detector] Loading model: {model_path}")
        self.model    = YOLO(model_path)
        self.conf     = conf

        self._zone_hist = deque(maxlen=smooth_n)
        self._dir_hist  = deque(maxlen=smooth_n)

        self._last_alert      = ""
        self._last_alert_time = 0.0

        self._t_prev = time.perf_counter()
        self.fps     = 0.0
        print("[Detector] Ready.")

    def _tick_fps(self):
        now          = time.perf_counter()
        self.fps     = round(1.0 / max(now - self._t_prev, 1e-6), 1)
        self._t_prev = now

    def _build_alert(self, top: dict, person_count: int) -> str:
        spoken = top["spoken"]
        zone   = top["zone"].lower()
        dirn   = top["direction"].lower()

        parts = []
        if person_count >= 3:
            parts.append("Crowd detected. Be careful.")
        if top["zone"] == "VERY CLOSE":
            parts.append(f"Warning! {spoken} very close {dirn}.")
        else:
            parts.append(f"{spoken} {dirn}, {zone}.")
        return " ".join(parts)

    def _alert_changed(self, alert_text: str) -> bool:
        now = time.perf_counter()
        if alert_text != self._last_alert:
            return (now - self._last_alert_time) >= DIFF_COOLDOWN
        return (now - self._last_alert_time) >= SAME_COOLDOWN

    def process(self, frame):
        h, w       = frame.shape[:2]
        frame_area = h * w
        self._tick_fps()

        # ── YOLO inference ───────────────────────────────────────────────────
        results = self.model(
            frame,
            imgsz   = 640,
            conf    = self.conf,
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
            area = max((x2 - x1) * (y2 - y1), 1)
            cx   = (x1 + x2) // 2

            if cls_id == 0:
                person_count += 1

            detections.append({
                "cls_id":   cls_id,
                "name":     name,
                "spoken":   spoken,
                "priority": priority,
                "conf":     float(box.conf[0]),
                "box":      (x1, y1, x2, y2),
                "area":     area,
                "cx":       cx,
                "zone":     _zone(area, frame_area),
                "direction": _direction(cx, w),
            })

        # ── Rank: priority first, then closest (largest area) ────────────────
        detections.sort(key=lambda d: (d["priority"], -d["area"]))

        # ── Smooth top-1 zone & direction ────────────────────────────────────
        if detections:
            top1 = detections[0]
            self._zone_hist.append(top1["zone"])
            self._dir_hist.append(top1["direction"])
            top1["zone"]      = _mode(self._zone_hist)
            top1["direction"] = _mode(self._dir_hist)
        else:
            self._zone_hist.clear()
            self._dir_hist.clear()

        # ── Build alert text ─────────────────────────────────────────────────
        alert_text = ""
        if detections:
            alert_text = self._build_alert(detections[0], person_count)
            if self._alert_changed(alert_text):
                self._last_alert      = alert_text
                self._last_alert_time = time.perf_counter()
            else:
                # Alert hasn't changed enough — don't push it to browser TTS
                alert_text = self._last_alert   # keep last for display only
        else:
            if self._last_alert:
                self._last_alert = ""

        # ── Draw ─────────────────────────────────────────────────────────────
        vis = frame.copy()

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["box"]
            color = PRIORITY_BGR.get(d["priority"], (180, 180, 180))
            thick = 3 if i == 0 else 2
            label = f"{d['name']}  {d['zone']}  {d['direction']}  {d['conf']:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty = max(y1 - 8, th + 6)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(vis, label, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # HUD
        cv2.putText(vis, f"FPS: {self.fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)
        cv2.putText(vis, f"Objects: {len(detections)}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 255), 2)
        if detections:
            t = detections[0]
            cv2.putText(vis,
                        f"TOP  {t['name']}  |  {t['zone']}  |  {t['direction']}",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 230, 0), 2)

        # Alert banner
        if alert_text:
            cv2.rectangle(vis, (0, h - 38), (w, h), (0, 0, 0), -1)
            cv2.putText(vis, alert_text, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)

        return vis, {
            "detected":  len(detections) > 0,
            "count":     len(detections),
            "top_name":  detections[0]["name"] if detections else None,
            "zone":      detections[0]["zone"] if detections else None,
            "direction": detections[0]["direction"] if detections else None,
            "alert":     alert_text,
            "fps":       self.fps,
        }
