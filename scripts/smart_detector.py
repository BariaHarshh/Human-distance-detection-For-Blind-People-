"""
smart_detector.py
═════════════════════════════════════════════════════════════════════════════
Multi-Object Assistive Detection System for Visually Impaired Users
─────────────────────────────────────────────────────────────────────────────
What this script does:
  ✓ Detects 12 important object classes (person, car, bus, chair, table, etc.)
  ✓ Priority-based selection  → person/vehicle first, then furniture, then small
  ✓ Distance zones            → far / medium / near / very close
  ✓ Direction detection       → left / ahead / right
  ✓ Mode smoothing            → removes zone/direction flickering
  ✓ Smart TTS                 → speaks only when top object changes or after cooldown
  ✓ Background TTS thread     → doesn't block the camera loop
  ✓ Beep alerts               → frequency rises as object gets closer
  ✓ Crowd warning             → "Crowd detected" when 3+ persons visible
  ✓ Obstacle warning          → "Warning! … very close" for immediate danger
  ✓ FPS + HUD overlay         → for monitoring system performance

HOW TO RUN:
  pip install ultralytics opencv-python pyttsx3
  python smart_detector.py

Press Q to quit.
═════════════════════════════════════════════════════════════════════════════
"""

import queue
import threading
import time
from collections import deque

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — CONFIGURATION
#  Change these values to customise the system behaviour.
# ─────────────────────────────────────────────────────────────────────────────

# Path to your YOLOv8 weights file.
# yolov8n.pt = fastest (nano); yolov8s.pt = more accurate (small)
MODEL_PATH = r"D:\object-detection(project)\yolov8n.pt"

CAMERA_IDX   = 0      # 0 = default webcam; try 1 or 2 if this doesn't work
CONF         = 0.35   # confidence threshold (0.35 = good real-time balance)
IMG_SIZE     = 640    # YOLO input size in pixels (don't change unless needed)
SMOOTH_N     = 5      # number of frames used for zone/direction smoothing

# How long (in seconds) before the same alert is repeated
SAME_COOLDOWN = 4.0
# Minimum gap between any two different alerts
DIFF_COOLDOWN = 1.5


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — TARGET OBJECT CLASSES
#
#  These are the COCO dataset class IDs that YOLOv8 knows.
#  Format: { class_id: (display_name, priority, spoken_name) }
#
#  Priority 1 = HIGH   → person, vehicles (announce first, always)
#  Priority 2 = MEDIUM → furniture (announce if no high-priority object)
#  Priority 3 = LOW    → small objects (announce only if nothing else)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES = {
    # ── HIGH PRIORITY: moving/large hazards ──────────────────────────────────
    0:  ("person",      1, "person"),
    1:  ("bicycle",     1, "bicycle"),
    2:  ("car",         1, "car"),
    3:  ("motorcycle",  1, "motorcycle"),
    5:  ("bus",         1, "bus"),
    7:  ("truck",       1, "truck"),

    # ── MEDIUM PRIORITY: stationary furniture hazards ─────────────────────────
    56: ("chair",       2, "chair"),
    57: ("couch",       2, "couch"),
    60: ("table",       2, "table"),

    # ── LOW PRIORITY: small objects ───────────────────────────────────────────
    39: ("bottle",      3, "bottle"),
    63: ("laptop",      3, "laptop"),
    67: ("phone",       3, "phone"),
}

# List of class IDs passed to YOLO's classes= filter for speed
CLASS_IDS = list(TARGET_CLASSES.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — DISTANCE ZONE THRESHOLDS
#
#  We use:  ratio = bounding_box_area / total_frame_area
#  This automatically adapts to any camera resolution.
#
#  Calibration guide (approximate, for 640×480 webcam):
#    ratio < 3%  → person is ~3+ metres away
#    3%–10%      → ~1.5 to 3 metres
#    10%–25%     → ~0.5 to 1.5 metres
#    25%+        → closer than 0.5 metres → danger!
# ─────────────────────────────────────────────────────────────────────────────

ZONE_THRESHOLDS = [
    (0.03, "far"),
    (0.10, "medium"),
    (0.25, "near"),
    (1.01, "very close"),   # catch-all for anything ≥ 25%
]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — COLOURS  (OpenCV uses BGR, not RGB)
# ─────────────────────────────────────────────────────────────────────────────

# Bounding box colours by priority
PRIORITY_BGR = {
    1: (0,   50, 255),   # red-ish  — high priority (person / vehicle)
    2: (0,  165, 255),   # orange   — medium priority (furniture)
    3: (0,  210, 100),   # green    — low priority (small objects)
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — TEXT-TO-SPEECH (TTS) — BACKGROUND THREAD
#
#  pyttsx3 is a blocking library, so we run it in a separate thread.
#  The main camera loop puts messages in _tts_q; this thread speaks them.
#  We always clear old messages so only the LATEST alert is spoken.
# ─────────────────────────────────────────────────────────────────────────────

_tts_q = queue.Queue()


def _tts_worker():
    """Runs in a background thread. Picks up text from _tts_q and speaks it."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   155)   # words per minute (slightly fast)
        engine.setProperty("volume", 1.0)   # maximum volume
    except Exception as e:
        print(f"[TTS] Cannot initialise pyttsx3: {e}")
        print("[TTS] Audio alerts will be disabled.")
        return

    while True:
        text = _tts_q.get()        # blocks until something arrives
        if text is None:           # None is the stop signal
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


# Start the TTS thread immediately (runs as daemon so it dies when main exits)
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text: str):
    """
    Non-blocking: put text in the TTS queue.
    Clears any pending (unspoken) messages first so only the latest is heard.
    """
    # Drop old queued messages (replace with the new one)
    while not _tts_q.empty():
        try:
            _tts_q.get_nowait()
        except queue.Empty:
            break
    _tts_q.put(text)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — BEEP SYSTEM  (Windows only via winsound)
#
#  Higher frequency = closer object = more urgent.
#  Duration decreases for closer objects (rapid urgent beeps).
# ─────────────────────────────────────────────────────────────────────────────

# zone → (frequency_hz, duration_ms)
BEEP_PARAMS = {
    "far":        (300,  400),
    "medium":     (500,  300),
    "near":       (750,  220),
    "very close": (1050, 150),
}


def beep(zone: str):
    """Play a warning beep. Silently skipped if winsound is not available."""
    try:
        import winsound
        freq, dur = BEEP_PARAMS.get(zone, (440, 250))
        winsound.Beep(freq, dur)
    except Exception:
        pass   # safe to ignore on non-Windows or if sound card unavailable


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_zone(box_area: int, frame_area: int) -> str:
    """Convert bounding-box / frame area ratio into a distance zone string."""
    ratio = box_area / max(frame_area, 1)
    for threshold, zone_name in ZONE_THRESHOLDS:
        if ratio < threshold:
            return zone_name
    return "very close"


def get_direction(cx: int, frame_w: int) -> str:
    """
    Divide the frame into thirds:
      left third   → "left"
      middle third → "ahead"
      right third  → "right"
    """
    if cx < frame_w // 3:
        return "left"
    if cx > (2 * frame_w) // 3:
        return "right"
    return "ahead"


def mode_of(history) -> str:
    """
    Return the most common value in a deque.
    Used for smoothing — avoids zone/direction flickering between frames.
    """
    if not history:
        return None
    lst = list(history)
    return max(set(lst), key=lst.count)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — SMART DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SmartDetector:
    """
    Multi-object assistive detector.

    Usage:
        detector = SmartDetector()
        annotated_frame, info = detector.process(frame)
    """

    def __init__(self):
        print(f"[SmartDetector] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        # Smoothing: keep last SMOOTH_N zone and direction values
        self._zone_hist = deque(maxlen=SMOOTH_N)
        self._dir_hist  = deque(maxlen=SMOOTH_N)

        # Cooldown tracking
        self._last_alert      = ""
        self._last_alert_time = 0.0

        # FPS tracking
        self._t_prev = time.perf_counter()
        self.fps     = 0.0

        print("[SmartDetector] Ready. Starting camera loop...\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tick_fps(self):
        """Compute and store current FPS."""
        now          = time.perf_counter()
        self.fps     = round(1.0 / max(now - self._t_prev, 1e-6), 1)
        self._t_prev = now

    def _rank_detections(self, detections: list) -> list:
        """
        Sort detections so the most important & closest object is first.
        Rule: lower priority number wins; ties broken by largest area (closest).
        """
        detections.sort(key=lambda d: (d["priority"], -d["area"]))
        return detections

    def _build_alert(self, top: dict, person_count: int) -> str:
        """
        Build a natural-language alert string for the top object.
        Examples:
          "person ahead, near."
          "Warning! car very close left."
          "Crowd detected. Be careful. person very close ahead."
        """
        spoken = top["spoken"]
        zone   = top["zone"]
        dirn   = top["direction"]

        parts = []

        # Crowd warning (3 or more people in frame)
        if person_count >= 3:
            parts.append("Crowd detected. Be careful.")

        # Proximity-based announcement
        if zone == "very close":
            parts.append(f"Warning! {spoken} very close {dirn}.")
        else:
            parts.append(f"{spoken} {dirn}, {zone}.")

        return " ".join(parts)

    def _should_speak(self, alert_text: str) -> bool:
        """
        Returns True if we should actually speak this alert.
        Logic:
          - New message → wait only DIFF_COOLDOWN seconds
          - Same message → wait SAME_COOLDOWN seconds before repeating
        """
        now = time.perf_counter()
        if alert_text != self._last_alert:
            return (now - self._last_alert_time) >= DIFF_COOLDOWN
        return (now - self._last_alert_time) >= SAME_COOLDOWN

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Run detection on one video frame.

        Parameters
        ----------
        frame : numpy.ndarray  (BGR image from cv2.read())

        Returns
        -------
        vis  : annotated frame with bounding boxes + HUD overlay
        info : dict  {detected, count, top_name, zone, direction, alert, fps}
        """
        h, w       = frame.shape[:2]
        frame_area = h * w
        self._tick_fps()

        # ── Step 1: Run YOLO inference ────────────────────────────────────────
        # classes=CLASS_IDS → only detect our target objects (faster)
        # verbose=False     → suppress per-frame console output
        results = self.model(
            frame,
            imgsz   = IMG_SIZE,
            conf    = CONF,
            classes = CLASS_IDS,
            verbose = False,
        )[0]

        # ── Step 2: Parse every detected bounding box ─────────────────────────
        detections   = []
        person_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue   # skip anything not in our list

            name, priority, spoken = TARGET_CLASSES[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area  = max((x2 - x1) * (y2 - y1), 1)  # box pixel area
            cx    = (x1 + x2) // 2                   # box centre x

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
                "zone":     get_zone(area, frame_area),
                "direction": get_direction(cx, w),
            })

        # ── Step 3: Rank by priority + proximity ──────────────────────────────
        detections = self._rank_detections(detections)

        # ── Step 4: Smooth zone & direction for the top object ────────────────
        if detections:
            top1 = detections[0]
            self._zone_hist.append(top1["zone"])
            self._dir_hist.append(top1["direction"])
            # Replace raw values with smoothed (mode) values
            top1["zone"]      = mode_of(self._zone_hist)
            top1["direction"] = mode_of(self._dir_hist)
        else:
            # Nothing detected → clear history so next detection starts fresh
            self._zone_hist.clear()
            self._dir_hist.clear()

        # ── Step 5: Build alert + decide whether to speak ─────────────────────
        alert_text = ""

        if detections:
            alert_text = self._build_alert(detections[0], person_count)

            if self._should_speak(alert_text):
                speak(alert_text)           # queue for TTS thread
                beep(detections[0]["zone"]) # play beep sound
                self._last_alert      = alert_text
                self._last_alert_time = time.perf_counter()

        else:
            # When frame becomes empty, reset so next detection is announced
            if self._last_alert != "":
                self._last_alert = ""
                # Uncomment the line below to announce when area clears:
                # speak("Area is clear.")

        # ── Step 6: Draw bounding boxes & labels ──────────────────────────────
        vis = frame.copy()

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["box"]
            color = PRIORITY_BGR.get(d["priority"], (180, 180, 180))
            thick = 3 if i == 0 else 2   # thicker box for the top (closest) object
            label = f"{d['name']}  {d['zone']}  {d['direction']}  {d['conf']:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

            # Draw a filled label background so text is readable on any scene
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty = max(y1 - 8, th + 6)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(vis, label, (x1 + 3, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # ── Step 7: HUD (heads-up display) in top-left corner ─────────────────
        hud = [
            f"FPS: {self.fps:.1f}",
            f"Objects: {len(detections)}",
        ]
        if detections:
            t = detections[0]
            hud.append(f"TOP  {t['name']}  |  {t['zone']}  |  {t['direction']}")

        for i, line in enumerate(hud):
            y_pos = 28 + i * 28
            color = (0, 230, 0) if i == 0 else (0, 225, 255)
            cv2.putText(vis, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

        # ── Step 8: Alert banner at the bottom of the frame ───────────────────
        if alert_text:
            cv2.rectangle(vis, (0, h - 38), (w, h), (0, 0, 0), -1)
            cv2.putText(vis, alert_text, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        else:
            cv2.rectangle(vis, (0, h - 38), (w, h), (30, 30, 30), -1)
            cv2.putText(vis, "No objects detected — area is clear", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)

        return vis, {
            "detected":  len(detections) > 0,
            "count":     len(detections),
            "top_name":  detections[0]["name"] if detections else None,
            "zone":      detections[0]["zone"] if detections else None,
            "direction": detections[0]["direction"] if detections else None,
            "alert":     alert_text,
            "fps":       self.fps,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 — MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Crowd Monitor — Multi-Object Assistive Detection System")
    print("  Classes: person, car, bus, truck, bicycle, motorcycle,")
    print("           chair, couch, table, bottle, laptop, phone")
    print("  Press Q to quit.")
    print("=" * 62 + "\n")

    detector = SmartDetector()

    # Open camera
    # cv2.CAP_DSHOW = DirectShow backend on Windows → faster startup
    cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_IDX}.")
        print("        Try changing CAMERA_IDX to 1 or 2 at the top of this file.")
        return

    print("Camera open. Running... Press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARNING] Empty frame — retrying...")
            time.sleep(0.05)
            continue

        # Run detection and get annotated frame + info dict
        vis, info = detector.process(frame)

        # Show the annotated frame in a window
        cv2.imshow("Crowd Monitor  —  Multi-Object  (press Q to quit)", vis)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    _tts_q.put(None)   # send stop signal to TTS thread
    print("\nDetector stopped. Goodbye.")


if __name__ == "__main__":
    main()
