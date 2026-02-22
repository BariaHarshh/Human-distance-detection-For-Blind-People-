# live_yolo_cam_debug.py
# Robust live camera with debug prints:
# - prints which weights are used
# - prints model.names
# - prints detections per frame
# - shows annotated frame

from ultralytics import YOLO
from pathlib import Path
import cv2, sys, time

ROOT = Path(r"D:\object-detection(project)")
RUNS_DIR = ROOT / "runs" / "detect"
CAM_INDEX = 0
IMGSZ = 640
CONF = 0.10   # lower for debugging; raise for less noise

def find_weight():
    # find latest run that actually contains best.pt or last.pt
    if not RUNS_DIR.exists():
        return None
    runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    for r in runs:
        w = r / "weights"
        if not w.exists(): 
            continue
        if (w / "best.pt").exists():
            return str((w / "best.pt").resolve())
        if (w / "last.pt").exists():
            return str((w / "last.pt").resolve())
    return None

# 1) Choose weights (auto) - you can replace with explicit path if you want

weights = r"D:\object-detection(project)\yolov8n.pt"

if weights:
    print("Using auto-found weights:", weights)
else:
    print("No trained weights found in runs/detect/* ; falling back to 'yolov8n.pt'")
    weights = "yolov8n.pt"

# 2) Load model with try/fallback
try:
    print("Loading model:", weights)
    t0 = time.time()
    model = YOLO(weights)
    print(f"Model loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print("Failed to load chosen weights:", e)
    print("Trying fallback 'yolov8n.pt' ...")
    model = YOLO("yolov8n.pt")

# 3) Print model class names
print("Model class names:", model.names)

# 4) Open camera
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Cannot open camera index {CAM_INDEX}. Try index 1 or 2.")
    sys.exit(1)

print("Camera opened — running. Press 'q' to quit.")

# 5) Loop: inference + prints + show
while True:
    ret, frame = cap.read()
    if not ret:
        print("Empty frame; exiting")
        break

    # Run inference with chosen confidence and imgsz
    results = model(frame, imgsz=IMGSZ, conf=CONF)[0]

    # Print detections
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print("No detections this frame")
    else:
        for b in boxes:
            cls_id = int(b.cls.cpu().numpy()[0])
            conf_v = float(b.conf.cpu().numpy()[0])
            xyxy = b.xyxy.cpu().numpy()[0].tolist()
            cls_name = model.names.get(cls_id, str(cls_id))
            print(f"Detected: {cls_name} (id={cls_id}) conf={conf_v:.3f} box={xyxy}")

    # Show annotated frame
    annotated = results.plot()
    cv2.imshow("YOLO Live Debug (press q to quit)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
