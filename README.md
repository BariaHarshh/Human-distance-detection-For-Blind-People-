# Object Detection & Navigation System for the Blind

A real-time object detection web application designed to assist visually impaired individuals by providing spatial awareness through audio feedback. Built with YOLOv8, Flask, and Vanilla JavaScript.

## 🚀 Features
- **Live Camera Mode**: Real-time detection with direction and distance estimation.
- **Image Upload Mode**: Analyze static images for objects.
- **Audio Navigation**: Text-to-Speech (TTS) alerts for detected objects (e.g., "Person center, near").
- **Distance Estimation**: Uses bounding box geometry to estimate proximity (Very Close, Near, Medium, Far).
- **Responsive UI**: Optimized for mobile and desktop browsers.

## 🛠️ Technology Stack
- **Backend**: Python 3.10, Flask, Gunicorn
- **AI Model**: YOLOv8 (Ultralytics)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Deployment**: Render (Web Service)

## 📂 Project Structure
```
OBJECT-DETECTION-PROJECT/
├── backend/                # Production code (API, UI, Weights)
│   ├── static/             # CSS & JS
│   ├── templates/          # HTML files
│   ├── app.py              # Flask API
│   ├── requirements.txt    # Dependencies
│   └── yolov8n.pt          # Model weights
├── docs/                   # Documentation & Test scripts
├── training/               # Training scripts, datasets, and experiments
├── render.yaml             # Render deployment config
└── README.md
```

## ⚙️ Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/BariaHarshh/Human-distance-detection-For-Blind-People-.git
   cd Human-distance-detection-For-Blind-People-
   ```
2. Navigate to backend:
   ```bash
   cd backend
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open `http://localhost:8000` in your browser.

## 🌐 Deployment (Render)
This project is configured for one-click deployment on Render using the `render.yaml` file.
1. Connect your GitHub repo to Render.
2. Render will automatically detect the configuration and deploy the service.

---
Developed by [Baria Harsh](https://github.com/BariaHarshh)