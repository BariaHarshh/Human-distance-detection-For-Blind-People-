# Local Webcam Prototype

> [!WARNING]
> This folder contains a prototype that uses `cv2.VideoCapture(0)` to read directly from the machine's connected webcam. 
> 
> **This will NOT work when hosted on cloud platforms (Render, Heroku, AWS, etc.)** because the cloud server does not have access to your laptop's physical webcam.

If you want to run this locally:
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000` in your browser.

For cloud hosting, use the `backend` folder instead, which captures images via the browser (`navigator.mediaDevices.getUserMedia`) and sends them over HTTP.
