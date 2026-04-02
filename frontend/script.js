// =============================================================================
//  Real-Time Object Detection - Frontend Controller
//  Handles: Live camera, image upload, API calls, bounding boxes, TTS
// =============================================================================

// ── CONFIGURATION ───────────────────────────────────────────────────────────
const API_URL = "https://real-time-object-detection-yolo-qse9.onrender.com";

// How often to send frames in live mode (milliseconds)
const FRAME_INTERVAL = 800;  // ~1.2 frames/sec (safe for Render free tier)

// ── WAKE UP RENDER ON PAGE LOAD ─────────────────────────────────────────────
// Render free tier sleeps after inactivity. Ping it immediately on load
// so it's warm before the user clicks Start Camera.
(function pingBackend() {
    fetch(API_URL + "/", { method: "GET" })
        .then(() => console.log("[API] Backend is awake."))
        .catch(() => console.warn("[API] Backend waking up... may take 30-50s on first load."));
})();

// ── DOM ELEMENTS ────────────────────────────────────────────────────────────
const tabCamera     = document.getElementById("tabCamera");
const tabUpload     = document.getElementById("tabUpload");
const cameraPanel   = document.getElementById("cameraPanel");
const uploadPanel   = document.getElementById("uploadPanel");

// Camera
const webcam        = document.getElementById("webcam");
const captureCanvas = document.getElementById("captureCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const startBtn      = document.getElementById("startBtn");
const stopBtn       = document.getElementById("stopBtn");
const audioBtn      = document.getElementById("audioBtn");
const fpsValue      = document.getElementById("fpsValue");
const latencyValue  = document.getElementById("latencyValue");

// Upload
const uploadArea    = document.getElementById("uploadArea");
const fileInput     = document.getElementById("fileInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage  = document.getElementById("previewImage");
const clearBtn      = document.getElementById("clearBtn");
const detectBtn     = document.getElementById("detectBtn");
const btnText       = document.getElementById("btnText");
const spinner       = document.getElementById("spinner");

// Status + Results
const statusBar     = document.getElementById("statusBar");
const statusText    = document.getElementById("statusText");
const resultsSection = document.getElementById("resultsSection");
const primaryResult = document.getElementById("primaryResult");
const distanceBadge = document.getElementById("distanceBadge");
const resObject     = document.getElementById("resObject");
const resDistance    = document.getElementById("resDistance");
const resDirection  = document.getElementById("resDirection");
const resConfidence = document.getElementById("resConfidence");
const totalObjects  = document.getElementById("totalObjects");
const personCount   = document.getElementById("personCount");
const crowdStatus   = document.getElementById("crowdStatus");
const allObjectsPanel = document.getElementById("allObjectsPanel");
const objectsList   = document.getElementById("objectsList");

// ── STATE ───────────────────────────────────────────────────────────────────
let cameraStream    = null;
let frameLoop       = null;
let isSending       = false;    // prevent overlapping requests
let audioEnabled    = true;
let lastSpokenAlert = "";
let lastSpeakTime   = 0;
let selectedFile    = null;
let frameCount      = 0;
let fpsTimer        = null;

// ── TAB SWITCHING ───────────────────────────────────────────────────────────
function switchTab(mode) {
    if (mode === "camera") {
        cameraPanel.style.display = "block";
        uploadPanel.style.display = "none";
        tabCamera.classList.add("active");
        tabUpload.classList.remove("active");
        setStatus("Ready - Click 'Start Camera' to begin", "");
    } else {
        stopCamera();
        cameraPanel.style.display = "none";
        uploadPanel.style.display = "block";
        tabUpload.classList.add("active");
        tabCamera.classList.remove("active");
        setStatus("Ready - Upload an image to detect", "");
    }
    resultsSection.style.display = "none";
}

// ── STATUS HELPER ───────────────────────────────────────────────────────────
function setStatus(msg, type) {
    statusText.textContent = msg;
    statusBar.className = "status-bar" + (type ? " " + type : "");
}

// ── ZONE CSS CLASS ──────────────────────────────────────────────────────────
function zoneClass(distance) {
    if (!distance) return "";
    const d = distance.toLowerCase().replace(" ", "-");
    if (d === "very-close") return "zone-very-close";
    if (d === "near")       return "zone-near";
    if (d === "medium")     return "zone-medium";
    if (d === "far")        return "zone-far";
    return "";
}

// Zone color for canvas drawing
function zoneColor(distance) {
    if (!distance) return "#aaa";
    switch (distance.toLowerCase()) {
        case "far":        return "#FFF176";
        case "medium":     return "#FFC107";
        case "near":       return "#FF9800";
        case "very close": return "#F44336";
        default:           return "#aaa";
    }
}


// =============================================================================
//  LIVE CAMERA MODE
// =============================================================================

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: 640, height: 480 }
        });
        webcam.srcObject = cameraStream;
        startBtn.style.display = "none";
        stopBtn.style.display = "inline-flex";
        setStatus("Camera active - Detecting objects...", "live");
        resultsSection.style.display = "none";

        // Start FPS counter
        frameCount = 0;
        fpsTimer = setInterval(() => {
            fpsValue.textContent = frameCount;
            frameCount = 0;
        }, 1000);

        // Wait for video to be ready then start sending frames
        webcam.onloadedmetadata = () => {
            captureCanvas.width  = webcam.videoWidth;
            captureCanvas.height = webcam.videoHeight;
            overlayCanvas.width  = webcam.videoWidth;
            overlayCanvas.height = webcam.videoHeight;
            frameLoop = setInterval(sendFrame, FRAME_INTERVAL);
        };

    } catch (err) {
        console.error("Camera error:", err);
        if (err.name === "NotAllowedError") {
            setStatus("Camera access denied. Please allow camera permission.", "error");
        } else {
            setStatus("Cannot access camera: " + err.message, "error");
        }
    }
}

function stopCamera() {
    if (frameLoop) {
        clearInterval(frameLoop);
        frameLoop = null;
    }
    if (fpsTimer) {
        clearInterval(fpsTimer);
        fpsTimer = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    webcam.srcObject = null;

    // Clear overlay
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    startBtn.style.display = "inline-flex";
    stopBtn.style.display = "none";
    fpsValue.textContent = "0";
    latencyValue.textContent = "0";
    setStatus("Camera stopped", "");
}

async function sendFrame() {
    if (isSending) return;  // skip if previous request still pending
    if (!cameraStream) return;

    isSending = true;
    const t0 = performance.now();

    try {
        // Capture frame from video to canvas
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);

        // Convert to JPEG base64
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

        // Send to backend (60s timeout handles Render cold start)
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 60000);
        const resp = await fetch(API_URL + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
            signal: controller.signal,
        });
        clearTimeout(timeout);

        if (!resp.ok) throw new Error("Server error " + resp.status);

        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        const latency = Math.round(performance.now() - t0);
        latencyValue.textContent = latency;
        frameCount++;

        // Draw bounding boxes on overlay
        drawBoxes(data);

        // Update results panel
        displayResults(data, true);

        // Speak alert (browser TTS)
        if (audioEnabled && data.detected) {
            speakAlert(data);
        }

    } catch (err) {
        console.error("Frame error:", err);
        if (err.name === "AbortError") {
            setStatus("Backend timeout - Render may be waking up. Try again in 10 seconds.", "error");
        } else if (err.message.includes("Failed to fetch")) {
            setStatus("Waking up backend (Render free tier)... Please wait 30s then try again.", "error");
        }
    } finally {
        isSending = false;
    }
}

// ── DRAW BOUNDING BOXES ON OVERLAY CANVAS ───────────────────────────────────
function drawBoxes(data) {
    const ctx = overlayCanvas.getContext("2d");
    const cw = overlayCanvas.width;
    const ch = overlayCanvas.height;
    ctx.clearRect(0, 0, cw, ch);

    if (!data.all_objects) return;

    data.all_objects.forEach((obj, i) => {
        if (!obj.bbox) return;
        const [x1, y1, x2, y2] = obj.bbox;
        const color = zoneColor(obj.distance);

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = i === 0 ? 3 : 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label background
        const label = `${obj.label}  ${obj.distance}  ${obj.direction}`;
        ctx.font = "bold 13px Segoe UI, sans-serif";
        const tw = ctx.measureText(label).width;
        const labelY = Math.max(y1 - 4, 18);

        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY - 15, tw + 10, 19);

        // Label text
        ctx.fillStyle = "#000";
        ctx.fillText(label, x1 + 5, labelY - 1);
    });
}


// =============================================================================
//  UPLOAD MODE
// =============================================================================

uploadArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    uploadArea.style.display = "block";
    detectBtn.disabled = true;
    resultsSection.style.display = "none";
    setStatus("Ready - Upload an image to detect", "");
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        setStatus("Please select an image file (JPG, PNG)", "error");
        return;
    }
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = "block";
        uploadArea.style.display = "none";
    };
    reader.readAsDataURL(file);

    detectBtn.disabled = false;
    resultsSection.style.display = "none";
    setStatus("Image loaded - Click 'Detect Objects'", "");
}

async function detectUpload() {
    if (!selectedFile) return;

    detectBtn.disabled = true;
    btnText.textContent = "Detecting...";
    spinner.style.display = "block";
    setStatus("Detecting objects...", "detecting");
    resultsSection.style.display = "none";

    try {
        const formData = new FormData();
        formData.append("image", selectedFile);

        const resp = await fetch(API_URL + "/predict", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.error || "Server error " + resp.status);
        }

        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        displayResults(data, false);

        if (audioEnabled && data.detected) {
            speakAlert(data);
        }

    } catch (err) {
        console.error("Detection error:", err);
        if (err.message.includes("Failed to fetch")) {
            setStatus("Cannot connect to server. Is the backend running?", "error");
        } else {
            setStatus("Error: " + err.message, "error");
        }
    } finally {
        detectBtn.disabled = false;
        btnText.textContent = "Detect Objects";
        spinner.style.display = "none";
    }
}


// =============================================================================
//  DISPLAY RESULTS
// =============================================================================

function displayResults(data, isLive) {
    resultsSection.style.display = "block";

    if (!data.detected) {
        if (!isLive) setStatus("No objects detected in this image", "empty");
        else setStatus("Camera active - No objects detected", "live");

        resObject.textContent = "-";
        resDistance.textContent = "-";
        resDirection.textContent = "-";
        resConfidence.textContent = "-";
        distanceBadge.textContent = "None";
        distanceBadge.className = "badge";
        primaryResult.className = "primary-result";
        totalObjects.textContent = "0";
        personCount.textContent = "0";
        crowdStatus.textContent = "No";
        crowdStatus.className = "summary-number";
        allObjectsPanel.style.display = "none";
        return;
    }

    const zc = zoneClass(data.distance);

    if (!isLive) {
        setStatus(
            `Detected ${data.object_count} object${data.object_count > 1 ? "s" : ""} - Closest: ${data.label} (${data.distance})`,
            "success"
        );
    } else {
        setStatus(
            `Live: ${data.label} ${data.direction}, ${data.distance} | ${data.object_count} objects`,
            "live"
        );
    }

    // Primary result
    resObject.textContent     = data.label;
    resDistance.textContent    = data.distance;
    resDirection.textContent  = data.direction;
    resConfidence.textContent = (data.confidence * 100).toFixed(1) + "%";

    distanceBadge.textContent = data.distance;
    distanceBadge.className   = "badge " + zc;
    primaryResult.className   = "primary-result " + zc;

    // Summary
    totalObjects.textContent = data.object_count;
    personCount.textContent  = data.person_count;
    crowdStatus.textContent  = data.is_crowd ? "Yes!" : "No";
    crowdStatus.className    = "summary-number" + (data.is_crowd ? " crowd-yes" : "");

    // All objects list
    if (data.all_objects && data.all_objects.length > 1) {
        allObjectsPanel.style.display = "block";
        objectsList.innerHTML = "";
        data.all_objects.forEach(obj => {
            const row = document.createElement("div");
            row.className = "object-row " + zoneClass(obj.distance);
            row.innerHTML = `
                <span class="object-name">${obj.label}</span>
                <span class="object-info">${obj.distance} &middot; ${obj.direction} &middot; ${(obj.confidence * 100).toFixed(0)}%</span>
            `;
            objectsList.appendChild(row);
        });
    } else {
        allObjectsPanel.style.display = "none";
    }
}


// =============================================================================
//  TEXT-TO-SPEECH (Browser Web Speech API)
// =============================================================================

function speakAlert(data) {
    if (!("speechSynthesis" in window)) return;

    // Build alert text
    let text = "";
    if (data.is_crowd) text += "Crowd detected. ";

    if (data.distance === "very close") {
        text += `Warning! ${data.label} very close ${data.direction}.`;
    } else {
        text += `${data.label} ${data.direction}, ${data.distance}.`;
    }

    // Cooldown: don't repeat same alert within 2 seconds
    const now = Date.now();
    if (text === lastSpokenAlert && (now - lastSpeakTime) < 2000) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);

    lastSpokenAlert = text;
    lastSpeakTime = now;
}

function toggleAudio() {
    audioEnabled = !audioEnabled;
    audioBtn.textContent = audioEnabled ? "Audio: ON" : "Audio: OFF";
    audioBtn.classList.toggle("audio-off", !audioEnabled);

    if (!audioEnabled) {
        window.speechSynthesis.cancel();
    }
}
