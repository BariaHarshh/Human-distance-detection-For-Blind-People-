// ── Configuration ───────────────────────────────────────────────────────────
// IMPORTANT: Replace this with your actual Render backend URL after deploying.
// Example: "https://your-app-name.onrender.com"
const API_URL = "https://YOUR-BACKEND.onrender.com";

// ── DOM Elements ────────────────────────────────────────────────────────────
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const clearBtn = document.getElementById("clearBtn");
const detectBtn = document.getElementById("detectBtn");
const btnText = document.getElementById("btnText");
const spinner = document.getElementById("spinner");
const statusBar = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");
const resultsSection = document.getElementById("resultsSection");

// Result elements
const objectName = document.getElementById("objectName");
const objectDistance = document.getElementById("objectDistance");
const objectDirection = document.getElementById("objectDirection");
const objectConfidence = document.getElementById("objectConfidence");
const distanceBadge = document.getElementById("distanceBadge");
const primaryResult = document.getElementById("primaryResult");
const totalObjects = document.getElementById("totalObjects");
const personCount = document.getElementById("personCount");
const crowdStatus = document.getElementById("crowdStatus");
const allObjectsPanel = document.getElementById("allObjectsPanel");
const objectsList = document.getElementById("objectsList");

// State
let selectedFile = null;

// ── Upload Area Events ──────────────────────────────────────────────────────

// Click to open file picker
uploadArea.addEventListener("click", () => fileInput.click());

// File selected via picker
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag & drop
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
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Clear button
clearBtn.addEventListener("click", () => {
    resetUI();
});

// Detect button
detectBtn.addEventListener("click", () => {
    if (selectedFile) {
        runDetection(selectedFile);
    }
});

// ── File Handling ───────────────────────────────────────────────────────────

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith("image/")) {
        setStatus("Please select an image file (JPG, PNG, WEBP)", "error");
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = "inline-block";
        uploadArea.style.display = "none";
    };
    reader.readAsDataURL(file);

    // Enable detect button
    detectBtn.disabled = false;

    // Hide previous results
    resultsSection.style.display = "none";
    setStatus("Image loaded - Click 'Detect Objects' to analyze", "");
}

function resetUI() {
    selectedFile = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    uploadArea.style.display = "block";
    detectBtn.disabled = true;
    resultsSection.style.display = "none";
    setStatus("Ready - Upload an image to start", "");
}

// ── Status Updates ──────────────────────────────────────────────────────────

function setStatus(message, type) {
    statusText.textContent = message;
    statusBar.className = "status-bar";
    if (type) {
        statusBar.classList.add(type);
    }
}

// ── Zone Helpers ────────────────────────────────────────────────────────────

function getZoneClass(distance) {
    if (!distance) return "";
    const d = distance.toLowerCase().replace(" ", "-");
    if (d === "very-close") return "zone-very-close";
    if (d === "near") return "zone-near";
    if (d === "medium") return "zone-medium";
    if (d === "far") return "zone-far";
    return "";
}

// ── Detection API Call ──────────────────────────────────────────────────────

async function runDetection(file) {
    // Show loading state
    detectBtn.disabled = true;
    btnText.textContent = "Detecting...";
    spinner.style.display = "block";
    setStatus("Detecting objects... Please wait", "detecting");
    resultsSection.style.display = "none";

    try {
        // Build form data
        const formData = new FormData();
        formData.append("image", file);

        // Call API
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server error (${response.status})`);
        }

        const data = await response.json();

        // Check for error in response
        if (data.error) {
            throw new Error(data.error);
        }

        // Display results
        displayResults(data);

    } catch (err) {
        console.error("Detection error:", err);

        if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
            setStatus("Cannot connect to server. Check if backend is running.", "error");
        } else {
            setStatus(`Error: ${err.message}`, "error");
        }
    } finally {
        // Reset button
        detectBtn.disabled = false;
        btnText.textContent = "Detect Objects";
        spinner.style.display = "none";
    }
}

// ── Display Results ─────────────────────────────────────────────────────────

function displayResults(data) {
    resultsSection.style.display = "block";

    if (!data.detected) {
        // No objects detected
        setStatus("No objects detected in this image", "empty");

        objectName.textContent = "-";
        objectDistance.textContent = "-";
        objectDirection.textContent = "-";
        objectConfidence.textContent = "-";
        distanceBadge.textContent = "None";
        distanceBadge.className = "badge";
        primaryResult.className = "primary-result";

        totalObjects.textContent = "0";
        personCount.textContent = "0";
        crowdStatus.textContent = "No";

        allObjectsPanel.style.display = "none";
        return;
    }

    // Success
    const zoneClass = getZoneClass(data.distance);

    setStatus(
        `Detected ${data.object_count} object${data.object_count > 1 ? "s" : ""} - Closest: ${data.label} (${data.distance})`,
        "success"
    );

    // Primary result
    objectName.textContent = data.label;
    objectDistance.textContent = data.distance;
    objectDirection.textContent = data.direction;
    objectConfidence.textContent = (data.confidence * 100).toFixed(1) + "%";

    // Badge
    distanceBadge.textContent = data.distance;
    distanceBadge.className = "badge " + zoneClass;

    // Card border color
    primaryResult.className = "primary-result " + zoneClass;

    // Summary
    totalObjects.textContent = data.object_count;
    personCount.textContent = data.person_count;
    crowdStatus.textContent = data.is_crowd ? "Yes!" : "No";
    if (data.is_crowd) {
        crowdStatus.style.color = "#F44336";
    } else {
        crowdStatus.style.color = "#1a1a1a";
    }

    // All objects list
    if (data.all_objects && data.all_objects.length > 1) {
        allObjectsPanel.style.display = "block";
        objectsList.innerHTML = "";

        data.all_objects.forEach((obj) => {
            const rowZone = getZoneClass(obj.distance);
            const row = document.createElement("div");
            row.className = "object-row " + rowZone;
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
