/**
 * script.js  —  Crowd Monitor (Multi-Object) frontend logic
 * ──────────────────────────────────────────────────────────
 * 1. Polls /state every 600 ms for latest detection data
 * 2. Updates: object name, zone badge, direction arrows, FPS, object count
 * 3. Speaks alerts via Web Speech API  (uses server-built alert text)
 * 4. Plays beep via Web Audio API      (frequency = proximity urgency)
 * 5. Handles audio ON/OFF toggle
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const POLL_MS         = 600;    // poll /state every 600 ms
const BEEP_COOLDOWN   = 1200;   // ms between beeps

// ── Object icon map ───────────────────────────────────────────────────────────
// Maps detected object names to emoji icons shown in the UI
const OBJECT_ICONS = {
  person:     '🧍',
  bicycle:    '🚲',
  car:        '🚗',
  motorcycle: '🏍️',
  bus:        '🚌',
  truck:      '🚛',
  chair:      '🪑',
  couch:      '🛋️',
  table:      '🪞',
  bottle:     '🍶',
  laptop:     '💻',
  phone:      '📱',
};

// ── Beep frequency + duration per zone ───────────────────────────────────────
const BEEP_PARAMS = {
  'FAR':        { freq: 300,  dur: 0.40 },
  'MEDIUM':     { freq: 500,  dur: 0.30 },
  'NEAR':       { freq: 750,  dur: 0.22 },
  'VERY CLOSE': { freq: 1050, dur: 0.15 },
};

// ── Badge CSS class per zone ──────────────────────────────────────────────────
const BADGE_CLASS = {
  'FAR':        'badge-far',
  'MEDIUM':     'badge-medium',
  'NEAR':       'badge-near',
  'VERY CLOSE': 'badge-very-close',
};

// ── Status dot class per zone ─────────────────────────────────────────────────
const DOT_CLASS = {
  'FAR':        'active',
  'MEDIUM':     'active',
  'NEAR':       'warning',
  'VERY CLOSE': 'danger',
};

// ── State ─────────────────────────────────────────────────────────────────────
let audioEnabled   = true;
let lastAlertSpoken = '';      // track last spoken text (avoid repeating)
let lastBeepTime   = 0;
let audioCtx       = null;    // Web AudioContext (created on first user gesture)

// ── Element references ────────────────────────────────────────────────────────
const el = {
  objectIcon:   document.getElementById('objectIcon'),
  objectName:   document.getElementById('objectName'),
  zoneBadge:    document.getElementById('zoneBadge'),
  dirLeft:      document.getElementById('dirLeft'),
  dirAhead:     document.getElementById('dirAhead'),
  dirRight:     document.getElementById('dirRight'),
  dirLabel:     document.getElementById('dirLabel'),
  alertMsg:     document.getElementById('alertMsg'),
  alertCard:    document.getElementById('alertCard'),
  statusDot:    document.getElementById('statusDot'),
  statusLabel:  document.getElementById('statusLabel'),
  videoOverlay: document.getElementById('videoOverlay'),
  fpsValue:     document.getElementById('fpsValue'),
  objCount:     document.getElementById('objCount'),
  audioBtn:     document.getElementById('audioBtn'),
  audioIcon:    document.getElementById('audioIcon'),
  audioText:    document.getElementById('audioText'),
};


// ── Web Audio: beep ───────────────────────────────────────────────────────────
function playBeep(zone) {
  const now = Date.now();
  if (now - lastBeepTime < BEEP_COOLDOWN) return;
  lastBeepTime = now;

  const params = BEEP_PARAMS[zone];
  if (!params) return;

  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.type            = 'sine';
    osc.frequency.value = params.freq;
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + params.dur);

    osc.start();
    osc.stop(audioCtx.currentTime + params.dur);
  } catch (e) {
    // Silently ignore — AudioContext may be blocked until user gesture
  }
}


// ── Web Speech API: speak ─────────────────────────────────────────────────────
function speak(text) {
  if (!audioEnabled || !text) return;
  // The server already handles cooldown logic and sends alert only when needed.
  // We track lastAlertSpoken client-side to avoid double-speaking on same poll.
  if (text === lastAlertSpoken) return;
  lastAlertSpoken = text;

  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();   // cut short any currently playing speech

  const utt   = new SpeechSynthesisUtterance(text);
  utt.rate    = 1.05;
  utt.volume  = 1.0;
  utt.pitch   = 1.0;
  window.speechSynthesis.speak(utt);
}


// ── Update zone badge ─────────────────────────────────────────────────────────
function updateZoneBadge(zone) {
  el.zoneBadge.className = 'badge';
  if (!zone) {
    el.zoneBadge.classList.add('badge-none');
    el.zoneBadge.textContent = '—';
    return;
  }
  el.zoneBadge.classList.add(BADGE_CLASS[zone] || 'badge-none');
  el.zoneBadge.textContent = zone;
}


// ── Update direction arrows ───────────────────────────────────────────────────
function updateDirection(direction) {
  el.dirLeft.classList.remove('active');
  el.dirAhead.classList.remove('active');
  el.dirRight.classList.remove('active');

  if (direction === 'LEFT') {
    el.dirLeft.classList.add('active');
    el.dirLabel.textContent = '← Object on your LEFT';
  } else if (direction === 'RIGHT') {
    el.dirRight.classList.add('active');
    el.dirLabel.textContent = '→ Object on your RIGHT';
  } else if (direction === 'AHEAD') {
    el.dirAhead.classList.add('active');
    el.dirLabel.textContent = '↑ Object AHEAD of you';
  } else {
    el.dirLabel.textContent = '–';
  }
}


// ── Update detected object display ───────────────────────────────────────────
function updateObjectDisplay(topName) {
  if (!topName) {
    el.objectIcon.textContent = '👁️';
    el.objectName.textContent = 'None';
    el.objectName.className   = 'object-name';
    return;
  }
  const icon = OBJECT_ICONS[topName] || '📦';
  el.objectIcon.textContent = icon;
  el.objectName.textContent = topName.charAt(0).toUpperCase() + topName.slice(1);
  el.objectName.className   = 'object-name object-name-active';
}


// ── Update header status dot ──────────────────────────────────────────────────
function updateStatusDot(detected, zone) {
  el.statusDot.className = 'status-dot';
  if (!detected || !zone) {
    el.statusLabel.textContent = 'No object detected';
    return;
  }
  el.statusDot.classList.add(DOT_CLASS[zone] || 'active');
  el.statusLabel.textContent = `Tracking — ${zone}`;
}


// ── Update alert card ─────────────────────────────────────────────────────────
function updateAlertCard(zone, alertText) {
  const isVeryClose = zone === 'VERY CLOSE';
  el.alertCard.classList.toggle('danger-alert', isVeryClose);
  el.alertMsg.classList.toggle('danger', isVeryClose);

  if (!alertText) {
    el.alertMsg.textContent = 'No objects detected — area is clear.';
    return;
  }
  el.alertMsg.textContent = alertText;
}


// ── Main state update (called every poll) ─────────────────────────────────────
function applyState(data) {
  const { detected, count, top_name, zone, direction, alert, fps } = data;

  // FPS & count bar
  el.fpsValue.textContent = fps   ? `${fps} fps` : '–';
  el.objCount.textContent = count ?? 0;

  // Video overlay
  el.videoOverlay.style.display = detected ? 'none' : 'flex';

  // Object card
  updateObjectDisplay(detected ? top_name : null);

  // Zone badge
  updateZoneBadge(detected ? zone : null);

  // Direction
  updateDirection(detected ? direction : null);

  // Status dot
  updateStatusDot(detected, zone);

  // Alert card
  updateAlertCard(detected ? zone : null, detected ? alert : '');

  // Audio: speak the server-built alert text
  if (detected && alert && audioEnabled) {
    speak(alert);
    playBeep(zone);
  } else if (!detected) {
    // Reset lastAlertSpoken so next detection is spoken fresh
    lastAlertSpoken = '';
    window.speechSynthesis && window.speechSynthesis.cancel();
  }
}


// ── Poll /state ───────────────────────────────────────────────────────────────
async function pollState() {
  try {
    const res  = await fetch('/state');
    const data = await res.json();

    // Sync audio toggle from server (in case server state diverged)
    if (typeof data.audio === 'boolean') {
      audioEnabled = data.audio;
      refreshAudioBtn();
    }

    applyState(data);
  } catch {
    el.statusLabel.textContent = 'Connection error…';
    el.statusDot.className = 'status-dot';
  }
}


// ── Audio toggle ──────────────────────────────────────────────────────────────
function refreshAudioBtn() {
  if (audioEnabled) {
    el.audioBtn.className   = 'audio-btn audio-on';
    el.audioIcon.textContent = '🔊';
    el.audioText.textContent = 'Audio ON';
  } else {
    el.audioBtn.className   = 'audio-btn audio-off';
    el.audioIcon.textContent = '🔇';
    el.audioText.textContent = 'Audio OFF';
  }
}

async function toggleAudio() {
  // Create AudioContext on first user gesture (browser security requirement)
  if (!audioCtx) {
    try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch(e) {}
  }
  try {
    const res  = await fetch('/toggle_audio', { method: 'POST' });
    const data = await res.json();
    audioEnabled = data.audio;
    refreshAudioBtn();
    if (!audioEnabled) {
      window.speechSynthesis && window.speechSynthesis.cancel();
      lastAlertSpoken = '';
    }
  } catch (err) {
    console.error('Toggle audio failed:', err);
  }
}


// ── Initialise ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshAudioBtn();
  pollState();                              // immediate first poll
  setInterval(pollState, POLL_MS);          // then every 600 ms
});
