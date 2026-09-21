/**
 * Sign Language "A" to Text Conversion Engine
 * Black, White & Gold Edition
 */

// DOM Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');

const cameraStatus = document.getElementById('cameraStatus');
const statusText = document.getElementById('statusText');
const fpsDisplay = document.getElementById('fpsDisplay');
const handPill = document.getElementById('handPill');
const handText = document.getElementById('handText');

const hugeLetterDisplay = document.getElementById('hugeLetterDisplay');
const targetCharDisplay = document.getElementById('targetCharDisplay');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const gestureDesc = document.getElementById('gestureDesc');
const textBuffer = document.getElementById('textBuffer');

const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const skeletonToggle = document.getElementById('skeletonToggle');
const speakToggle = document.getElementById('speakToggle');

const appendBtn = document.getElementById('appendBtn');
const speakBtn = document.getElementById('speakBtn');
const copyBtn = document.getElementById('copyBtn');
const clearBtn = document.getElementById('clearBtn');

// State Variables
let currentStream = null;
let isCameraActive = false;
let animFrameId = null;
let lastFrameTime = performance.now();
let frameCount = 0;

let detectedChar = '-';
let currentConfidence = 0;
let sentenceText = "";
let lastSpokenTime = 0;

const synth = window.speechSynthesis;

// ----------------------------------------------------
// 1. MediaPipe Hands Initialization
// ----------------------------------------------------
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(onResults);

// ----------------------------------------------------
// 2. Camera Setup
// ----------------------------------------------------
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusText.textContent = "Browser webcam unsupported";
    return;
  }

  statusText.textContent = "Accessing Camera...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });

    currentStream = stream;
    videoElement.srcObject = stream;
    
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      isCameraActive = true;
      statusText.textContent = "Camera Active";
      cameraStatus.querySelector('.gold-dot').className = "gold-dot active pulse";
      processVideoFrame();
    };

  } catch (err) {
    console.error("Camera Error:", err);
    statusText.textContent = "Camera Error / Permission Denied";
    cameraStatus.querySelector('.gold-dot').className = "gold-dot";
    drawErrorOnCanvas("Camera access denied or unavailable. Please grant camera permission.");
  }
}

function stopCamera() {
  isCameraActive = false;
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  videoElement.srcObject = null;
  statusText.textContent = "Camera Paused";
  cameraStatus.querySelector('.gold-dot').className = "gold-dot";
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function drawErrorOnCanvas(msg) {
  canvasCtx.fillStyle = "#0a1128";
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.fillStyle = "#2563eb";
  canvasCtx.font = "bold 18px 'Cinzel', serif";
  canvasCtx.textAlign = "center";
  canvasCtx.fillText("âš ï¸ WEBCAM NOTICE", canvasElement.width / 2, canvasElement.height / 2 - 20);
  canvasCtx.fillStyle = "#FFFFFF";
  canvasCtx.font = "14px 'Outfit', sans-serif";
  canvasCtx.fillText(msg, canvasElement.width / 2, canvasElement.height / 2 + 15);
}

async function processVideoFrame() {
  if (!isCameraActive) return;
  if (videoElement.readyState >= 2) {
    await hands.send({ image: videoElement });
  }
  animFrameId = requestAnimationFrame(processVideoFrame);
}

// ----------------------------------------------------
// 3. Landmark Euclidean Distance Math & Gesture A Classifier
// ----------------------------------------------------
function dist(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * High-Precision Sign "A" Classifier
 */
function classifySignA(landmarks) {
  const wrist = landmarks[0];
  const thumbMcp = landmarks[2];
  const thumbTip = landmarks[4];
  
  const indexMcp = landmarks[5];
  const indexPip = landmarks[6];
  const indexTip = landmarks[8];
  
  const middleMcp = landmarks[9];
  const middlePip = landmarks[10];
  const middleTip = landmarks[12];
  
  const ringMcp = landmarks[13];
  const ringPip = landmarks[14];
  const ringTip = landmarks[16];
  
  const pinkyMcp = landmarks[17];
  const pinkyPip = landmarks[18];
  const pinkyTip = landmarks[20];

  const handScale = dist(wrist, middleMcp);
  if (handScale === 0) return { isA: false, confidence: 0 };

  // Check finger folding (tips tucked near palm)
  const isIndexFolded = dist(indexTip, wrist) < dist(indexPip, wrist) || dist(indexTip, indexMcp) < 0.7 * handScale;
  const isMiddleFolded = dist(middleTip, wrist) < dist(middlePip, wrist) || dist(middleTip, middleMcp) < 0.7 * handScale;
  const isRingFolded = dist(ringTip, wrist) < dist(ringPip, wrist) || dist(ringTip, ringMcp) < 0.7 * handScale;
  const isPinkyFolded = dist(pinkyTip, wrist) < dist(pinkyPip, wrist) || dist(pinkyTip, pinkyMcp) < 0.7 * handScale;

  const allFourFolded = isIndexFolded && isMiddleFolded && isRingFolded && isPinkyFolded;
  
  // Thumb position relative to index knuckle
  const thumbIndexDist = dist(thumbTip, indexMcp) / handScale;
  const isThumbUpright = thumbTip.y < thumbMcp.y || thumbTip.y < indexMcp.y + 0.1;

  if (allFourFolded && isThumbUpright && thumbIndexDist < 0.65) {
    let conf = Math.round(85 + Math.min(13, (0.65 - thumbIndexDist) * 30));
    return {
      isA: true,
      confidence: conf,
      desc: "SIGN 'A' RECOGNIZED! Fist closed with thumb upright along index finger."
    };
  }

  if (allFourFolded) {
    return {
      isA: true,
      confidence: 76,
      desc: "Fist Detected. Keep thumb upright alongside index finger for Sign 'A'."
    };
  }

  return {
    isA: false,
    confidence: 25,
    desc: "Form a closed fist with thumb upright to make Sign 'A'."
  };
}

// ----------------------------------------------------
// 4. Render Frame & Results
// ----------------------------------------------------
function onResults(results) {
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    fpsDisplay.textContent = `${frameCount} FPS`;
    frameCount = 0;
    lastFrameTime = now;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    
    handPill.classList.add('detected');
    handText.textContent = "Hand Detected";

    // Draw Gold Skeleton Overlay
    if (skeletonToggle.checked) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#2563eb', lineWidth: 4 });
      drawLandmarks(canvasCtx, landmarks, { color: '#22c55e', lineWidth: 2, radius: 5 });
    }

    const classification = classifySignA(landmarks);

    if (classification.confidence >= 70 && classification.isA) {
      detectedChar = 'A';
      currentConfidence = classification.confidence;

      hugeLetterDisplay.textContent = 'A';
      targetCharDisplay.textContent = 'A';
      confidenceFill.style.width = `${classification.confidence}%`;
      confidenceText.textContent = `${classification.confidence}% Confidence`;
      gestureDesc.textContent = classification.desc;

      // TTS Pronunciation
      if (speakToggle.checked && (now - lastSpokenTime > 2500)) {
        speakText("Letter A");
        lastSpokenTime = now;
      }
    } else {
      detectedChar = '-';
      hugeLetterDisplay.textContent = '-';
      confidenceFill.style.width = `${classification.confidence}%`;
      confidenceText.textContent = `${classification.confidence}% (Low)`;
      gestureDesc.textContent = classification.desc;
    }
  } else {
    handPill.classList.remove('detected');
    handText.textContent = "No Hand Detected";
    hugeLetterDisplay.textContent = '-';
    targetCharDisplay.textContent = '?';
    confidenceFill.style.width = '0%';
    confidenceText.textContent = '0% Confidence';
    gestureDesc.textContent = "Hold hand up to camera to make Sign 'A'";
  }

  canvasCtx.restore();
}

// ----------------------------------------------------
// 5. Audio & Buffer Management
// ----------------------------------------------------
function speakText(msg) {
  if (!synth) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(msg);
  u.rate = 0.95;
  synth.speak(u);
}

appendBtn.addEventListener('click', () => {
  if (detectedChar === 'A') {
    sentenceText += 'A';
    textBuffer.textContent = sentenceText;
  }
});

speakBtn.addEventListener('click', () => {
  if (sentenceText) speakText(sentenceText);
});

copyBtn.addEventListener('click', () => {
  if (!sentenceText) return;
  navigator.clipboard.writeText(sentenceText).then(() => {
    copyBtn.style.color = 'var(--gold-light)';
    setTimeout(() => copyBtn.style.color = 'var(--pure-white)', 1200);
  });
});

clearBtn.addEventListener('click', () => {
  sentenceText = "";
  textBuffer.textContent = "";
});

toggleCameraBtn.addEventListener('click', () => {
  if (isCameraActive) {
    stopCamera();
    toggleCameraBtn.textContent = "Resume Camera";
  } else {
    startCamera();
    toggleCameraBtn.textContent = "Pause Camera";
  }
});

document.addEventListener('DOMContentLoaded', () => {
  startCamera();
});


