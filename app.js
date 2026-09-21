/**
 * Sign Language to Text Converter - Two-Handed Sign Classifier (BANZSL / SSL / BSL)
 * Built with MediaPipe Hands & Native WebRTC getUserMedia Pipeline
 */

// DOM Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');

// Status & UI Displays
const cameraStatus = document.getElementById('cameraStatus');
const statusText = document.getElementById('statusText');
const fpsDisplay = document.getElementById('fpsDisplay');
const handDetectedPill = document.getElementById('handDetectedPill');
const handDetectedText = document.getElementById('handDetectedText');

const liveCharDisplay = document.getElementById('liveCharDisplay');
const currentLetterDisplay = document.getElementById('currentLetterDisplay');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceVal = document.getElementById('confidenceVal');
const gestureDesc = document.getElementById('gestureDesc');
const textBuffer = document.getElementById('textBuffer');

// Controls & Buttons
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const skeletonToggle = document.getElementById('skeletonToggle');
const speakToggle = document.getElementById('speakToggle');
const confidenceThresholdInput = document.getElementById('confidenceThreshold');
const thresholdVal = document.getElementById('thresholdVal');

const appendCharBtn = document.getElementById('appendCharBtn');
const addSpaceBtn = document.getElementById('addSpaceBtn');
const copyBtn = document.getElementById('copyBtn');
const speakSentenceBtn = document.getElementById('speakSentenceBtn');
const clearBtn = document.getElementById('clearBtn');
const retryCameraBtn = document.getElementById('retryCameraBtn');

// Debug Inspector Elements
const debugToggleHeader = document.getElementById('debugToggleHeader');
const debugContent = document.getElementById('debugContent');
const debugHandedness = document.getElementById('debugHandedness');
const debugWrist = document.getElementById('debugWrist');
const debugFingers = document.getElementById('debugFingers');
const debugThumbDist = document.getElementById('debugThumbDist');

// Application State Variables
let currentStream = null;
let isCameraActive = false;
let animFrameId = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

let detectedChar = '-';
let currentConfidence = 0;
let minConfidenceThreshold = 65;
let sentenceText = "";
let lastSpokenChar = "";

// Temporal Landmark Smoothing & Gesture Stabilization Cache
let previousHandLandmarks = [];
const gestureWindow = [];
const GESTURE_WINDOW_SIZE = 5;

function smoothLandmarks(newHandList) {
  if (!previousHandLandmarks || previousHandLandmarks.length !== newHandList.length) {
    previousHandLandmarks = newHandList;
    return newHandList;
  }

  const alpha = 0.65; // Smoothing factor
  const smoothedList = [];

  for (let h = 0; h < newHandList.length; h++) {
    const newLm = newHandList[h].landmarks;
    const oldLm = previousHandLandmarks[h] ? previousHandLandmarks[h].landmarks : newLm;
    const smoothedLm = [];

    for (let i = 0; i < newLm.length; i++) {
      smoothedLm.push({
        x: alpha * newLm[i].x + (1 - alpha) * oldLm[i].x,
        y: alpha * newLm[i].y + (1 - alpha) * oldLm[i].y,
        z: alpha * (newLm[i].z || 0) + (1 - alpha) * (oldLm[i].z || 0)
      });
    }
    smoothedList.push({ landmarks: smoothedLm, handedness: newHandList[h].handedness });
  }

  previousHandLandmarks = smoothedList;
  return smoothedList;
}

function getStabilizedGesture(newResult) {
  gestureWindow.push(newResult);
  if (gestureWindow.length > GESTURE_WINDOW_SIZE) {
    gestureWindow.shift();
  }

  const counts = {};
  let maxChar = '?';
  let maxCount = 0;
  let maxResult = newResult;

  for (const item of gestureWindow) {
    if (item.char !== '?') {
      counts[item.char] = (counts[item.char] || 0) + 1;
      if (counts[item.char] > maxCount) {
        maxCount = counts[item.char];
        maxChar = item.char;
        maxResult = item;
      }
    }
  }

  if (maxCount >= 2) {
    return maxResult;
  }

  return newResult;
}

// Speech Synthesis Engine
const synth = window.speechSynthesis;

// ----------------------------------------------------
// 1. MediaPipe Hands Initialization (UPDATED: maxNumHands: 2)
// ----------------------------------------------------
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2, // Changed to 2 for two-handed manual sign language
  modelComplexity: 1,
  minDetectionConfidence: 0.65,
  minTrackingConfidence: 0.65
});

hands.onResults(onResults);

// ----------------------------------------------------
// 2. Native WebRTC Webcam Setup
// ----------------------------------------------------
async function startCamera() {
  const cameraErrorOverlay = document.getElementById('cameraErrorOverlay');
  if (cameraErrorOverlay) cameraErrorOverlay.style.display = 'none';

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError("Your browser does not support webcam access.");
    return;
  }

  statusText.textContent = "Requesting Camera Access...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    currentStream = stream;
    videoElement.srcObject = stream;
    
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      isCameraActive = true;
      statusText.textContent = "Camera Live";
      if (cameraStatus.querySelector('.status-dot')) {
        cameraStatus.querySelector('.status-dot').className = "status-dot active pulse";
      }
      if (cameraErrorOverlay) cameraErrorOverlay.style.display = 'none';
      processVideoFrame();
    };

  } catch (err) {
    console.error("Camera getUserMedia error:", err);
    showCameraError("Camera access denied or unreadable. Please check permissions.");
  }
}

function stopCamera() {
  isCameraActive = false;
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }

  videoElement.srcObject = null;
  statusText.textContent = "Camera Paused";
  if (cameraStatus.querySelector('.status-dot')) {
    cameraStatus.querySelector('.status-dot').className = "status-dot";
  }
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function showCameraError(message) {
  statusText.textContent = "Camera Error";
  if (cameraStatus.querySelector('.status-dot')) {
    cameraStatus.querySelector('.status-dot').className = "status-dot danger";
  }
}

async function processVideoFrame() {
  if (!isCameraActive) return;

  if (videoElement.readyState >= 2) {
    await hands.send({ image: videoElement });
  }

  animFrameId = requestAnimationFrame(processVideoFrame);
}

// ----------------------------------------------------
// 3. Geometry Helpers & Two-Handed Classifier Rules
// ----------------------------------------------------
function dist(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function classifyTwoHandedGesture(handList) {
  if (!handList || handList.length === 0) {
    return { char: '?', confidence: 0, desc: 'No hands detected' };
  }

  // Single-Handed Fallback Check for signs like C
  if (handList.length === 1) {
    const lm = handList[0].landmarks;
    const handScale = dist(lm[0], lm[9]);
    const thumbTip = lm[4], indexTip = lm[8], middleTip = lm[12], ringTip = lm[16], pinkyTip = lm[20];
    
    const isCurled = dist(indexTip, lm[0]) < 1.3 * handScale && dist(pinkyTip, lm[0]) < 1.3 * handScale;
    const thumbIndexGap = dist(thumbTip, indexTip) / handScale;
    
    // Letter 'C': Curved single hand forming C shape
    if (isCurled && thumbIndexGap > 0.40 && thumbIndexGap < 1.15) {
      return { char: 'C', confidence: 92, desc: 'Letter "C" - Curved hand forming C shape' };
    }

    return { char: '?', confidence: 40, desc: 'Place both hands in view for two-handed BANZSL sign gestures' };
  }

  // Extract both hand landmark arrays
  const h1 = handList[0].landmarks;
  const h2 = handList[1].landmarks;

  // Scale estimation based on average palm length
  const scale = (dist(h1[0], h1[9]) + dist(h2[0], h2[9])) / 2;
  if (scale === 0) return { char: '?', confidence: 0, desc: 'Invalid Scale' };

  // Hand 1 Landmark Shortcuts
  const h1_thumb = h1[4], h1_index = h1[8], h1_middle = h1[12], h1_ring = h1[16], h1_pinky = h1[20], h1_palm = h1[9];
  // Hand 2 Landmark Shortcuts
  const h2_thumb = h2[4], h2_index = h2[8], h2_middle = h2[12], h2_ring = h2[16], h2_pinky = h2[20], h2_palm = h2[9];

  // Distances between key points on opposing hands (Normalized by scale)
  const indexToIndex = dist(h1_index, h2_index) / scale;
  const thumbToThumb = dist(h1_thumb, h2_thumb) / scale;
  const index1ToPalm2 = dist(h1_index, h2_palm) / scale;
  const index2ToPalm1 = dist(h2_index, h1_palm) / scale;

  // Check which hand is open (non-dominant palm)
  const isH1Open = dist(h1_pinky, h1[0]) > 1.1 * scale && dist(h1_index, h1[0]) > 1.1 * scale && dist(h1_middle, h1[0]) > 1.1 * scale;
  const isH2Open = dist(h2_pinky, h2[0]) > 1.1 * scale && dist(h2_index, h2[0]) > 1.1 * scale && dist(h2_middle, h2[0]) > 1.1 * scale;

  // 1. VOWELS (A, E, I, O, U)
  // Active index finger touching individual fingertips of open non-dominant hand
  if (isH1Open || isH2Open) {
    const activePoint = isH1Open ? h2_index : h1_index;
    const targetHand = isH1Open ? h1 : h2;

    const distThumb = dist(activePoint, targetHand[4]) / scale;
    const distIndex = dist(activePoint, targetHand[8]) / scale;
    const distMiddle = dist(activePoint, targetHand[12]) / scale;
    const distRing = dist(activePoint, targetHand[16]) / scale;
    const distPinky = dist(activePoint, targetHand[20]) / scale;

    if (distThumb < 0.48) return { char: 'A', confidence: 95, desc: 'Letter "A" - Active index finger touching thumb tip of open hand' };
    if (distIndex < 0.48) return { char: 'E', confidence: 95, desc: 'Letter "E" - Active index finger touching index tip of open hand' };
    if (distMiddle < 0.48) return { char: 'I', confidence: 95, desc: 'Letter "I" - Active index finger touching middle tip of open hand' };
    if (distRing < 0.48) return { char: 'O', confidence: 95, desc: 'Letter "O" - Active index finger touching ring tip of open hand' };
    if (distPinky < 0.48) return { char: 'U', confidence: 95, desc: 'Letter "U" - Active index finger touching pinky tip of open hand' };
  }

  // Circles for OK shape
  const h1_circle = dist(h1_thumb, h1_index) / scale;
  const h2_circle = dist(h2_thumb, h2_index) / scale;

  // 2. Letter 'B': Two index-thumb circles joined together (two OK shapes touching)
  if (h1_circle < 0.45 && h2_circle < 0.45 && indexToIndex < 0.45) {
    return { char: 'B', confidence: 94, desc: 'Letter "B" - Two thumb-index circles joined together' };
  }

  // 3. Letter 'C': Two hands curved into C shape facing each other or single hand C
  const isH1Curved = dist(h1_index, h1[0]) < 1.3 * scale && dist(h1_thumb, h1_index) / scale > 0.4;
  const isH2Curved = dist(h2_index, h2[0]) < 1.3 * scale && dist(h2_thumb, h2_index) / scale > 0.4;
  if (isH1Curved && isH2Curved && dist(h1_index, h2_index) / scale < 0.5 && dist(h1_thumb, h2_thumb) / scale < 0.5) {
    return { char: 'C', confidence: 92, desc: 'Letter "C" - Curved hands forming C shape' };
  }

  // 4. Letter 'D': Index tips touching + dominant thumb touching non-dominant index base
  if (indexToIndex < 0.4 && (dist(h1_thumb, h2[5]) / scale < 0.55 || dist(h2_thumb, h1[5]) / scale < 0.55)) {
    return { char: 'D', confidence: 91, desc: 'Letter "D" - Index fingers touching with thumb loop at base' };
  }

  // 5. Letter 'F': Two index fingers crossed over each other + two middle fingers crossed
  const indexCross = dist(h1_index, h2[6]) / scale < 0.45 || dist(h2_index, h1[6]) / scale < 0.45;
  const middleCross = dist(h1_middle, h2[10]) / scale < 0.45 || dist(h2_middle, h1[10]) / scale < 0.45;
  if (indexCross && middleCross) {
    return { char: 'F', confidence: 90, desc: 'Letter "F" - Two index & middle fingers crossed in grid' };
  }

  // 6. Letter 'G': Two closed fists placed vertically on top of each other
  const isH1Fist = dist(h1_index, h1[0]) < 1.1 * scale && dist(h1_pinky, h1[0]) < 1.1 * scale;
  const isH2Fist = dist(h2_index, h2[0]) < 1.1 * scale && dist(h2_pinky, h2[0]) < 1.1 * scale;
  if (isH1Fist && isH2Fist && dist(h1[0], h2[0]) / scale < 1.5 && (index1ToPalm2 < 0.5 || index2ToPalm1 < 0.5)) {
    return { char: 'G', confidence: 89, desc: 'Letter "G" - Two closed fists stacked vertically' };
  }

  // 7. Letter 'H': Flat hand resting/swiping flat across open palm
  if ((isH1Open || isH2Open) && (dist(h1_index, h2_palm) / scale < 0.45 && dist(h1_middle, h2_palm) / scale < 0.45)) {
    return { char: 'H', confidence: 88, desc: 'Letter "H" - Flat hand resting across open palm' };
  }

  // 8. Letter 'J': Active index finger tracing/touching middle finger to palm
  if ((dist(h1_index, h2[12]) / scale < 0.45 || dist(h2_index, h1[12]) / scale < 0.45) && (index1ToPalm2 < 0.5 || index2ToPalm1 < 0.5)) {
    return { char: 'J', confidence: 87, desc: 'Letter "J" - Active index finger tracing J shape from middle finger' };
  }

  // 9. Letter 'K': Active index finger hooked around extended non-dominant index finger
  if (indexToIndex < 0.4 && (dist(h1_index, h2[7]) / scale < 0.4 || dist(h2_index, h1[7]) / scale < 0.4)) {
    return { char: 'K', confidence: 88, desc: 'Letter "K" - Active index finger hooked onto opposite index finger' };
  }

  // 10. Letter 'L': Active index finger resting perpendicularly across open palm
  if ((index1ToPalm2 < 0.4 || index2ToPalm1 < 0.4) && !isH1Open && !isH2Open) {
    return { char: 'L', confidence: 89, desc: 'Letter "L" - Index finger placed perpendicularly across opposite palm' };
  }

  // 11. Letter 'M': Three fingers (index, middle, ring) laid across open palm
  const h1_M = dist(h1_index, h2_palm) / scale < 0.42 && dist(h1_middle, h2_palm) / scale < 0.42 && dist(h1_ring, h2_palm) / scale < 0.42;
  const h2_M = dist(h2_index, h1_palm) / scale < 0.42 && dist(h2_middle, h1_palm) / scale < 0.42 && dist(h2_ring, h1_palm) / scale < 0.42;
  if (h1_M || h2_M) {
    return { char: 'M', confidence: 92, desc: 'Letter "M" - 3 fingers (index, middle, ring) laid on opposite palm' };
  }

  // 12. Letter 'N': Two fingers (index, middle) laid across open palm
  const h1_N = dist(h1_index, h2_palm) / scale < 0.42 && dist(h1_middle, h2_palm) / scale < 0.42;
  const h2_N = dist(h2_index, h1_palm) / scale < 0.42 && dist(h2_middle, h1_palm) / scale < 0.42;
  if (h1_N || h2_N) {
    return { char: 'N', confidence: 91, desc: 'Letter "N" - 2 fingers (index, middle) laid on opposite palm' };
  }

  // 13. Letter 'P': Active thumb-index circle touching opposite index tip
  if ((dist(h1_index, h2_index) / scale < 0.4 && h1_circle < 0.45) || (dist(h2_index, h1_index) / scale < 0.4 && h2_circle < 0.45)) {
    return { char: 'P', confidence: 89, desc: 'Letter "P" - Thumb/index circle touching index finger tip' };
  }

  // 14. Letter 'Q': Active index finger hooked inside opposite thumb-index circle
  if ((h1_circle < 0.45 && dist(h2_index, h1_thumb) / scale < 0.4) || (h2_circle < 0.45 && dist(h1_index, h2_thumb) / scale < 0.4)) {
    return { char: 'Q', confidence: 88, desc: 'Letter "Q" - Active index finger hooked through thumb-index circle' };
  }

  // 15. Letter 'R': Curved index finger resting on flat palm
  if ((dist(h1_index, h2_palm) / scale < 0.45 && dist(h1_index, h1[0]) / scale < 1.0) ||
      (dist(h2_index, h1_palm) / scale < 0.45 && dist(h2_index, h2[0]) / scale < 1.0)) {
    return { char: 'R', confidence: 87, desc: 'Letter "R" - Hooked index finger resting on open palm' };
  }

  // 16. Letter 'S': Pinky fingers hooked/interlocked together
  if (dist(h1_pinky, h2_pinky) / scale < 0.4) {
    return { char: 'S', confidence: 90, desc: 'Letter "S" - Pinky fingers hooked together' };
  }

  // 17. Letter 'T': Active index finger touching edge/side of non-dominant hand near wrist
  if (dist(h1_index, h2[17]) / scale < 0.45 || dist(h2_index, h1[17]) / scale < 0.45) {
    return { char: 'T', confidence: 87, desc: 'Letter "T" - Active index finger touching side edge of hand' };
  }

  // 18. Letter 'V': Active hand forming V shape (index & middle spread) resting on palm
  const h1_V = dist(h1_index, h2_palm) / scale < 0.45 && dist(h1_middle, h2_palm) / scale < 0.45 && dist(h1_index, h1_middle) / scale > 0.35;
  const h2_V = dist(h2_index, h1_palm) / scale < 0.45 && dist(h2_middle, h1_palm) / scale < 0.45 && dist(h2_index, h2_middle) / scale > 0.35;
  if (h1_V || h2_V) {
    return { char: 'V', confidence: 90, desc: 'Letter "V" - V-shape fingers resting on open palm' };
  }

  // 19. Letter 'W': Both hands with fingers spread and interlocked forming W pattern
  if (dist(h1_index, h2_middle) / scale < 0.45 && dist(h1_middle, h2_index) / scale < 0.45) {
    return { char: 'W', confidence: 91, desc: 'Letter "W" - Both hands interlocking fingers in W pattern' };
  }

  // 20. Letter 'X': Two index fingers crossed over each other forming an X shape
  if (indexToIndex < 0.4 && dist(h1[7], h2[7]) / scale < 0.4) {
    return { char: 'X', confidence: 92, desc: 'Letter "X" - Two index fingers crossed in an X shape' };
  }

  // 21. Letter 'Y': Active index finger resting in thumb-index crease/webbing of open hand
  if (dist(h1_index, h2[1]) / scale < 0.45 || dist(h2_index, h1[1]) / scale < 0.45 || dist(h1_index, h2[2]) / scale < 0.45 || dist(h2_index, h1[2]) / scale < 0.45) {
    return { char: 'Y', confidence: 88, desc: 'Letter "Y" - Active index finger resting in thumb-index crease' };
  }

  // 22. Letter 'Z': Both flat palms placed edge-to-edge facing each other
  if (dist(h1_palm, h2_palm) / scale < 0.65 && thumbToThumb < 0.55) {
    return { char: 'Z', confidence: 86, desc: 'Letter "Z" - Both flat hands placed side-by-side' };
  }

  return { char: '?', confidence: 35, desc: 'Unrecognized gesture. Refer to BANZSL chart and hold hands clearly in frame.' };
}

// ----------------------------------------------------
// 4. MediaPipe Results Processing
// ----------------------------------------------------
function onResults(results) {
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    currentFps = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    if (fpsDisplay) fpsDisplay.textContent = `${currentFps} FPS`;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Draw Camera Frame
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const handsDetected = results.multiHandLandmarks.length;
    handDetectedPill.classList.add('detected');
    handDetectedText.textContent = `${handsDetected} Hand(s) Detected`;

    const handList = [];

    // Loop through each detected hand
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      const handedness = results.multiHandedness[i] ? results.multiHandedness[i].label : `Hand ${i+1}`;
      
      handList.push({ landmarks, handedness });

      if (skeletonToggle.checked) {
        const colorScheme = i === 0 ? '#2563eb' : '#e11d48';
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: colorScheme, lineWidth: 3 });
        drawLandmarks(canvasCtx, landmarks, { color: '#22c55e', lineWidth: 2, radius: 4 });
      }
    }

    // Update Debug Inspector
    if (debugHandedness) debugHandedness.textContent = handList.map(h => h.handedness).join(', ');
    if (debugWrist && handList[0]) {
      const w = handList[0].landmarks[0];
      debugWrist.textContent = `X: ${w.x.toFixed(2)}, Y: ${w.y.toFixed(2)}`;
    }
    if (debugFingers && handList[0]) {
      const lm = handList[0].landmarks;
      const thumbExt = dist(lm[4], lm[0]) > dist(lm[2], lm[0]);
      const indexExt = dist(lm[8], lm[0]) > dist(lm[5], lm[0]);
      const midExt = dist(lm[12], lm[0]) > dist(lm[9], lm[0]);
      const ringExt = dist(lm[16], lm[0]) > dist(lm[13], lm[0]);
      const pinkyExt = dist(lm[20], lm[0]) > dist(lm[17], lm[0]);
      debugFingers.textContent = `T:${thumbExt ? '✓' : '✗'} I:${indexExt ? '✓' : '✗'} M:${midExt ? '✓' : '✗'} R:${ringExt ? '✓' : '✗'} P:${pinkyExt ? '✓' : '✗'}`;
    }
    if (debugThumbDist && handList[0]) {
      const td = dist(handList[0].landmarks[4], handList[0].landmarks[8]);
      debugThumbDist.textContent = td.toFixed(3);
    }

    // Smooth landmarks across frames to remove jitter
    const smoothedHandList = smoothLandmarks(handList);

    // Classify gesture on smoothed landmarks
    const rawGestureResult = classifyTwoHandedGesture(smoothedHandList);

    // Stabilize results using temporal majority voting window
    const gestureResult = getStabilizedGesture(rawGestureResult);

    if (gestureResult.confidence >= minConfidenceThreshold && gestureResult.char !== '?') {
      detectedChar = gestureResult.char;
      currentConfidence = gestureResult.confidence;

      liveCharDisplay.textContent = gestureResult.char;
      currentLetterDisplay.textContent = gestureResult.char;
      confidenceBar.style.width = `${gestureResult.confidence}%`;
      confidenceVal.textContent = `${gestureResult.confidence}% Confidence`;
      gestureDesc.textContent = gestureResult.desc;

      updateAlphabetGridHighlight(gestureResult.char);

      if (speakToggle.checked && lastSpokenChar !== detectedChar) {
        speakCharacter(detectedChar);
        lastSpokenChar = detectedChar;
      }
    } else {
      liveCharDisplay.textContent = '-';
      confidenceBar.style.width = `${gestureResult.confidence}%`;
      confidenceVal.textContent = `${gestureResult.confidence}% (Low)`;
      gestureDesc.textContent = gestureResult.desc;
      updateAlphabetGridHighlight('-');
    }
  } else {
    handDetectedPill.classList.remove('detected');
    handDetectedText.textContent = "No Hands Detected";
    liveCharDisplay.textContent = '-';
    currentLetterDisplay.textContent = '?';
    confidenceBar.style.width = '0%';
    confidenceVal.textContent = '0% Confidence';
    gestureDesc.textContent = 'Hold both hands in front of the camera to perform two-handed signs (A-Z)';
    updateAlphabetGridHighlight('-');
  }

  canvasCtx.restore();
}

function getSignSvg(letter) {
  const primary = '#2563eb';
  const accent = '#22c55e';

  switch (letter) {
    case 'A':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 25 75 L 25 35 A 6 6 0 0 1 37 35 L 37 75 Z" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 37 40 L 37 25 A 5 5 0 0 1 47 25 L 47 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 47 42 L 47 28 A 5 5 0 0 1 57 28 L 57 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 57 45 L 57 33 A 5 5 0 0 1 67 33 L 67 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 25 50 C 15 45 10 35 15 25 A 5 5 0 0 1 25 25" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 80 50 L 25 25" fill="none" stroke="${accent}" stroke-width="4" stroke-dasharray="3,3"/>
        <circle cx="20" cy="25" r="6" fill="${accent}"/>
      </svg>`;
    case 'B':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <circle cx="38" cy="50" r="16" fill="none" stroke="${primary}" stroke-width="4"/>
        <circle cx="62" cy="50" r="16" fill="none" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    case 'C':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 65 25 C 25 20 25 80 65 75" fill="none" stroke="${primary}" stroke-width="6" stroke-linecap="round"/>
      </svg>`;
    case 'D':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <line x1="40" y1="20" x2="40" y2="80" stroke="${primary}" stroke-width="6" stroke-linecap="round"/>
        <path d="M 40 80 C 75 80 75 35 40 35" fill="none" stroke="${accent}" stroke-width="5"/>
      </svg>`;
    case 'E':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 30 75 L 30 25 A 5 5 0 0 1 40 25 L 40 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 40 30 L 40 20 A 5 5 0 0 1 50 20 L 50 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="80" y1="50" x2="40" y2="20" stroke="${accent}" stroke-width="4" stroke-dasharray="3,3"/>
        <circle cx="40" cy="20" r="6" fill="${accent}"/>
      </svg>`;
    case 'F':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <line x1="30" y1="40" x2="70" y2="40" stroke="${primary}" stroke-width="5"/>
        <line x1="30" y1="60" x2="70" y2="60" stroke="${primary}" stroke-width="5"/>
        <line x1="40" y1="30" x2="40" y2="70" stroke="${accent}" stroke-width="5"/>
        <line x1="60" y1="30" x2="60" y2="70" stroke="${accent}" stroke-width="5"/>
      </svg>`;
    case 'G':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="35" y="20" width="30" height="28" rx="6" fill="none" stroke="${primary}" stroke-width="4"/>
        <rect x="35" y="52" width="30" height="28" rx="6" fill="none" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    case 'H':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="55" width="60" height="15" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <rect x="30" y="35" width="40" height="15" rx="4" fill="none" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    case 'I':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 30 75 L 30 30 A 5 5 0 0 1 40 30 L 40 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 40 25 L 40 15 A 5 5 0 0 1 50 15 L 50 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <circle cx="50" cy="15" r="6" fill="${accent}"/>
        <line x1="80" y1="50" x2="50" y2="15" stroke="${accent}" stroke-width="4" stroke-dasharray="3,3"/>
      </svg>`;
    case 'J':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 50 20 L 50 65 A 12 12 0 0 1 28 65" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'K':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <line x1="35" y1="20" x2="35" y2="80" stroke="${primary}" stroke-width="5"/>
        <path d="M 65 30 L 35 45 L 65 70" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'L':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="45" width="60" height="30" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="50" y1="15" x2="50" y2="60" stroke="${accent}" stroke-width="6" stroke-linecap="round"/>
      </svg>`;
    case 'M':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="50" width="60" height="25" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="35" y1="25" x2="35" y2="62" stroke="${accent}" stroke-width="4"/>
        <line x1="50" y1="25" x2="50" y2="62" stroke="${accent}" stroke-width="4"/>
        <line x1="65" y1="25" x2="65" y2="62" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    case 'N':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="50" width="60" height="25" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="40" y1="25" x2="40" y2="62" stroke="${accent}" stroke-width="4"/>
        <line x1="60" y1="25" x2="60" y2="62" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    case 'O':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 50 30 L 50 18 A 5 5 0 0 1 60 18 L 60 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <circle cx="60" cy="18" r="6" fill="${accent}"/>
        <line x1="20" y1="50" x2="60" y2="18" stroke="${accent}" stroke-width="4" stroke-dasharray="3,3"/>
      </svg>`;
    case 'P':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <circle cx="35" cy="40" r="14" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="35" y1="54" x2="35" y2="80" stroke="${primary}" stroke-width="4"/>
        <line x1="49" y1="40" x2="80" y2="40" stroke="${accent}" stroke-width="5"/>
      </svg>`;
    case 'Q':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <circle cx="45" cy="50" r="18" fill="none" stroke="${primary}" stroke-width="5"/>
        <path d="M 80 50 L 45 50" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'R':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="55" width="60" height="20" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <path d="M 50 25 C 65 25 65 50 50 65" fill="none" stroke="${accent}" stroke-width="5"/>
      </svg>`;
    case 'S':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 35 30 C 20 40 45 60 35 70" fill="none" stroke="${primary}" stroke-width="5" stroke-linecap="round"/>
        <path d="M 65 30 C 80 40 55 60 65 70" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'T':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="25" y="30" width="35" height="50" rx="6" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="80" y1="65" x2="60" y2="65" stroke="${accent}" stroke-width="6" stroke-linecap="round"/>
      </svg>`;
    case 'U':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 60 40 L 60 30 A 5 5 0 0 1 70 30 L 70 75" fill="none" stroke="${primary}" stroke-width="4"/>
        <circle cx="70" cy="30" r="6" fill="${accent}"/>
        <line x1="20" y1="50" x2="70" y2="30" stroke="${accent}" stroke-width="4" stroke-dasharray="3,3"/>
      </svg>`;
    case 'V':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="20" y="55" width="60" height="20" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="38" y1="20" x2="46" y2="60" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
        <line x1="62" y1="20" x2="54" y2="60" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'W':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 20 25 L 35 75 L 50 35 L 65 75 L 80 25" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
    case 'X':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <line x1="25" y1="25" x2="75" y2="75" stroke="${primary}" stroke-width="6" stroke-linecap="round"/>
        <line x1="75" y1="25" x2="25" y2="75" stroke="${accent}" stroke-width="6" stroke-linecap="round"/>
      </svg>`;
    case 'Y':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <path d="M 30 75 L 30 35 C 20 30 15 45 25 50" fill="none" stroke="${primary}" stroke-width="4"/>
        <line x1="80" y1="40" x2="25" y2="40" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      </svg>`;
    case 'Z':
      return `<svg viewBox="0 0 100 100" class="sign-svg">
        <rect x="25" y="25" width="22" height="50" rx="4" fill="none" stroke="${primary}" stroke-width="4"/>
        <rect x="53" y="25" width="22" height="50" rx="4" fill="none" stroke="${accent}" stroke-width="4"/>
      </svg>`;
    default:
      return `<svg viewBox="0 0 100 100" class="sign-svg"><text x="50" y="55" text-anchor="middle" fill="${primary}" font-size="30">?</text></svg>`;
  }
}

function updateFeaturedSignCard(letter, customDesc) {
  const container = document.getElementById('featuredSignSvgContainer');
  const title = document.getElementById('featuredSignTitle');
  const desc = document.getElementById('featuredSignDesc');
  const tag = document.getElementById('featuredSignTag');

  if (!container || !title || !desc || !tag) return;
  if (!letter || letter === '?' || letter === '-') return;

  container.innerHTML = getSignSvg(letter);
  title.textContent = `Letter ${letter}`;

  const isVowel = ['A', 'E', 'I', 'O', 'U'].includes(letter);
  tag.textContent = isVowel ? 'Vowel Sign' : 'Consonant Sign';
  tag.style.background = isVowel ? 'rgba(34, 197, 94, 0.2)' : 'rgba(37, 99, 235, 0.2)';
  tag.style.color = isVowel ? '#22c55e' : '#3b82f6';
  tag.style.borderColor = isVowel ? 'rgba(34, 197, 94, 0.35)' : 'rgba(37, 99, 235, 0.35)';

  if (customDesc) {
    desc.textContent = customDesc;
  }
}

function updateAlphabetGridHighlight(activeChar, customDesc) {
  const gridButtons = document.querySelectorAll('.grid-letter-btn');
  gridButtons.forEach(btn => {
    if (btn.dataset.letter === activeChar) {
      btn.classList.add('active-detected');
    } else {
      btn.classList.remove('active-detected');
    }
  });

  if (activeChar && activeChar !== '-' && activeChar !== '?') {
    updateFeaturedSignCard(activeChar, customDesc);
  }
}

// ----------------------------------------------------
// 5. Text & Audio Sentence Management
// ----------------------------------------------------
function speakCharacter(char) {
  if (!synth) return;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(char);
  utterance.rate = 1.0;
  synth.speak(utterance);
}

function speakText(text) {
  if (!synth || !text) return;
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  synth.speak(utterance);
}

function appendToSentence(char) {
  if (!char || char === '?' || char === '-') return;
  sentenceText += char;
  textBuffer.textContent = sentenceText;
}

// ----------------------------------------------------
// 6. Event Listeners & Binding
// ----------------------------------------------------
toggleCameraBtn.addEventListener('click', () => {
  if (isCameraActive) {
    stopCamera();
    toggleCameraBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
        <circle cx="12" cy="13" r="4"/>
      </svg>
      Resume Camera`;
  } else {
    startCamera();
    toggleCameraBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
        <circle cx="12" cy="13" r="4"/>
      </svg>
      Stop Camera`;
  }
});

if (retryCameraBtn) {
  retryCameraBtn.addEventListener('click', () => {
    startCamera();
  });
}

confidenceThresholdInput.addEventListener('input', (e) => {
  minConfidenceThreshold = parseInt(e.target.value);
  thresholdVal.textContent = `${minConfidenceThreshold}%`;
});

appendCharBtn.addEventListener('click', () => {
  if (detectedChar !== '-' && detectedChar !== '?') {
    appendToSentence(detectedChar);
  }
});

if (addSpaceBtn) {
  addSpaceBtn.addEventListener('click', () => {
    sentenceText += " ";
    textBuffer.textContent = sentenceText;
  });
}

copyBtn.addEventListener('click', () => {
  if (!sentenceText) return;
  navigator.clipboard.writeText(sentenceText);
});

speakSentenceBtn.addEventListener('click', () => {
  if (sentenceText) speakText(sentenceText);
});

clearBtn.addEventListener('click', () => {
  sentenceText = "";
  textBuffer.textContent = "";
});

if (debugToggleHeader && debugContent) {
  debugToggleHeader.addEventListener('click', () => {
    const isHidden = debugContent.style.display === 'none' || getComputedStyle(debugContent).display === 'none';
    debugContent.style.display = isHidden ? 'flex' : 'none';
    const arrow = debugToggleHeader.querySelector('.arrow-icon');
    if (arrow) arrow.style.transform = isHidden ? 'rotate(180deg)' : 'rotate(0deg)';
  });
}

// ----------------------------------------------------
// Modal Demonstration Controller
// ----------------------------------------------------
const signModal = document.getElementById('signModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const modalLetterBadge = document.getElementById('modalLetterBadge');
const modalTitle = document.getElementById('modalTitle');
const modalCategoryTag = document.getElementById('modalCategoryTag');
const modalMediaContainer = document.getElementById('modalMediaContainer');
const modalDescription = document.getElementById('modalDescription');
const modalPracticeBtn = document.getElementById('modalPracticeBtn');
const modalAppendBtn = document.getElementById('modalAppendBtn');

let currentModalLetter = 'A';

function openSignModal(letter) {
  if (!signModal) return;
  currentModalLetter = letter;
  
  if (modalLetterBadge) modalLetterBadge.textContent = letter;
  if (modalTitle) modalTitle.textContent = `Letter ${letter} Sign Demonstration`;
  
  const isVowel = ['A', 'E', 'I', 'O', 'U'].includes(letter);
  if (modalCategoryTag) {
    modalCategoryTag.textContent = isVowel ? 'Vowel Sign' : 'Consonant Sign';
    modalCategoryTag.style.background = isVowel ? 'rgba(34, 197, 94, 0.2)' : 'rgba(37, 99, 235, 0.2)';
    modalCategoryTag.style.color = isVowel ? '#22c55e' : '#3b82f6';
  }

  if (modalMediaContainer) {
    modalMediaContainer.innerHTML = getSignSvg(letter);
  }

  const descMap = {
    'A': 'Touch the thumb tip of your open non-dominant hand using the index finger of your active hand.',
    'B': 'Form two circles (OK shapes) with the thumb and index finger of both hands and touch the circles together.',
    'C': 'Curve your dominant hand (or both hands) into a curved C shape.',
    'D': 'Touch index finger tips together and form a circle with your dominant thumb at the base of the non-dominant index finger.',
    'E': 'Touch the index finger tip of your open non-dominant hand using your active index finger.',
    'F': 'Cross the index and middle fingers of both hands over each other in a 2x2 grid pattern.',
    'G': 'Stack two closed fists vertically on top of each other.',
    'H': 'Lay your active flat hand horizontally across the open palm of your non-dominant hand.',
    'I': 'Touch the middle finger tip of your open non-dominant hand using your active index finger.',
    'J': 'Touch the middle finger tip of your open hand and trace a J hook down along the palm.',
    'K': 'Hook your active index finger around the extended index finger of your non-dominant hand.',
    'L': 'Lay your active index finger perpendicularly across the open palm of your non-dominant hand.',
    'M': 'Lay three fingers (index, middle, ring) across the open palm of your non-dominant hand.',
    'N': 'Lay two fingers (index, middle) across the open palm of your non-dominant hand.',
    'O': 'Touch the ring finger tip of your open non-dominant hand using your active index finger.',
    'P': 'Form a circle with thumb & index of your active hand and touch the extended index tip of your non-dominant hand.',
    'Q': 'Hook your active index finger through the thumb-index circle formed by your non-dominant hand.',
    'R': 'Rest a curved/hooked active index finger on the open palm of your non-dominant hand.',
    'S': 'Hook the pinky fingers of both hands together.',
    'T': 'Touch the outer edge/side of your non-dominant hand near the wrist using your active index finger.',
    'U': 'Touch the pinky finger tip of your open non-dominant hand using your active index finger.',
    'V': 'Form a V shape (index & middle spread) and rest it on the open palm of your non-dominant hand.',
    'W': 'Interlock the fingers of both hands together forming a W zigzag pattern.',
    'X': 'Cross both index fingers over each other forming an X shape.',
    'Y': 'Rest your active index finger in the thumb-index webbing/crease of your non-dominant open hand.',
    'Z': 'Hold both flat palms side-by-side facing each other.'
  };

  if (modalDescription) {
    modalDescription.textContent = descMap[letter] || 'Hold both hands in front of the camera to perform this gesture.';
  }

  signModal.style.display = 'flex';
}

function closeSignModal() {
  if (signModal) signModal.style.display = 'none';
}

if (closeModalBtn) {
  closeModalBtn.addEventListener('click', closeSignModal);
}

if (signModal) {
  signModal.addEventListener('click', (e) => {
    if (e.target === signModal) closeSignModal();
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeSignModal();
});

if (modalPracticeBtn) {
  modalPracticeBtn.addEventListener('click', () => {
    closeSignModal();
    if (!isCameraActive) startCamera();
  });
}

if (modalAppendBtn) {
  modalAppendBtn.addEventListener('click', () => {
    appendToSentence(currentModalLetter);
    closeSignModal();
  });
}

// Allow clicking reference grid letters to open modal demonstration & append manually
document.querySelectorAll('.grid-letter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const letter = btn.dataset.letter;
    if (letter) {
      updateFeaturedSignCard(letter);
      openSignModal(letter);
    }
  });
});

// ----------------------------------------------------
// 7. Start Camera on Load
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  startCamera();
});