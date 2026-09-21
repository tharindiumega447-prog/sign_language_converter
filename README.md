# Sign Language to Text Converter (ASL Alphabet A–Z)

Real-time American Sign Language (ASL) gesture recognition system inspired by the `nightfury217836/Sign-Language-to-Text-Conversion` repository. This project captures webcam feeds, extracts 21 3D hand landmarks via MediaPipe, and classifies ASL gestures across the complete **ASL Alphabet (A–Z)** into digital text and spoken audio.

---

## Features

- **Real-Time Landmark Tracking**: Uses MediaPipe 21 hand-joint landmark model for 60 FPS gesture detection.
- **Full ASL Alphabet (A–Z) Classifier**: Multi-dimensional spatial geometric classifier that analyzes finger extension, folding, curling, thumb positioning, tip distances, and crossing angles.
- **Interactive A–Z Reference Grid**: Interactive 26-letter UI panel with real-time green glowing active highlights on detected signs.
- **Sentence Builder & Audio TTS**: Accumulate letters into sentences, insert spaces, copy to clipboard, and speak words out loud using native Web Speech API.
- **Dual Execution Modes**:
  1. **Web Browser Interface ([`index.html`](file:///c:/Users/Tharindi%20Umege/Desktop/wedding/sign_language_converter/index.html))**: Runs instantly in any modern browser with 0 installation required.
  2. **Python Application Server ([`app.py`](file:///c:/Users/Tharindi%20Umege/Desktop/wedding/sign_language_converter/app.py))**: Flask server hosting the web application interface.

---

## How to Run

### Method 1: Web Application (Easiest - 0 Setup)
1. Navigate to the `sign_language_converter` folder.
2. Open [`index.html`](file:///c:/Users/Tharindi%20Umege/Desktop/wedding/sign_language_converter/index.html) in Google Chrome, Microsoft Edge, or Mozilla Firefox.
3. Allow camera permissions when prompted.
4. Perform ASL alphabet gestures (A through Z) in front of your camera!

### Method 2: Python Web Application Server
1. Open terminal in this directory:
   ```bash
   cd sign_language_converter
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Flask application:
   ```bash
   python app.py
   ```
4. Open browser to `http://127.0.0.1:5000`.

---

## Tech Stack
- **Web Core**: HTML5, Vanilla CSS3, JavaScript (ES6+), MediaPipe Hands JS SDK, HTML5 Web Speech API
- **Python Backend**: Python 3.x, Flask, OpenCV (`cv2`), MediaPipe, NumPy, PyTTSx3
- **Color Theme**: Royal Blue (`#2563eb`), Vivid Green (`#22c55e`), Crisp White (`#ffffff`), Deep Navy (`#0a1128`)
