import cv2
import numpy as np
from flask import Flask, Response
from tensorflow.keras.models import load_model

app = Flask(__name__)

try:
    model = load_model('saved_models/best_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Could not load haarcascade file")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    face_cascade = None

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Facial Emotion Recognition</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f2f5;
            flex-direction: column;
        }
        .container {
            text-align: center;
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        h1 {
            color: #333;
            margin-bottom: 1rem;
        }
        #video-container {
            width: 640px;
            height: 480px;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: #2c3e50;
        }
        img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        p {
            color: #666;
            margin-top: 1rem;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Facial Emotion Recognition</h1>
        <div id="video-container">
            <img src="/video_feed" alt="Video Stream">
        </div>
        <p>The model will detect faces and classify their emotions in real-time.</p>
    </div>
</body>
</html>
"""

def generate_frames():
    """
    Captures frames from the camera, detects faces, predicts emotions,
    and streams the result.
    """
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        if model is None or face_cascade is None:
            cv2.putText(frame, "Error: Model or Cascade not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                
                prediction = model.predict(roi)
                emotion_label = EMOTION_LABELS[np.argmax(prediction)]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            continue

    camera.release()

@app.route('/')
def index():
    """Returns the embedded HTML page."""
    return Response(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Provides the video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)