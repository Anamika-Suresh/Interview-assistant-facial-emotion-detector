from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load trained emotion model
model = load_model("model.h5")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV Haar cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Face cascade loaded:", not face_cascade.empty())


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Read uploaded image
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60)
        )

        print("Faces detected:", len(faces))

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        # Use largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop face region
        face = gray[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Reshape for model
        face = face.reshape(1, 48, 48, 1)

        # Predict
        pred = model.predict(face, verbose=0)[0]
        pred_index = int(np.argmax(pred))
        emotion = emotions[pred_index]
        confidence = float(np.max(pred))

        print("Prediction scores:", pred)
        print("Predicted emotion:", emotion)
        print("Confidence:", confidence)

        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)