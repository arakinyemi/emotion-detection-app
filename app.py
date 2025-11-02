from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import joblib
import os
import sqlite3
from datetime import datetime
import base64

app = Flask(__name__)

# Set maximum file upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the trained model
model = joblib.load("emotion_mlp_model.pkl")

# Emotion labels (same order as training)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Database setup ---
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        image_path TEXT,
                        predicted_emotion TEXT,
                        created_at TEXT
                    )''')
    conn.commit()
    conn.close()

init_db()

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    file = request.files['image']

    if not file:
        return "No image uploaded", 400

    # Save uploaded image
    image_path = os.path.join('static', file.filename)
    file.save(image_path)

    # Read and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)).flatten().reshape(1, -1)
    img = img / 255.0

    # Make prediction
    pred = model.predict(img)[0]
    predicted_emotion = class_names[int(pred)]

    # Save to database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, image_path, predicted_emotion, created_at) VALUES (?, ?, ?, ?)",
        (name, image_path, predicted_emotion, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

    return render_template('index.html', emotion=predicted_emotion, image=image_path, name=name)

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    name = request.form.get('name')
    image_data = request.form.get('image_data')

    if not image_data:
        return "No image data received", 400

    # Decode base64 image
    try:
        # Remove the data URL prefix (e.g., "data:image/jpeg;base64," or "data:image/png;base64,")
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save the image as JPEG to save space
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"webcam_{timestamp}.jpg"
        image_path = os.path.join('static', filename)
        cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Preprocess for prediction
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (48, 48)).flatten().reshape(1, -1)
        img_normalized = img_resized / 255.0
        
        # Make prediction
        pred = model.predict(img_normalized)[0]
        predicted_emotion = class_names[int(pred)]
        
        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, image_path, predicted_emotion, created_at) VALUES (?, ?, ?, ?)",
            (name, image_path, predicted_emotion, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()
        
        return render_template('index.html', emotion=predicted_emotion, image=image_path, name=name)
    except Exception as e:
        return f"Error processing image: {str(e)}", 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
