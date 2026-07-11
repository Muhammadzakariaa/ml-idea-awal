import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load Model dan Label Encoder
model = pickle.load(open('../models/model.pkl', 'rb'))
le = pickle.load(open('../models/label_encoder.pkl', 'rb'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

current_gesture = "Mencari Tangan..."

def gen_frames():
    global current_gesture
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Balik kamera (mirror mode) agar lebih natural
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Prediksi
                    prediction = model.predict([landmarks])
                    current_gesture = le.inverse_transform(prediction)[0]
                    
                    # Gambar titik tangan di layar
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                current_gesture = "Tidak Ada Tangan"

            # Resize frame agar tidak terlalu besar di browser
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    return jsonify(gesture=current_gesture)

if __name__ == '__main__':
    app.run(debug=True, port=5000)