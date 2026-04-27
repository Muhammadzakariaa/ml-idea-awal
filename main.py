# main.py

import cv2
import mediapipe as mp
import time
import random

from camera_capture import get_camera
from gesture import hands, get_fingers, detect_gesture

cap = get_camera()

mp_draw = mp.solutions.drawing_utils

lamp_state = "OFF"
blink = False
random_mode = False

last_time = 0
lamp_color = (50, 50, 50)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Gagal menangkap frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture_text = "No Hand"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            fingers = get_fingers(handLms)
            gesture_text = detect_gesture(fingers)

            # ===== LOGIC LAMPU =====
            if gesture_text == "ON":
                lamp_state = "ON"
                blink = False
                random_mode = False

            elif gesture_text == "OFF":
                lamp_state = "OFF"
                blink = False
                random_mode = False

            elif gesture_text == "BLINK":
                blink = True
                random_mode = False

            elif gesture_text == "RANDOM":
                random_mode = True
                blink = False

    # ===== BLINK MODE =====
    if blink:
        if time.time() - last_time > 0.5:
            lamp_state = "ON" if lamp_state == "OFF" else "OFF"
            last_time = time.time()

    # ===== RANDOM COLOR MODE =====
    if random_mode:
        if time.time() - last_time > 0.3:
            lamp_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            lamp_state = "RANDOM"
            last_time = time.time()

    # ===== WARNA LAMPU =====
    if lamp_state == "ON":
        color = (0, 255, 255)

    elif lamp_state == "OFF":
        color = (50, 50, 50)

    elif lamp_state == "RANDOM":
        color = lamp_color

    else:
        color = (50, 50, 50)

    # ===== TAMPILAN =====
    cv2.rectangle(frame, (50, 50), (200, 200), color, -1)

    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (250, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Lamp: {lamp_state}",
        (250, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Lamp System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()