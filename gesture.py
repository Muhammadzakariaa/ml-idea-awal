import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def get_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]

    fingers = []

    # thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 jari lain
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def detect_gesture(fingers):
    # 👍 ON
    if fingers == [1, 0, 0, 0, 0]:
        return "ON"

    # ✊ OFF
    if fingers == [0, 0, 0, 0, 0]:
        return "OFF"

    # ✌️ BLINK
    if fingers == [0, 1, 1, 0, 0]:
        return "BLINK"
    
    return "UNKNOWN"