import cv2

def get_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Tidak dapat membuka camera")
        exit()

    return cap