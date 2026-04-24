import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka camera")
    exit()

while(cap.isOpened):
    ret,frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break
    #tampilkan frame
    cv2.imshow("Hand gesture Recognition dengan MediaPipe dan Opencv", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()