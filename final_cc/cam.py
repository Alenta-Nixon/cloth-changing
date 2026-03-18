import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print("Camera", i, "WORKING")
    else:
        print("Camera", i, "NOT WORKING")
    cap.release()