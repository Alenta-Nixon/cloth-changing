import cv2
import os

video_path = "ansel.mp4"   #ideo.mp4 your video
output_dir = "test_cases/frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # save every 30th frame (1 sec approx)
    if frame_id % 30 == 0:
        filename = f"f{count}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        print("Saved:", filename)
        count += 1

    frame_id += 1

cap.release()
print("Done extracting frames.")