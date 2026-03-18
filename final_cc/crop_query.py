import cv2
import os

img = cv2.imread("ansell.jpeg")

if img is None:
    print("Image not found")
    exit()

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print(points)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click)

print("Click TOP-LEFT and BOTTOM-RIGHT")

while True:
    cv2.imshow("Image", img)
    if len(points) == 2:
        break
    cv2.waitKey(1)

cv2.destroyAllWindows()

(x1,y1),(x2,y2) = points

crop = img[y1:y2, x1:x2]

# ensure folder exists
os.makedirs("test_cases/query", exist_ok=True)

save_path = os.path.abspath("test_cases/query/p1.jpg")

cv2.imwrite(save_path, crop)

print("Saved at:", save_path)