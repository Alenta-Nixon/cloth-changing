from ultralytics import YOLO
import numpy as np

pose_model = YOLO("yolov8n-pose.pt")

def extract_pose_feature(img_bgr):
    results = pose_model(img_bgr, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kpts = results[0].keypoints.xy.cpu().numpy()
    if len(kpts) == 0:
        return None
    vec = kpts[0].flatten()
    return vec / (np.linalg.norm(vec) + 1e-6)