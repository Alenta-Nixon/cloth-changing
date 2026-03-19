import cv2
import numpy as np

def extract_structure_feature(img_bgr):

    if img_bgr is None or img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]

    # Basic geometric body proportions
    aspect_ratio = h / (w + 1e-8)
    area = h * w

    # Color histogram of lower body (structure proxy)
    lower_half = img_bgr[h//2:, :]
    hist = cv2.calcHist(
        [lower_half],
        [0, 1, 2],
        None,
        [4, 4, 4],
        [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()

    feature = np.concatenate((
        np.array([aspect_ratio, area / 1000000]),
        hist
    ))

    return feature.astype(np.float32)