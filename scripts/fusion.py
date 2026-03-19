from scipy.spatial.distance import cosine
import numpy as np

FACE_STRONG_MATCH = 80
FACE_GATE = 65
FINAL_THRESHOLD = 70

def similarity(a, b):
    if a is None or b is None:
        return 0
    return (1 - cosine(a, b)) * 100

def hierarchical_decision(face_sim, app_sim, pose_sim, body_sim):
    if face_sim >= FACE_STRONG_MATCH:
        return True, face_sim

    if face_sim < FACE_GATE:
        return False, face_sim

    final_score = (
        0.6 * face_sim +
        0.25 * app_sim +
        0.1 * body_sim +
        0.05 * pose_sim
    )

    return final_score >= FINAL_THRESHOLD, final_score