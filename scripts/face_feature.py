import numpy as np
from deepface import DeepFace

def extract_face_feature(img_bgr):
    try:
        reps = DeepFace.represent(
            img_path=img_bgr,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )
        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        return emb / np.linalg.norm(emb)
    except:
        return None