import numpy as np
from scripts.face_feature import extract_face_feature
from scripts.appearance_feature import extract_appearance_feature
from scripts.structure_feature import extract_structure_feature

def normalize(feat):
    if feat is None:
        return None
    return feat / (np.linalg.norm(feat) + 1e-8)

def extract_features(crop, keypoints):

    face = extract_face_feature(crop)
    app = extract_appearance_feature(crop)
    struct = extract_structure_feature(crop)

    face = normalize(face)
    app = normalize(app)
    struct = normalize(struct)

    return face, app, struct