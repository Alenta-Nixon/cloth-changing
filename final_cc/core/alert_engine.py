import numpy as np

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    )

def match_identity(s_face, s_app, s_struct,
                   t_face, t_app, t_struct):

    face_sim = cosine_similarity(s_face, t_face)
    app_sim = cosine_similarity(s_app, t_app)
    struct_sim = cosine_similarity(s_struct, t_struct)

    if s_face is not None and t_face is not None:
        final_score = (
            0.65 * face_sim +
            0.20 * struct_sim +
            0.15 * app_sim
        )
    else:
        final_score = (
            0.55 * struct_sim +
            0.45 * app_sim
        )

    percentage = final_score * 100

    return final_score > 0.60, percentage