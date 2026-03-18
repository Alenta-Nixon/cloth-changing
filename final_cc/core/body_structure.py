import numpy as np

def extract_body_structure(keypoints):

    if keypoints is None or len(keypoints) < 17:
        return None

    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]

        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
        torso_length = np.linalg.norm(left_shoulder[:2] - left_hip[:2])
        leg_length = np.linalg.norm(left_hip[:2] - left_knee[:2])

        height = torso_length + leg_length + 1e-6

        features = np.array([
            shoulder_width / height,
            hip_width / height,
            torso_length / height,
            leg_length / height,
            shoulder_width / (hip_width + 1e-6)
        ])

        return features

    except:
        return None