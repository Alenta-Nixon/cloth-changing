import numpy as np

class IdentityDatabase:

    def __init__(self):
        self.data = {}

    def update_identity(self, track_id, face, app, struct):

        if track_id not in self.data:
            self.data[track_id] = {
                "face": [],
                "app": [],
                "struct": []
            }

        if face is not None:
            self.data[track_id]["face"].append(face)

        if app is not None:
            self.data[track_id]["app"].append(app)

        if struct is not None:
            self.data[track_id]["struct"].append(struct)

    def mean_feature(self, features):
        if len(features) == 0:
            return None
        return np.mean(features, axis=0)

    def get_mean(self, track_id):

        if track_id not in self.data:
            return None, None, None

        return (
            self.mean_feature(self.data[track_id]["face"]),
            self.mean_feature(self.data[track_id]["app"]),
            self.mean_feature(self.data[track_id]["struct"])
        )