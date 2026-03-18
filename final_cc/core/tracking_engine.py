from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.3,
    nn_budget=100
)

def update_tracks(detections, frame):

    tracks = tracker.update_tracks(detections, frame=frame)

    valid_tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > 1:
            continue

        l, t, r, b = track.to_ltrb()
        valid_tracks.append((
            track.track_id,
            int(l), int(t), int(r), int(b)
        ))

    return valid_tracks