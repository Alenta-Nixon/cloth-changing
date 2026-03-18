import pandas as pd
import os

query_dir = "test_cases/query"
frame_dir = "test_cases/frames"

queries = os.listdir(query_dir)
frames = os.listdir(frame_dir)

data = []

for q in queries:
    for f in frames:

        # ⚠️ You must manually set correct matches
        label = 0

        # example logic (you edit this)
        if "p1" in q and "f0" in f:
            label = 1

        if "p2" in q and "f10" in f:
            label = 1

        data.append([q, f, label])

df = pd.DataFrame(data, columns=["query","frame","label"])
df.to_csv("test_cases/labels.csv", index=False)

print("labels.csv created")