import pickle
import os

DB_PATH = "identity_db.pkl"

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)