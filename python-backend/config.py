import os

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
