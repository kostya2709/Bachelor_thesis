import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMG_DIR = os.path.join(ROOT_DIR, "img")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
SRC_DIR = os.path.join(ROOT_DIR, "src")


MODELS_DIR = os.path.join(SRC_DIR, "models")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
