from pathlib import Path
import torch

# =====================================================
# PROJECT
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# DATASET
# =====================================================

DATASET_DIR = BASE_DIR / "dataset" / "food-101"

IMAGES_DIR = DATASET_DIR / "images"

META_DIR = DATASET_DIR / "meta"

TRAIN_FILE = META_DIR / "train.txt"
TEST_FILE = META_DIR / "test.txt"
CLASSES_FILE = META_DIR / "classes.txt"

# =====================================================
# MODELS
# =====================================================

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "resnet50_best.pth"

NUM_CLASSES = 101

# =====================================================
# METRICS
# =====================================================

METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

BEST_MODEL_TXT = METRICS_DIR / "best_model.txt"

# =====================================================
# HISTORY
# =====================================================

HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

DB_PATH = HISTORY_DIR / "history.db"

# =====================================================
# FLASK
# =====================================================

STATIC_DIR = BASE_DIR / "static"

UPLOAD_FOLDER = STATIC_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# =====================================================
# TRAINING
# =====================================================

DEVICE = torch.device("cuda")

IMAGE_SIZE = 224

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

EPOCHS = 10

PATIENCE = 5

NUM_WORKERS = 2

# =====================================================
# RANDOM
# =====================================================

SEED = 42