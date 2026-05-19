from pathlib import Path
import torch

# =========================================================
# BASE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"

LOGS_DIR = BASE_DIR / "logs"

# =========================================================
# DEVICE CONFIGURATION
# =========================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =========================================================
# IMAGE CONFIGURATION
# =========================================================

IMAGE_SIZE = 224
CHANNELS = 3

# =========================================================
# TRAINING CONFIGURATION
# =========================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25

# =========================================================
# DATASET CONFIGURATION
# =========================================================

CLASS_NAMES = [
    "cracked",
    "non_cracked"
]

NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "resnet18"

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42