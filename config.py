import torch
from pathlib import Path

# =============================================================================
# File Paths
# =============================================================================
DATA_ROOT = Path("data")
EXPERT    = "MedicalExpert-I"
EXPERT_II = "MedicalExpert-II"

CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR     = Path("results")

# Structured Results Directories
INDIVIDUAL_MODELS_DIR = RESULTS_DIR / "individual_models"
ENSEMBLES_DIR         = RESULTS_DIR / "ensembles"
FIGURES_DIR           = RESULTS_DIR / "figures"
LOGS_DIR              = RESULTS_DIR / "logs"

# CSV cache produced by load_dual_expert_samples() on first run.
MATCH_CACHE_CSV = RESULTS_DIR / "expert_matches_cache.csv"

# Persistent split manifest shared by main.py and ensemble.py.
# main.py creates it; ensemble.py reuses it to guarantee the identical holdout set.
SPLIT_MANIFEST_JSON = RESULTS_DIR / "split_manifest.json"

# Paths of samples flagged as uncertain by Mega_Ensemble (written by ensemble.py,
# consumed by explainability.py to generate the system-flagged Grad-CAM category).
SYSTEM_FLAGGED_JSON = RESULTS_DIR / "system_flagged_paths.json"

# =============================================================================
# Classes
# =============================================================================
CLASS_NAMES         = ["0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"]
CLASS_DISPLAY_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
NUM_CLASSES         = len(CLASS_NAMES)

# =============================================================================
# CERTAIN / UNCERTAIN labels
# =============================================================================
CERTAIN_LABEL         = 0
UNCERTAIN_LABEL       = 1
AGREEMENT_CLASS_NAMES = ["Certain", "Uncertain"]

# =============================================================================
# Uncertainty threshold
# =============================================================================
UNCERTAINTY_PERCENTILE       = 95
UNCERTAINTY_SIGMA_MULTIPLIER = 3

# =============================================================================
# Image preprocessing
# =============================================================================
IMAGE_SIZE     = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# Data split, folds, random seed
# =============================================================================
TEST_RATIO  = 0.15
NUM_FOLDS   = 5
RANDOM_SEED = 108

# =============================================================================
# Training
# NUM_EPOCHS increased 20 → 30 to give models more room to converge.
# PATIENCE  increased  5 → 8  so early stopping doesn't fire before the
# LR scheduler (patience=3 in train.py) has a chance to reduce the rate twice.
# =============================================================================
BATCH_SIZE    = 64
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 8

# =============================================================================
# Device
# =============================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

# =============================================================================
# Models
# =============================================================================
MODELS_CONFIG = [
    {"name": "resnet50",          "timm_id": "resnet50",              "pretrained": True, "description": "ResNet-50"},
    {"name": "efficientnet_b3",   "timm_id": "efficientnet_b3",       "pretrained": True, "description": "EfficientNet-B3"},
    {"name": "densenet121",       "timm_id": "densenet121",           "pretrained": True, "description": "DenseNet-121"},
    {"name": "mobilenetv3_large", "timm_id": "mobilenetv3_large_100", "pretrained": True, "description": "MobileNetV3"},
    {"name": "convnext_tiny",     "timm_id": "convnext_tiny",         "pretrained": True, "description": "ConvNeXt-Tiny"},
]
