import torch
from pathlib import Path

# =============================================================================
# File Paths
# =============================================================================
DATA_ROOT = Path("data")
# Which expert is source of labels
EXPERT = "MedicalExpert-I"
# Second expert used to designate certain/uncertain
EXPERT_II = "MedicalExpert-II"

CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

# =============================================================================
# Classes
# =============================================================================
CLASS_NAMES = ["0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"]
CLASS_DISPLAY_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# CERTAIN / UNCERTAIN Labels
# =============================================================================
CERTAIN_LABEL = 0   # CERTAIN_LABEL   = 0  when both experts agree on label (l1 == l2)
UNCERTAIN_LABEL = 1 # UNCERTAIN_LABEL = 1  when both experts disagree about label  (l1 != l2)
AGREEMENT_CLASS_NAMES = ["Certain", "Uncertain"]

# Ensemble Uncertainty treshold
#   threshold = mean(std) + UNCERTAINTY_SIGMA_MULTIPLIER * std(std)
UNCERTAINTY_SIGMA_MULTIPLIER = 3  # 3 sigma rule

# =============================================================================
# Image Preprocessing
# =============================================================================
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# Data split, folds number and random seed
# =============================================================================
TEST_RATIO  = 0.15
NUM_FOLDS   = 5
RANDOM_SEED = 123

# =============================================================================
#  Training
# =============================================================================
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5

# =============================================================================
#  Use gpu with cuda capability if possible
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

# =============================================================================
# Models
# =============================================================================
MODELS_CONFIG = [
    {
        "name": "resnet50",
        "timm_id": "resnet50",
        "pretrained": True,
        "description": "ResNet-50"
    },
    {
        "name": "efficientnet_b3",
        "timm_id": "efficientnet_b3",
        "pretrained": True,
        "description": "EfficientNet-B3"
    },
    {
        "name": "densenet121",
        "timm_id": "densenet121",
        "pretrained": True,
        "description": "DenseNet-121"
    },
    {
        "name": "mobilenetv3_large",
        "timm_id": "mobilenetv3_large_100",
        "pretrained": True,
        "description": "MobileNetV3"
    },
    {
        "name": "convnext_tiny",
        "timm_id": "convnext_tiny",
        "pretrained": True,
        "description": "ConvNeXt-Tiny"
    },
]
