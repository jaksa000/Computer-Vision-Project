
import torch
from pathlib import Path

# =============================================================================
# ŚCIEŻKI
# =============================================================================
DATA_ROOT = Path("data")
# Który ekspert medyczny jako źródło etykiet: "MedicalExpert-I" lub "MedicalExpert-II"
EXPERT = "MedicalExpert-I"
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

# =============================================================================
# KLASY (etykiety)
# =============================================================================

CLASS_NAMES = ["0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"]
CLASS_DISPLAY_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# PREPROCESSING OBRAZÓW
# =============================================================================
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# PODZIAŁ DANYCH
# =============================================================================
TEST_RATIO  = 0.15
NUM_FOLDS   = 5
RANDOM_SEED = 42

# =============================================================================
# TRENING
# =============================================================================

BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LEARNING_RATE = 1e-4    # Krok uczenia (Adam optimizer)
WEIGHT_DECAY  = 1e-4    # Regularyzacja L2 (zapobiega overfittingowi)
#early stopping
PATIENCE = 5

# =============================================================================
# SPRZĘT
# =============================================================================


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0

# =============================================================================
# MODELE DO TRENOWANIA
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
