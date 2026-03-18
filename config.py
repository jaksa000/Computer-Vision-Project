
import torch
from pathlib import Path

# =============================================================================
# ŚCIEŻKI
# =============================================================================

# Główny folder z danymi — zmień na swoją ścieżkę
DATA_ROOT = Path("data")

# Który ekspert medyczny jako źródło etykiet: "MedicalExpert-I" lub "MedicalExpert-II"
EXPERT = "MedicalExpert-I"

# Gdzie zapisywać wytrenowane modele i wyniki
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

# =============================================================================
# KLASY (etykiety)
# =============================================================================

# Nazwy folderów = nazwy klas. Kolejność ważna — to będą indeksy 0,1,2,3,4.
CLASS_NAMES = ["0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"]

# Ładniejsze nazwy do wykresów i raportów
CLASS_DISPLAY_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]

NUM_CLASSES = len(CLASS_NAMES)  # 5

# =============================================================================
# PREPROCESSING OBRAZÓW
# =============================================================================

# Rozmiar do jakiego skalujemy obrazy przed podaniem do sieci
# 224x224 = standard dla większości architektur
IMAGE_SIZE = 224

# Normalizacja — wartości z ImageNet (pretraining był na tych danych)
# Używamy ich bo nasze modele są pretrenowane na ImageNet
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# PODZIAŁ DANYCH
# =============================================================================

# Proporcje: 70% trening, 15% walidacja, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Seed dla reproducibility — zawsze ten sam split przy tym samym seedzie
RANDOM_SEED = 42

# =============================================================================
# TRENING
# =============================================================================

BATCH_SIZE   = 32       # Ile obrazów na raz widzi GPU/CPU
NUM_EPOCHS   = 20       # Ile razy przejdziemy przez cały dataset
LEARNING_RATE = 1e-4    # Krok uczenia (Adam optimizer)
WEIGHT_DECAY  = 1e-4    # Regularyzacja L2 (zapobiega overfittingowi)

# Ile epok bez poprawy zanim zatrzymamy trening (early stopping)
PATIENCE = 5

# =============================================================================
# SPRZĘT
# =============================================================================

# Automatycznie wybierz GPU jeśli dostępne, inaczej CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ile wątków do ładowania danych (0 = główny wątek, bezpieczne na Windows)
NUM_WORKERS = 0

# =============================================================================
# MODELE DO TRENOWANIA
# =============================================================================

# Każdy model to słownik z:
#   "name"     — identyfikator (do nazwy pliku)
#   "timm_id"  — nazwa w bibliotece timm (szukaj na timm.fast.ai)
#   "pretrained" — czy startować z wag ImageNet (prawie zawsze True)

MODELS_CONFIG = [
    {
        "name": "resnet50",
        "timm_id": "resnet50",
        "pretrained": True,
        "description": "ResNet-50 — klasyk, dobry baseline"
    },
    {
        "name": "efficientnet_b3",
        "timm_id": "efficientnet_b3",
        "pretrained": True,
        "description": "EfficientNet-B3 — dobry stosunek dokładności do rozmiaru"
    },
    {
        "name": "densenet121",
        "timm_id": "densenet121",
        "pretrained": True,
        "description": "DenseNet-121 — popularny w medycznym imagingu"
    },
    {
        "name": "mobilenetv3_large",
        "timm_id": "mobilenetv3_large_100",
        "pretrained": True,
        "description": "MobileNetV3 — lekki, szybki"
    },
    {
        "name": "convnext_tiny",
        "timm_id": "convnext_tiny",
        "pretrained": True,
        "description": "ConvNeXt-Tiny — nowoczesna architektura"
    },
]
