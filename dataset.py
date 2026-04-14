
import random
from pathlib import Path
from collections import Counter
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import Image

import config


# =============================================================================
# FUNKCJA: wczytaj ścieżki i etykiety z folderów
# =============================================================================

def load_image_paths(data_root=config.DATA_ROOT, expert=config.EXPERT):
    expert_folder = data_root / expert
    samples = []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = expert_folder / class_name
        image_extensions = {".png"}
        images_in_folder = [
            f for f in class_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        for img_path in images_in_folder:
            samples.append((img_path, label_idx))
        print(f"  Klasa {label_idx} ({class_name}): {len(images_in_folder)} obrazów")

    print(f"\n  Łącznie: {len(samples)} obrazów")
    return samples


# =============================================================================
# FUNKCJA: Odcięcie żelaznego sejfu (Hold-out Test Set)
# =============================================================================
def split_holdout(all_samples):
    labels = [s[1] for s in all_samples]

    cv_samples, test_samples = train_test_split(
        all_samples,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels  # Zapewnia równe proporcje klas w sejfie
    )

    print("\n" + "=" * 60)
    print("PODZIAŁ NA ZBIÓR CV ORAZ HOLD-OUT (SEJF)")
    print("=" * 60)
    print(f"  Dane do K-Fold CV (85%): {len(cv_samples)} obrazów")
    print(f"  Dane Testowe / Sejf (15%): {len(test_samples)} obrazów")

    return cv_samples, test_samples


# =============================================================================
# FUNKCJA: Zbuduj DataLoader dla sejfu
# =============================================================================
def build_test_dataloader(test_samples):
    test_dataset = KneeXrayDataset(test_samples, transform=get_transforms("val"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader

# =============================================================================
# FUNKCJA: buduj DataLoadery dla jednego folda
# =============================================================================

def build_fold_dataloaders(cv_samples, fold_idx):
    labels = [s[1] for s in cv_samples]

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    splits = list(skf.split(cv_samples, labels))
    train_idx, val_idx = splits[fold_idx]

    train_samples = [cv_samples[i] for i in train_idx]
    val_samples   = [cv_samples[i] for i in val_idx]

    print(f"\n  Fold {fold_idx + 1}/{config.NUM_FOLDS}:")
    print(f"    Train: {len(train_samples)} obrazów")
    print(f"    Val:   {len(val_samples)} obrazów")

    # Wagi klas liczone z części treningowej danego folda
    train_labels = [s[1] for s in train_samples]
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [
        total / (config.NUM_CLASSES * label_counts.get(i, 1))
        for i in range(config.NUM_CLASSES)
    ]
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"\n    Wagi klas (fold {fold_idx + 1}):")
    for i, (name, w) in enumerate(zip(config.CLASS_DISPLAY_NAMES, class_weights)):
        print(f"      Klasa {i} ({name}): waga = {w:.3f}  (count = {label_counts.get(i, 0)})")

        # Zmieniamy wywołania wewnątrz build_fold_dataloaders:
    train_dataset = KneeXrayDataset(train_samples, transform=get_transforms("train"))
    val_dataset = KneeXrayDataset(val_samples, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights_tensor


# =============================================================================
# TRANSFORMACJE
# =============================================================================

def get_transforms(split: str) -> A.Compose:
    """
    Zwraca transformacje dla danego splitu (używając Albumentations).

    split = "train" → augmentacja (losowe przekształcenia + CLAHE)
    split = "val" lub "test" → tylko resize + normalize (deterministyczne!)

    Augmentacje dobrane dla RTG kolana:
    - Horizontal flip ✓ (kolano lewe/prawe wygląda odwrotnie, ale zmiany są symetryczne)
    - Rotation ±10° ✓ (małe obroty są realistyczne — różne ułożenie pacjenta)
    - CLAHE + Brightness/Contrast ✓ (wyciąganie detali kości i szpary stawowej, różne ekspozycje)
    - Vertical flip ✗ NIE — kolano zawsze jest na górze/dole w określony sposób
    - Duże skalowanie ✗ NIE — mogłoby ukryć ważne szczegóły anatomiczne
    """
    img_size = config.IMAGE_SIZE
    padding_buffer = 20

    if split == "train":
        return A.Compose([
            # 1. Powiększenie z buforem (używamy interpolacji liniowej)
            A.Resize(img_size + padding_buffer, img_size + padding_buffer, interpolation=cv2.INTER_LINEAR),

            # 2. Rotacja (border_mode=cv2.BORDER_CONSTANT wstawia czarne tło w rogi)
            A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

            # 3. Wycięcie docelowego rozmiaru (pozbywamy się czarnych rogów z rotacji)
            A.RandomCrop(width=img_size, height=img_size),

            # 4. Geometria symetryczna
            A.HorizontalFlip(p=0.5),

            # 5. Fotometria: Medyczne CLAHE oraz lekka korekta ekspozycji
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),

            # 6. Normalizacja i konwersja na Tensor (zawsze na samym końcu)
            A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            ToTensorV2(),
        ])
    else:
        # Val i test: deterministyczny resize i normalize
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            ToTensorV2(),
        ])

# =============================================================================
# KLASA DATASET
# =============================================================================

class KneeXrayDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # 1. Wczytujemy obraz przez OpenCV
        image = cv2.imread(str(img_path))
        # OpenCV domyślnie czyta kolory w formacie BGR, a modele oczekują RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            # 2. Albumentations zwraca słownik, z którego wyciągamy przetworzony obraz
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# =============================================================================
# FUNKCJA GŁÓWNA — wczytaj wszystkie sample (fold buduje build_fold_dataloaders)
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Ładowanie danych z: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Łącznie: {len(all_samples)} obrazów, {config.NUM_FOLDS} foldów CV")
    return all_samples
