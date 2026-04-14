
import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
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
# FUNKCJA: buduj DataLoadery dla jednego folda
# =============================================================================

def build_fold_dataloaders(all_samples, fold_idx):
    labels = [s[1] for s in all_samples]

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    splits = list(skf.split(all_samples, labels))
    train_idx, val_idx = splits[fold_idx]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

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

    train_dataset = KneeXrayDataset(train_samples, transform=get_transforms())
    val_dataset   = KneeXrayDataset(val_samples,   transform=get_transforms())

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

def get_transforms():
    normalize = transforms.Normalize(
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
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
        image = Image.open(img_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# FUNKCJA GŁÓWNA — wczytaj wszystkie sample (fold buduje build_fold_dataloaders)
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Ładowanie danych z: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Łącznie: {len(all_samples)} obrazów, {config.NUM_FOLDS} foldów CV")
    return all_samples
