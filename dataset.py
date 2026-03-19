
import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
# FUNKCJA: podział stratified na train/val/test
# =============================================================================

def stratified_split(samples,train_ratio,val_ratio,seed):
    random.seed(seed)
    by_class = {}
    for sample in samples:
        label = sample[1]
        if label not in by_class:
            by_class[label] = []
        by_class[label].append(sample)
    train_list, val_list, test_list = [], [], []

    for label, class_samples in sorted(by_class.items()):
        random.shuffle(class_samples)
        n = len(class_samples)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_list.extend(class_samples[:n_train])
        val_list.extend(class_samples[n_train:n_train + n_val])
        test_list.extend(class_samples[n_train + n_val:])

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    return train_list, val_list, test_list


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
# FUNKCJA GŁÓWNA
# =============================================================================

def build_dataloaders(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Ładowanie danych z: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.RANDOM_SEED,
    )

    print(f"\nPodział danych:")
    print(f"  Train: {len(train_samples)} ({len(train_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} ({len(val_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} ({len(test_samples)/len(all_samples)*100:.1f}%)")

    train_labels = [s[1] for s in train_samples]
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [
        total / (config.NUM_CLASSES * label_counts.get(i, 1))
        for i in range(config.NUM_CLASSES)
    ]
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"\nWagi klas (wyrównanie imbalance):")
    for i, (name, w) in enumerate(zip(config.CLASS_DISPLAY_NAMES, class_weights)):
        print(f"  Klasa {i} ({name}): waga = {w:.3f}  (count = {label_counts.get(i, 0)})")


    train_dataset = KneeXrayDataset(train_samples, transform=get_transforms())
    val_dataset   = KneeXrayDataset(val_samples,   transform=get_transforms())
    test_dataset  = KneeXrayDataset(test_samples,  transform=get_transforms())

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\nDataLoadery gotowe.")
    return train_loader, val_loader, test_loader, class_weights_tensor
