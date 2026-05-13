import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

import config

# =============================================================================
# HASHING FUNCTION
# =============================================================================

def get_image_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# =============================================================================
# FUNCTION- loads paths and labels from each expert
# =============================================================================

def load_image_paths(data_root=config.DATA_ROOT, expert=config.EXPERT):
    expert_folder = data_root / expert
    samples = []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = expert_folder / class_name
        images_in_folder = [
            f for f in class_folder.iterdir()
            if f.suffix.lower() == ".png"
        ]
        for img_path in images_in_folder:
            samples.append((img_path, label_idx))
        print(f"  Class {label_idx} ({class_name}): {len(images_in_folder)} images")

    print(f"\n  Together: {len(samples)} images")
    return samples


# =============================================================================
# FUNCTION- Dual-expert labeling (Zbudowane o HASH zamiast nazwy pliku)
# =============================================================================

def build_hash_map(expert_root, quiet=False):
    mapping = {}
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = expert_root / class_name
        if not class_folder.exists():
            if not quiet:
                print(f" FOLDER WITH THAT NAME DOESNT EXIST {class_folder}")
            continue
        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() == ".png":
                file_hash = get_image_hash(img_path)
                mapping[file_hash] = (img_path, label_idx)
    return mapping


# =============================================================================
# FUNCTION- Confusion Matrix
# =============================================================================
def print_expert_confusion_matrix(matched, map_1, map_2):
    path_to_hash = {str(path): file_hash for file_hash, (path, _) in map_1.items()}

    disagreements = defaultdict(int)
    for path_1, label_1, agreement in matched:
        if agreement == config.UNCERTAIN_LABEL:
            file_hash = path_to_hash.get(str(path_1))

            if file_hash and file_hash in map_2:
                _, label_2 = map_2[file_hash]
                disagreements[(label_1, label_2)] += 1

    if not disagreements:
        return

    print("\n  Confusion Matrix (Expert-I vs Expert-II):")
    print("  (shows only disagreements )")
    header = "  Expert-I \\ II |" + "".join(f"  KL{j}" for j in range(config.NUM_CLASSES))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(config.NUM_CLASSES):
        row_vals = [disagreements.get((i, j), 0) for j in range(config.NUM_CLASSES)]
        if any(v > 0 for j, v in enumerate(row_vals) if i != j):
            row_str = f"  KL{i}          |" + "".join(f"  {v:3d}" for v in row_vals)
            print(row_str)


def load_dual_expert_samples(
        data_root=config.DATA_ROOT,
        expert_1=config.EXPERT,
        expert_2=config.EXPERT_II,
):

    map_1 = build_hash_map(data_root / expert_1)
    map_2 = build_hash_map(data_root / expert_2)

    matched = []
    not_found_in_2 = 0
    for file_hash, (path_1, label_1) in map_1.items():
        if file_hash not in map_2:
            not_found_in_2 += 1
            continue

        _, label_2 = map_2[file_hash]
        agreement = (
            config.CERTAIN_LABEL
            if label_1 == label_2
            else config.UNCERTAIN_LABEL
        )
        matched.append((path_1, label_1, agreement))

    # Statistics
    certain_count = sum(1 for _, _, a in matched if a == config.CERTAIN_LABEL)
    uncertain_count = sum(1 for _, _, a in matched if a == config.UNCERTAIN_LABEL)
    total = len(matched)

    print("\n" + "=" * 60)
    print("DUAL-EXPERT LABELING — CERTAIN vs UNCERTAIN (HASH MATCHING)")
    print("=" * 60)
    print(f"  Images in Expert-I:            {len(map_1)}")
    print(f"  Images in Expert-II:           {len(map_2)}")
    print(f"  Images matched:                {total}")
    if not_found_in_2:
        print(f"  Skipped, missing in Expert-II: {not_found_in_2}")
    print(f"\n  Certain Labels (Experts agree): {certain_count:4d}  ({100 * certain_count / total:.1f}%)")
    print(f"  Uncertain Labels:               {uncertain_count:4d}  ({100 * uncertain_count / total:.1f}%)")

    print_expert_confusion_matrix(matched, map_1, map_2)

    return matched



# =============================================================================
# FUNCTION- Data split
# =============================================================================

def split_holdout(all_samples):
    labels = [s[1] for s in all_samples]
    cv_samples, test_samples = train_test_split(
        all_samples,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    print("\n" + "=" * 60)
    print(" DATA SPLIT")
    print("=" * 60)
    print(f"  K-Fold CV data (85%): {len(cv_samples)} images")
    print(f"  Hold-out data (15%): {len(test_samples)} images")

    return cv_samples, test_samples


# =============================================================================
# FUNCTION- Data loaders
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
    print(f"    Train: {len(train_samples)} images")
    print(f"    Val:   {len(val_samples)} images")

    train_labels = [s[1] for s in train_samples]
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    class_weights = [
        total / (config.NUM_CLASSES * label_counts.get(i, 1))
        for i in range(config.NUM_CLASSES)
    ]
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"\n    Class weights (fold {fold_idx + 1}):")
    for i, (name, w) in enumerate(zip(config.CLASS_DISPLAY_NAMES, class_weights)):
        print(f"      Class {i} ({name}): weight = {w:.3f}  (count = {label_counts.get(i, 0)})")

    train_dataset = KneeXrayDataset(train_samples, transform=get_transforms("train"))
    val_dataset   = KneeXrayDataset(val_samples,   transform=get_transforms("val"))

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
# FUNCTION- Normalisation and Trahsformation
# =============================================================================

def get_transforms(split):
    img_size = config.IMAGE_SIZE
    padding_buffer = 20

    if split == "train":
        return A.Compose([
            A.Resize(img_size + padding_buffer, img_size + padding_buffer, interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomCrop(width=img_size, height=img_size),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
            ToTensorV2(),
        ])


# =============================================================================
# Dataset Class
# =============================================================================

class KneeXrayDataset(Dataset):
    """
Supports both (path, kl_label) and (path, kl_label, agreement_label) samples.
While __getitem__ always returns (image, kl_label),
agreement_label is stored separately and used in UQ analysis after the prediction is complete.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx][0], self.samples[idx][1]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_agreement_labels(self):
        if len(self.samples) > 0 and len(self.samples[0]) == 3:
            return np.array([s[2] for s in self.samples])
        return None


# =============================================================================
# MAIN FUNCTION - Loading all samples
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Loading data: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Together: {len(all_samples)} images, {config.NUM_FOLDS}  folds CV")
    return all_samples