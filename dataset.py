import csv
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import Image, ImageOps

import config

# =============================================================================
# HELPERS FOR MSE-BASED IMAGE MATCHING
# =============================================================================
_MATCH_SIZE = (128, 128)

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


def image_to_vector(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize(_MATCH_SIZE, _RESAMPLE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()


# =============================================================================
# FUNCTION — loads paths and labels from a single expert folder
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
# FUNCTION — dual-expert labelling via Hungarian image matching
# =============================================================================

def load_expert_records(expert_root: Path):
    records = []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = expert_root / class_name
        if not class_folder.exists():
            print(f"  WARNING: folder not found — {class_folder}")
            continue
        for img_path in sorted(class_folder.iterdir()):
            if img_path.suffix.lower() != ".png":
                continue
            try:
                vec = image_to_vector(img_path)
                records.append({"path": img_path, "label": label_idx, "vector": vec})
            except Exception as e:
                print(f"  WARNING: could not load {img_path}: {e}")
    return records


def hungarian_match(records_a, records_b):
    A_mat = np.stack([r["vector"] for r in records_a])   # [N_a, D]
    B_mat = np.stack([r["vector"] for r in records_b])   # [N_b, D]

    D            = A_mat.shape[1]
    A_sq         = np.sum(A_mat ** 2, axis=1, keepdims=True)       # [N_a, 1]
    B_sq         = np.sum(B_mat ** 2, axis=1, keepdims=True).T     # [1, N_b]
    cost_matrix  = (A_sq + B_sq - 2.0 * np.dot(A_mat, B_mat.T)) / float(D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return [
        (records_a[r], records_b[c], float(cost_matrix[r, c]))
        for r, c in zip(row_ind, col_ind)
    ]


def save_cache(matches, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "label_a", "path_b", "label_b", "mse"])
        for rec_a, rec_b, mse in matches:
            writer.writerow([str(rec_a["path"]), rec_a["label"],
                             str(rec_b["path"]), rec_b["label"],
                             f"{mse:.10f}"])
    print(f"  Match cache saved: {cache_path}")


def load_cache(cache_path: Path):
    if not cache_path.exists():
        return None

    samples = []
    with open(cache_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_a   = int(row["label_a"])
            label_b   = int(row["label_b"])
            agreement = (
                config.CERTAIN_LABEL
                if label_a == label_b
                else config.UNCERTAIN_LABEL
            )
            samples.append((Path(row["path_a"]), label_a, agreement))
    return samples


def load_dual_expert_samples(data_root=config.DATA_ROOT,expert_1=config.EXPERT,expert_2=config.EXPERT_II,cache_path=config.MATCH_CACHE_CSV,force_rematch=False,):
    if not force_rematch:
        cached = load_cache(cache_path)
        if cached is not None:
            certain   = sum(1 for _, _, a in cached if a == config.CERTAIN_LABEL)
            uncertain = sum(1 for _, _, a in cached if a == config.UNCERTAIN_LABEL)
            print(f"\n  Loaded {len(cached)} matched pairs from cache: {cache_path}")
            print(f"  Certain: {certain}  Uncertain: {uncertain}")
            return cached

    print("\n" + "=" * 60)
    print("DUAL-EXPERT LABELLING — Hungarian image matching (128×128 MSE)")
    print("=" * 60)
    print(f"  Loading Expert-I  ({expert_1}.")
    records_a = load_expert_records(data_root / expert_1)
    print(f"  Loading Expert-II ({expert_2})")
    records_b = load_expert_records(data_root / expert_2)
    print(f"\n  Expert-I images:  {len(records_a)}")
    print(f"  Expert-II images: {len(records_b)}")

    matches = hungarian_match(records_a, records_b)
    save_cache(matches, cache_path)
    matched_samples = []
    for rec_a, rec_b, _ in matches:
        agreement = (
            config.CERTAIN_LABEL
            if rec_a["label"] == rec_b["label"]
            else config.UNCERTAIN_LABEL
        )
        matched_samples.append((rec_a["path"], rec_a["label"], agreement))

    certain_count   = sum(1 for _, _, a in matched_samples if a == config.CERTAIN_LABEL)
    uncertain_count = sum(1 for _, _, a in matched_samples if a == config.UNCERTAIN_LABEL)
    total           = len(matched_samples)
    print(f"\n  Images matched:               {total}")
    print(f"  Certain  (experts agree):     {certain_count:4d}  ({100*certain_count/total:.1f}%)")
    print(f"  Uncertain (experts disagree): {uncertain_count:4d}  ({100*uncertain_count/total:.1f}%)")
    return matched_samples


# =============================================================================
# FUNCTION — data split (holdout)
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
    print(f"  Hold-out data  (15%): {len(test_samples)} images")
    return cv_samples, test_samples


# =============================================================================
# FUNCTION — data loaders
# =============================================================================

def build_test_dataloader(test_samples):
    return DataLoader(
        KneeXrayDataset(test_samples, transform=get_transforms("val")),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )


def build_fold_dataloaders(cv_samples, fold_idx):
    labels = [s[1] for s in cv_samples]
    skf    = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True,
                              random_state=config.RANDOM_SEED)
    splits                 = list(skf.split(cv_samples, labels))
    train_idx, val_idx     = splits[fold_idx]
    train_samples          = [cv_samples[i] for i in train_idx]
    val_samples            = [cv_samples[i] for i in val_idx]

    print(f"\n  Fold {fold_idx + 1}/{config.NUM_FOLDS}:")
    print(f"    Train: {len(train_samples)} images")
    print(f"    Val:   {len(val_samples)} images")

    train_labels  = [s[1] for s in train_samples]
    label_counts  = Counter(train_labels)
    total         = sum(label_counts.values())
    class_weights = [
        total / (config.NUM_CLASSES * label_counts.get(i, 1))
        for i in range(config.NUM_CLASSES)
    ]
    class_weights_tensor = torch.FloatTensor(class_weights)

    print(f"\n    Class weights (fold {fold_idx + 1}):")
    for i, (name, w) in enumerate(zip(config.CLASS_DISPLAY_NAMES, class_weights)):
        print(f"      Class {i} ({name}): weight = {w:.3f}  (count = {label_counts.get(i, 0)})")

    train_loader = DataLoader(
        KneeXrayDataset(train_samples, transform=get_transforms("train")),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        KneeXrayDataset(val_samples, transform=get_transforms("val")),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, class_weights_tensor


# =============================================================================
# FUNCTION — transforms / augmentation
# =============================================================================

def get_transforms(split):
    img_size       = config.IMAGE_SIZE
    padding_buffer = 20
    if split == "train":
        return A.Compose([
            A.Resize(img_size + padding_buffer, img_size + padding_buffer,
                     interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomCrop(width=img_size, height=img_size),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.5),
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
# Dataset class
# =============================================================================

class KneeXrayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        label    = self.samples[idx][1]
        image    = cv2.imread(str(img_path))
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def get_agreement_labels(self):
        if len(self.samples) > 0 and len(self.samples[0]) == 3:
            return np.array([s[2] for s in self.samples])
        return None


# =============================================================================
# MAIN FUNCTION — single-expert loading used by main.py for training
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Loading data: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Together: {len(all_samples)} images, {config.NUM_FOLDS}-fold CV")
    return all_samples
