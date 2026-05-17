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


def _image_to_vector(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize(_MATCH_SIZE, _RESAMPLE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


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
# FUNCTION — dual-expert labelling via MSE-based Hungarian image matching
#
# Strategy:
#   1. Load every image from both expert folders as a 128×128 grayscale vector.
#   2. Compute the full pairwise MSE cost matrix using an optimized dot-product
#      expansion to drastically save CPU cycles and prevent RAM overflow.
#   3. Solve the linear sum assignment problem (Hungarian algorithm) to find the
#      globally optimal, order-independent bijective mapping between experts.
#   4. If both experts placed the matched pair in the same KL class → CERTAIN.
#      If they placed it in different KL classes → UNCERTAIN.
#
# Why MSE and not MD5 hashing?
#   File-byte MD5 fails when the same radiograph is stored with different PNG
#   metadata (timestamps, software tags, compression settings) across folders.
#   Pixel-level MD5 can similarly fail after EXIF rotation or colour-profile
#   normalisation. The MSE approach is robust to all of these because it
#   matches on visual content, not on byte identity.
# =============================================================================

def _load_expert_records(expert_root: Path):
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
                vec = _image_to_vector(img_path)
                records.append({
                    "path": img_path,
                    "label": label_idx,
                    "vector": vec,
                })
            except Exception as e:
                print(f"  WARNING: could not load {img_path}: {e}")
    return records


def _hungarian_match(records_a, records_b):
    A = np.stack([r["vector"] for r in records_a])
    B = np.stack([r["vector"] for r in records_b])

    # Efficient matrix-based pairwise MSE computation: (||A||^2 + ||B||^2 - 2AB^T) / D
    # This prevents allocating massive 3D arrays and keeps memory footprint minimal
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True).T
    AB_prod = np.dot(A, B.T)

    D_features = A.shape[1]
    cost_matrix = (A_sq + B_sq - 2.0 * AB_prod) / float(D_features)

    # Global linear sum assignment optimization (Kuhn-Munkres)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        matches.append((records_a[r], records_b[c], float(cost_matrix[r, c])))
    return matches


# =============================================================================
# FUNCTION — expert confusion matrix (disagreements only)
# =============================================================================

def _print_expert_confusion_matrix(matched_triples):
    disagreements = defaultdict(int)
    for rec_a, rec_b, _ in matched_triples:
        if rec_a["label"] != rec_b["label"]:
            disagreements[(rec_a["label"], rec_b["label"])] += 1

    if not disagreements:
        print("  No disagreements found.")
        return

    print("\n  Expert Confusion Matrix (disagreements only):")
    header = "  Expert-I \\ II |" + "".join(f"  KL{j}" for j in range(config.NUM_CLASSES))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(config.NUM_CLASSES):
        row = [disagreements.get((i, j), 0) for j in range(config.NUM_CLASSES)]
        if any(row[j] > 0 for j in range(config.NUM_CLASSES) if j != i):
            print(f"  KL{i}          |" + "".join(f"  {v:3d}" for v in row))


# =============================================================================
# FUNCTION — main dual-expert loading entry point
# =============================================================================

def load_dual_expert_samples(
        data_root=config.DATA_ROOT,
        expert_1=config.EXPERT,
        expert_2=config.EXPERT_II,
):
    root_1 = data_root / expert_1
    root_2 = data_root / expert_2

    print("\n" + "=" * 60)
    print("DUAL-EXPERT LABELLING — MSE-based Hungarian image matching")
    print("=" * 60)
    print(f"  Loading Expert-I  ({expert_1})...")
    records_a = _load_expert_records(root_1)
    print(f"  Loading Expert-II ({expert_2})...")
    records_b = _load_expert_records(root_2)

    print(f"\n  Expert-I images:  {len(records_a)}")
    print(f"  Expert-II images: {len(records_b)}")
    print(f"  Matching (globally optimal Hungarian algorithm on {_MATCH_SIZE[0]}x{_MATCH_SIZE[1]} grayscale)...")

    matches = _hungarian_match(records_a, records_b)
    matched_samples = []
    for rec_a, rec_b, _ in matches:
        agreement = (
            config.CERTAIN_LABEL
            if rec_a["label"] == rec_b["label"]
            else config.UNCERTAIN_LABEL
        )
        matched_samples.append((rec_a["path"], rec_a["label"], agreement))

    certain_count = sum(1 for _, _, a in matched_samples if a == config.CERTAIN_LABEL)
    uncertain_count = sum(1 for _, _, a in matched_samples if a == config.UNCERTAIN_LABEL)
    total = len(matched_samples)

    print(f"\n  Images matched:                 {total}")
    print(f"  Certain  (experts agree):       {certain_count:4d}  ({100 * certain_count / total:.1f}%)")
    print(f"  Uncertain (experts disagree):   {uncertain_count:4d}  ({100 * uncertain_count / total:.1f}%)")

    _print_expert_confusion_matrix(matches)

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
    test_dataset = KneeXrayDataset(test_samples, transform=get_transforms("val"))
    return DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )


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
    val_samples = [cv_samples[i] for i in val_idx]

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
        print(f"      Class {i} ({name}): weight = {w:.3f}  "
              f"(count = {label_counts.get(i, 0)})")

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
# FUNCTION — transforms / augmentation
# =============================================================================

def get_transforms(split):
    img_size = config.IMAGE_SIZE
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
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        label = self.samples[idx][1]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_agreement_labels(self):
        if len(self.samples) > 0 and len(self.samples[0]) == 3:
            return np.array([s[2] for s in self.samples])
        return None


# =============================================================================
# MAIN FUNCTION — simple single-expert loading (used by main.py for training)
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Loading data: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Together: {len(all_samples)} images, {config.NUM_FOLDS}-fold CV")
    return all_samples