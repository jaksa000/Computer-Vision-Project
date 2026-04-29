import random
from pathlib import Path
from collections import Counter

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

import config


# =============================================================================
# FUNKCJA: wczytaj ścieżki i etykiety KL z folderów jednego eksperta
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
        print(f"  Klasa {label_idx} ({class_name}): {len(images_in_folder)} obrazów")

    print(f"\n  Łącznie: {len(samples)} obrazów")
    return samples


# =============================================================================
# FUNKCJA: Dual-expert labeling — certain (0) / uncertain (1)
# =============================================================================

def load_dual_expert_samples(
    data_root: Path = config.DATA_ROOT,
    expert_i: str = config.EXPERT,
    expert_ii: str = config.EXPERT_II,
) -> list[tuple[Path, int, int]]:
    """
    Dopasowuje zdjęcia z folderu Expert-I i Expert-II po nazwie pliku.

    Zwraca listę trójek: (img_path, kl_label, agreement_label)
      - img_path        : ścieżka do zdjęcia z Expert-I (używamy Expert-I jako źródła)
      - kl_label        : etykieta KL (0–4) wg Expert-I
      - agreement_label : config.CERTAIN_LABEL (0) jeśli eksperci zgodni,
                          config.UNCERTAIN_LABEL (1) jeśli niezgodni
    """

    def _build_filename_map(expert_root: Path) -> dict[str, tuple[Path, int]]:
        """Słownik: nazwa_pliku -> (pełna_ścieżka, etykieta_KL)"""
        mapping: dict[str, tuple[Path, int]] = {}
        for label_idx, class_name in enumerate(config.CLASS_NAMES):
            class_folder = expert_root / class_name
            if not class_folder.exists():
                print(f"  UWAGA: brak folderu {class_folder}")
                continue
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() == ".png":
                    mapping[img_path.name] = (img_path, label_idx)
        return mapping

    map_i  = _build_filename_map(data_root / expert_i)
    map_ii = _build_filename_map(data_root / expert_ii)

    matched: list[tuple[Path, int, int]] = []
    not_found_in_ii = 0

    for filename, (path_i, label_i) in map_i.items():
        if filename not in map_ii:
            not_found_in_ii += 1
            continue
        _, label_ii = map_ii[filename]
        agreement = (
            config.CERTAIN_LABEL
            if label_i == label_ii
            else config.UNCERTAIN_LABEL
        )
        matched.append((path_i, label_i, agreement))

    # --- Statystyki ---
    certain_count   = sum(1 for _, _, a in matched if a == config.CERTAIN_LABEL)
    uncertain_count = sum(1 for _, _, a in matched if a == config.UNCERTAIN_LABEL)
    total = len(matched)

    print("\n" + "=" * 60)
    print("DUAL-EXPERT LABELING — CERTAIN vs UNCERTAIN")
    print("=" * 60)
    print(f"  Zdjęcia w Expert-I:            {len(map_i)}")
    print(f"  Zdjęcia w Expert-II:           {len(map_ii)}")
    print(f"  Dopasowane (oba eksperci):     {total}")
    if not_found_in_ii:
        print(f"  Pominięte (brak w Expert-II): {not_found_in_ii}")
    print(f"\n  Pewne   (eksperci zgodni):    {certain_count:4d}  ({100*certain_count/total:.1f}%)")
    print(f"  Niepewne (eksperci różnią się): {uncertain_count:4d}  ({100*uncertain_count/total:.1f}%)")

    # Macierz niezgodności po klasach KL
    _print_expert_confusion_matrix(matched)

    return matched


def _print_expert_confusion_matrix(
    matched: list[tuple[Path, int, int]]
) -> None:
    """
    Drukuje macierz: ile razy Expert-I i Expert-II przypisali inne klasy KL.
    Pomaga zrozumieć, między którymi stopniami KL najczęściej zachodzi niezgodność.
    """
    from collections import defaultdict
    import numpy as np

    # Odbuduj parę (label_i, label_ii) dla niezgodnych
    disagreements: dict[tuple[int, int], int] = defaultdict(int)

    # Potrzebujemy label_ii — ładujemy go ponownie
    map_ii = _build_filename_map_quiet(
        config.DATA_ROOT / config.EXPERT_II
    )

    for path_i, label_i, agreement in matched:
        if agreement == config.UNCERTAIN_LABEL:
            filename = path_i.name
            if filename in map_ii:
                _, label_ii = map_ii[filename]
                disagreements[(label_i, label_ii)] += 1

    if not disagreements:
        return

    print("\n  Macierz niezgodności (Expert-I vs Expert-II):")
    print("  (pokazuje tylko pary z niezgodnością)")
    header = "  Expert-I \\ II |" + "".join(f"  KL{j}" for j in range(config.NUM_CLASSES))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(config.NUM_CLASSES):
        row_vals = [disagreements.get((i, j), 0) for j in range(config.NUM_CLASSES)]
        if any(v > 0 for j, v in enumerate(row_vals) if i != j):
            row_str = f"  KL{i}          |" + "".join(f"  {v:3d}" for v in row_vals)
            print(row_str)


def _build_filename_map_quiet(expert_root: Path) -> dict[str, tuple[Path, int]]:
    """Wersja bez printu — używana wewnętrznie."""
    mapping: dict[str, tuple[Path, int]] = {}
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = expert_root / class_name
        if not class_folder.exists():
            continue
        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() == ".png":
                mapping[img_path.name] = (img_path, label_idx)
    return mapping


# =============================================================================
# FUNKCJA: Hold-Out
# =============================================================================

def split_holdout(all_samples):
    """Przyjmuje listę (path, kl_label) LUB (path, kl_label, agreement_label)."""
    labels = [s[1] for s in all_samples]

    cv_samples, test_samples = train_test_split(
        all_samples,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=labels,
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
# TRANSFORMACJE
# =============================================================================

def get_transforms(split: str) -> A.Compose:
    img_size = config.IMAGE_SIZE
    padding_buffer = 20

    if split == "train":
        return A.Compose([
            A.Resize(img_size + padding_buffer, img_size + padding_buffer, interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
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
# KLASA DATASET
# =============================================================================

class KneeXrayDataset(Dataset):
    """
    Obsługuje zarówno próbki (path, kl_label) jak i (path, kl_label, agreement_label).
    Podczas __getitem__ zawsze zwraca (image, kl_label) — agreement_label jest
    przechowywany oddzielnie i używany w analizie UQ po zakończeniu predykcji.
    """

    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Próbka może być dwójką lub trójką — bierzemy pierwsze dwa elementy
        img_path, label = self.samples[idx][0], self.samples[idx][1]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_agreement_labels(self) -> np.ndarray | None:
        """
        Zwraca tablicę etykiet agreement (certain/uncertain) jeśli próbki
        zawierają tę informację (trójki). W przeciwnym razie zwraca None.
        """
        if len(self.samples) > 0 and len(self.samples[0]) == 3:
            return np.array([s[2] for s in self.samples])
        return None


# =============================================================================
# FUNKCJA GŁÓWNA — wczytaj wszystkie sample KL (trening/CV)
# =============================================================================

def load_all_samples(data_root=config.DATA_ROOT, expert=config.EXPERT):
    print(f"Ładowanie danych z: {data_root / expert}")
    all_samples = load_image_paths(data_root, expert)
    print(f"  Łącznie: {len(all_samples)} obrazów, {config.NUM_FOLDS} foldów CV")
    return all_samples
