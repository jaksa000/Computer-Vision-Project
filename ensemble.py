import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import config
from models import build_model
from dataset import load_all_samples, split_holdout, build_test_dataloader
from evaluate import compute_metrics, print_summary_table


# =============================================================================
# 1. KLASY ENSEMBLI
# =============================================================================

class SimpleEnsemble(nn.Module):
    """
    Uśrednia prawdopodobieństwa modeli (Softmax Averaging).
    Działa zarówno dla Homogeneous (te same sieci) jak i Heterogeneous (różne).
    """

    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    @torch.no_grad()
    def forward(self, x):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)  # [liczba_modeli, batch_size, num_classes]
        avg_probs = torch.mean(stacked_probs, dim=0)  # [batch_size, num_classes]
        return avg_probs


class WeightedEnsemble(nn.Module):
    """
    Mixture of Experts: Waży głosy modeli na podstawie ich historycznego
    F1-score dla poszczególnych klas medycznych.
    """

    def __init__(self, models_list, weight_matrix):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        # weight_matrix ma wymiar [liczba_modeli, num_classes]
        self.weights = torch.tensor(weight_matrix, dtype=torch.float32).to(config.DEVICE)

        # Normalizujemy wagi, żeby dla każdej klasy suma głosów modeli wynosiła 1.0
        row_sums = self.weights.sum(dim=0, keepdim=True)
        self.weights = self.weights / (row_sums + 1e-8)

    @torch.no_grad()
    def forward(self, x):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)  # [liczba_modeli, batch_size, num_classes]

        # self.weights ma kształt [liczba_modeli, num_classes]
        # Zmieniamy kształt na [liczba_modeli, 1, num_classes] aby móc pomnożyć przez batche
        w = self.weights.unsqueeze(1)

        # Mnożymy prawdopodobieństwa przez wagi i sumujemy głosy modeli
        weighted_probs = torch.sum(stacked_probs * w, dim=0)  # [batch_size, num_classes]
        return weighted_probs


def build_mega_ensemble():
    """
    Realizuje Ensemble Typ D: Heterogeneous + Training Diversity.
    Ładuje WSZYSTKICH 25 ekspertów (5 architektur x 5 foldów).
    """
    print("\n" + "=" * 60)
    print("BUDOWANIE MEGA ENSEMBLE (Typ D: 25 modeli)")
    print("=" * 60)

    all_25_models = []

    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]
        print(f"Ładowanie modeli dla: {model_name}...")

        for fold_idx in range(config.NUM_FOLDS):
            checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{fold_idx + 1}_best.pt"

            if checkpoint_path.exists():
                # Budujemy szkielet i ładujemy wagi
                model = build_model(model_cfg)
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                all_25_models.append(model)
                print(f"  [+] Fold {fold_idx + 1} załadowany.")
            else:
                print(f"  [!] UWAGA: Brak pliku {checkpoint_path}")

    print(f"\nGotowe! Skompletowano komitet {len(all_25_models)} modeli.")

    # Używamy SimpleEnsemble (softmax averaging) dla wszystkich 25 modeli
    ensemble = SimpleEnsemble(all_25_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()

    return ensemble


# =============================================================================
# 2. FUNKCJE BUDUJĄCE KOMITETY Z DYSKU
# =============================================================================

def load_best_fold_for_model(model_cfg):
    """Znajduje i ładuje plik .pt z foldu, który miał najwyższe F1/Kappa."""
    model_name = model_cfg["name"]
    best_kappa = -1
    best_fold = -1
    best_metrics = None

    for fold_idx in range(config.NUM_FOLDS):
        json_path = config.RESULTS_DIR / f"{model_name}_fold{fold_idx + 1}_metrics.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                metrics = json.load(f)
                if metrics["cohen_kappa_Quadratic"] > best_kappa:
                    best_kappa = metrics["cohen_kappa_Quadratic"]
                    best_fold = fold_idx + 1
                    best_metrics = metrics

    if best_fold == -1:
        raise FileNotFoundError(f"Nie znaleziono logów JSON dla {model_name}. Uruchom najpierw main.py!")

    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{best_fold}_best.pt"
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, best_metrics, best_fold


def build_homogeneous_ensemble(model_cfg):
    """
    Buduje komitet z 5 modeli tej samej architeORTkury (z 5 foldów CV).
    """
    model_name = model_cfg["name"]
    print(f"\nSkładanie komitetu Homogeneous dla: {model_name}")

    loaded_models = []

    for fold_idx in range(config.NUM_FOLDS):
        checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{fold_idx + 1}_best.pt"

        # Budujemy szkielet i ładujemy wagi z danego foldu
        model = build_model(model_cfg)
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        loaded_models.append(model)
        print(f"  ✓ Załadowano eksperta z Foldu {fold_idx + 1}")

    # Pakujemy 5 modeli do naszej klasy Ensembla (zwykłe uśrednianie)
    ensemble = SimpleEnsemble(loaded_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()

    return ensemble

def build_heterogeneous_ensemble():
    """Buduje komitet z najlepszych modeli różnych architektur."""
    print("\n" + "=" * 60)
    print("Składanie Heterogeneous Ensemble (Najlepsze z każdej arch.)")
    loaded_models = []

    for model_cfg in config.MODELS_CONFIG:
        model, _, best_fold = load_best_fold_for_model(model_cfg)
        loaded_models.append(model)
        print(f"  ✓ Załadowano {model_cfg['name']} (zwycięzca: Fold {best_fold})")

    ensemble = SimpleEnsemble(loaded_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


def build_weighted_ensemble():
    """Buduje Mixture of Experts wykorzystując wagi z f1_per_class."""
    print("\n" + "=" * 60)
    print("Składanie Weighted Ensemble (Mixture of Experts na bazie F1)")
    loaded_models = []
    weight_matrix = []

    for model_cfg in config.MODELS_CONFIG:
        model, best_metrics, best_fold = load_best_fold_for_model(model_cfg)
        loaded_models.append(model)

        # Pobieramy F1 dla każdej klasy i dodajemy jako wiersz do macierzy wag
        f1_scores = best_metrics["f1_per_class"]
        weight_matrix.append(f1_scores)
        print(f"  ✓ {model_cfg['name']} (Fold {best_fold}) | Wagi F1: {[round(f, 2) for f in f1_scores]}")

    ensemble = WeightedEnsemble(loaded_models, weight_matrix)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


# =============================================================================
# 3. FUNKCJA EWALUACJI
# =============================================================================

@torch.no_grad()
def evaluate_ensemble(ensemble_name, ensemble_model, test_loader):
    all_labels, all_preds = [], []
    print(f"Ewaluacja {ensemble_name} na ostatecznym zbiorze testowym...")

    for images, labels in test_loader:
        images = images.to(config.DEVICE)
        probs = ensemble_model(images)
        _, preds = torch.max(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["model_name"] = ensemble_name

    # Zapis
    with open(config.RESULTS_DIR / f"{ensemble_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# =============================================================================
# 4. SKRYPT GŁÓWNY
# =============================================================================
def main():
    print("============================================================")
    print("TESTOWANIE WSZYSTKICH KOMITETÓW NA ZBIORZE HOLD-OUT")
    print("============================================================")

    all_samples = load_all_samples()
    _, test_samples = split_holdout(all_samples)
    test_loader = build_test_dataloader(test_samples)

    ensemble_results = []

    # =========================================================================
    # 1. HOMOGENEOUS ENSEMBLES (Typ B) - Komitety dla pojedynczych architektur
    # =========================================================================
    for model_cfg in config.MODELS_CONFIG:
        ensemble_name = f"{model_cfg['name']}_Homogeneous"

        # Używamy stworzonej wcześniej funkcji (upewnij się, że masz ją w pliku)
        hom_ensemble = build_homogeneous_ensemble(model_cfg)
        metrics_hom = evaluate_ensemble(ensemble_name, hom_ensemble, test_loader)
        ensemble_results.append(metrics_hom)

        del hom_ensemble
        torch.cuda.empty_cache()

    # =========================================================================
    # 2. HETEROGENEOUS ENSEMBLE (Typ C) - Najlepsze z każdej architektury (Średnia)
    # =========================================================================
    het_ensemble = build_heterogeneous_ensemble()
    metrics_het = evaluate_ensemble("Heterogeneous_Avg", het_ensemble, test_loader)
    ensemble_results.append(metrics_het)

    del het_ensemble
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. WEIGHTED ENSEMBLE (Mixture of Experts) - Ważone wg F1
    # =========================================================================
    wei_ensemble = build_weighted_ensemble()
    metrics_wei = evaluate_ensemble("Heterogeneous_Weighted", wei_ensemble, test_loader)
    ensemble_results.append(metrics_wei)

    del wei_ensemble
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. MEGA ENSEMBLE (Typ D) - Wszystkie 25 modeli z CV
    # =========================================================================
    mega_ensemble = build_mega_ensemble()
    metrics_mega = evaluate_ensemble("Mega_Ensemble_TypD", mega_ensemble, test_loader)
    ensemble_results.append(metrics_mega)

    del mega_ensemble
    torch.cuda.empty_cache()

    # =========================================================================
    # PODSUMOWANIE WYNIKÓW
    # =========================================================================
    print("\n\n" + "=" * 105)
    print("OSTATECZNE WYNIKI WSZYSTKICH ENSEMBLI NA ZBIORZE TESTOWYM (HOLD-OUT)")
    print_summary_table(ensemble_results)


if __name__ == "__main__":
    main()
