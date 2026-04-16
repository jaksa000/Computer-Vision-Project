import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import config
from models import build_model
from dataset import load_all_samples, split_holdout, build_test_dataloader
from evaluate import compute_metrics


# =============================================================================
# 1. KLASY ENSEMBLI
# =============================================================================

class SimpleEnsemble(nn.Module):
    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)  # [liczba_modeli, batch_size, num_classes]
        avg_probs = torch.mean(stacked_probs, dim=0)

        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return avg_probs, std_probs
        return avg_probs


class WeightedEnsemble(nn.Module):
    def __init__(self, models_list, weight_matrix):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.weights = torch.tensor(weight_matrix, dtype=torch.float32).to(config.DEVICE)

        row_sums = self.weights.sum(dim=0, keepdim=True)
        self.weights = self.weights / (row_sums + 1e-8)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)

        w = self.weights.unsqueeze(1)
        weighted_probs = torch.sum(stacked_probs * w, dim=0)

        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return weighted_probs, std_probs
        return weighted_probs


# =============================================================================
# 2. BUDOWANIE ENSEMBLI
# =============================================================================

def load_best_fold_for_model(model_cfg):
    model_name = model_cfg["name"]
    best_kappa, best_fold, best_metrics = -1, -1, None

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
        raise FileNotFoundError(f"Brak logów JSON dla {model_name}.")

    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{best_fold}_best.pt"
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, best_metrics, best_fold


def build_homogeneous_ensemble(model_cfg):
    model_name = model_cfg["name"]
    print(f"\nSkładanie komitetu Homogeneous dla: {model_name}")
    loaded_models = []
    for fold_idx in range(config.NUM_FOLDS):
        checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{fold_idx + 1}_best.pt"
        model = build_model(model_cfg)
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaded_models.append(model)

    ensemble = SimpleEnsemble(loaded_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


def build_heterogeneous_ensemble():
    print("\nSkładanie Heterogeneous Ensemble (Najlepsze z każdej arch.)")
    loaded_models = []
    for model_cfg in config.MODELS_CONFIG:
        model, _, _ = load_best_fold_for_model(model_cfg)
        loaded_models.append(model)

    ensemble = SimpleEnsemble(loaded_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


def build_weighted_ensemble():
    print("\nSkładanie Weighted Ensemble (Mixture of Experts na bazie F1)")
    loaded_models, weight_matrix = [], []
    for model_cfg in config.MODELS_CONFIG:
        model, best_metrics, _ = load_best_fold_for_model(model_cfg)
        loaded_models.append(model)
        weight_matrix.append(best_metrics["f1_per_class"])

    ensemble = WeightedEnsemble(loaded_models, weight_matrix)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


def build_mega_ensemble():
    print("\nBUDOWANIE MEGA ENSEMBLE (Typ D: 25 modeli)")
    all_25_models = []
    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]
        for fold_idx in range(config.NUM_FOLDS):
            checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{fold_idx + 1}_best.pt"
            if checkpoint_path.exists():
                model = build_model(model_cfg)
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                all_25_models.append(model)

    ensemble = SimpleEnsemble(all_25_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


# =============================================================================
# 3. FUNKCJA EWALUACJI Z OBLICZANIEM NIEPEWNOŚCI
# =============================================================================

@torch.no_grad()
def evaluate_ensemble(ensemble_name, ensemble_model, test_loader):
    all_labels, all_preds = [], []
    all_uncertainty = []

    print(f"Ewaluacja {ensemble_name} na ostatecznym zbiorze testowym...")

    for images, labels in test_loader:
        images = images.to(config.DEVICE)

        # Pobieramy uśrednione prawdopodobieństwa oraz odchylenie modeli
        probs, std_probs = ensemble_model(images, return_std=True)
        _, preds = torch.max(probs, dim=1)

        # Uśredniamy odchylenie po klasach dla każdego pacjenta z batcha
        sample_std = std_probs.mean(dim=1)  # [batch_size]
        all_uncertainty.extend(sample_std.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["model_name"] = ensemble_name

    # Zapisujemy nasz wymiar UQ
    metrics["uq_model_disagreement_std"] = float(np.mean(all_uncertainty)) if len(all_uncertainty) > 0 else 0.0

    with open(config.RESULTS_DIR / f"{ensemble_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# =============================================================================
# 4. TABELA WYNIKÓW
# =============================================================================

def print_uq_summary_table(all_metrics):
    print("\n\n" + "=" * 105)
    print("OSTATECZNE WYNIKI KOMITETÓW + KWANITYFIKACJA NIEPEWNOŚCI (UQ)")
    print("=" * 105)
    print(
        f"{'Model':<28} {'Kappa':>8} {'F1-Mac':>8} | {'UQ-Std':>8} | {'KL0':>6} {'KL1':>6} {'KL2':>6} {'KL3':>6} {'KL4':>6}")
    print("-" * 105)

    for m in sorted(all_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        print(
            f"{m['model_name']:<28} "
            f"{m['cohen_kappa_Quadratic']:>8.4f} "
            f"{m['f1_macro']:>8.4f} | "
            f"{m['uq_model_disagreement_std']:>8.4f} | "
            f"{f1_c[0]:>6.4f} {f1_c[1]:>6.4f} {f1_c[2]:>6.4f} {f1_c[3]:>6.4f} {f1_c[4]:>6.4f}"
        )
    print("=" * 105)


# =============================================================================
# 5. SKRYPT GŁÓWNY
# =============================================================================
def main():
    print("============================================================")
    print("TESTOWANIE KOMITETÓW NA ZBIORZE HOLD-OUT")
    print("============================================================")

    all_samples = load_all_samples()
    _, test_samples = split_holdout(all_samples)
    test_loader = build_test_dataloader(test_samples)

    ensemble_results = []

    # 1. Homogeneous
    for model_cfg in config.MODELS_CONFIG:
        ensemble_name = f"{model_cfg['name']}_Homogeneous"
        hom_ensemble = build_homogeneous_ensemble(model_cfg)
        metrics_hom = evaluate_ensemble(ensemble_name, hom_ensemble, test_loader)
        ensemble_results.append(metrics_hom)
        del hom_ensemble
        torch.cuda.empty_cache()

    # 2. Heterogeneous Avg
    het_ensemble = build_heterogeneous_ensemble()
    metrics_het = evaluate_ensemble("Heterogeneous_Avg", het_ensemble, test_loader)
    ensemble_results.append(metrics_het)
    del het_ensemble
    torch.cuda.empty_cache()

    # 3. Heterogeneous Weighted
    wei_ensemble = build_weighted_ensemble()
    metrics_wei = evaluate_ensemble("Heterogeneous_Weighted", wei_ensemble, test_loader)
    ensemble_results.append(metrics_wei)
    del wei_ensemble
    torch.cuda.empty_cache()

    # 4. Mega Ensemble
    mega_ensemble = build_mega_ensemble()
    metrics_mega = evaluate_ensemble("Mega_Ensemble_TypD", mega_ensemble, test_loader)
    ensemble_results.append(metrics_mega)
    del mega_ensemble
    torch.cuda.empty_cache()

    print_uq_summary_table(ensemble_results)


if __name__ == "__main__":
    main()