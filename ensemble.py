import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)

import config
from models import build_model
from dataset import load_all_samples, load_dual_expert_samples, split_holdout, build_test_dataloader
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
        stacked_probs = torch.stack(all_probs)  # [N_models, batch, num_classes]
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
    print("\nSkładanie Heterogeneous Ensemble (najlepszy fold każdej architektury)")
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
# 3. EWALUACJA ENSEMBLA — z zapisem niepewności per próbka
# =============================================================================

@torch.no_grad()
def evaluate_ensemble(ensemble_name, ensemble_model, test_loader):
    """
    Ewaluuje ensemble na zbiorze testowym.
    Zwraca metryki KL + per-próbkowe odchylenie standardowe (sygnał UQ).

    Zapisuje do pliku .npz:
      - y_true       : prawdziwe etykiety KL
      - y_pred       : predykcje KL
      - uncertainty  : unc(x) = mean_class std_i[p_i(c|x)] dla każdej próbki
    """
    all_labels, all_preds = [], []
    all_uncertainty = []

    print(f"Ewaluacja {ensemble_name} na zbiorze testowym...")

    for images, labels in test_loader:
        images = images.to(config.DEVICE)

        avg_probs, std_probs = ensemble_model(images, return_std=True)
        _, preds = torch.max(avg_probs, dim=1)

        # unc(x) = uśrednione odchylenie po klasach dla każdej próbki
        # std_probs: [batch, num_classes] → mean po klasach → [batch]
        sample_uncertainty = std_probs.mean(dim=1)
        all_uncertainty.extend(sample_uncertainty.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    uncertainty = np.array(all_uncertainty)

    # --- Metryki klasyfikacji KL ---
    metrics = compute_metrics(y_true, y_pred)
    metrics["model_name"] = ensemble_name
    metrics["uq_mean_uncertainty"] = float(np.mean(uncertainty))

    # --- Zapis JSON ---
    with open(config.RESULTS_DIR / f"{ensemble_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Zapis niepewności per próbka ---
    npz_path = config.RESULTS_DIR / f"{ensemble_name}_uncertainty.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, uncertainty=uncertainty)
    print(f"  Niepewności per próbka zapisane: {npz_path}")

    return metrics, uncertainty, y_true, y_pred


# =============================================================================
# 4. WYZNACZANIE PROGU 3σ
# =============================================================================

def compute_uncertainty_threshold(
    uncertainty_scores: np.ndarray,
    sigma_multiplier: float = config.UNCERTAINTY_SIGMA_MULTIPLIER,
) -> tuple[float, float, float]:
    """
    Wyznacza próg detekcji uncertain na podstawie rozkładu wartości unc(x).

    Próg: threshold = mean(unc) + sigma_multiplier * std(unc)

    Interpretacja:
      - Rozkład unc(x) jest (w przybliżeniu) normalny.
      - Reguła 3σ zostawia w prawym ogonie ~0.15% obserwacji.
      - Tylko ekstremalnie wysokie odchylenie jest flagowane jako uncertain.

    Zwraca: (threshold, mean_unc, std_unc)
    """
    mean_unc = float(np.mean(uncertainty_scores))
    std_unc  = float(np.std(uncertainty_scores))
    threshold = mean_unc + sigma_multiplier * std_unc

    print(f"\n  Próg niepewności ({sigma_multiplier}σ):")
    print(f"    mean(unc) = {mean_unc:.4f}")
    print(f"    std(unc)  = {std_unc:.4f}")
    print(f"    threshold = {threshold:.4f}")
    print(f"    Próbki uncertain (unc > threshold): "
          f"{(uncertainty_scores > threshold).sum()} / {len(uncertainty_scores)}")

    return threshold, mean_unc, std_unc


# =============================================================================
# 5. WALIDACJA UQ — porównanie z etykietami ekspertów
# =============================================================================

def evaluate_uncertainty_detection(
    ensemble_name: str,
    uncertainty_scores: np.ndarray,
    expert_agreement_labels: np.ndarray,
    threshold: float,
) -> dict:
    """
    Główna funkcja walidacji UQ — porównuje flagowanie przez ensemble (std > threshold)
    z obiektywną trudnością próbki (niezgodność ekspertów).

    Parametry
    ----------
    ensemble_name          : nazwa ensembla (do logowania)
    uncertainty_scores     : unc(x) dla każdej próbki — ciągły sygnał UQ
    expert_agreement_labels: 0 = certain (eksperci zgodni), 1 = uncertain (niezgodni)
    threshold              : próg wyznaczony przez compute_uncertainty_threshold()

    Zwraca słownik z metrykami detekcji uncertain.
    """
    ensemble_flags = (uncertainty_scores > threshold).astype(int)

    # --- AUROC: ciągły sygnał UQ jako detektor expert-uncertain ---
    try:
        auroc = float(roc_auc_score(expert_agreement_labels, uncertainty_scores))
    except ValueError:
        auroc = float("nan")

    # --- Metryki binarne po progowaniu ---
    f1_uncertain = float(f1_score(
        expert_agreement_labels, ensemble_flags,
        pos_label=config.UNCERTAIN_LABEL, zero_division=0,
    ))
    cm = confusion_matrix(
        expert_agreement_labels, ensemble_flags,
        labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL],
    )

    n_expert_uncertain  = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())
    n_ensemble_flagged  = int(ensemble_flags.sum())
    n_total             = len(uncertainty_scores)

    # --- Mean uncertainty: certain vs uncertain ---
    mask_certain   = expert_agreement_labels == config.CERTAIN_LABEL
    mask_uncertain = expert_agreement_labels == config.UNCERTAIN_LABEL
    mean_unc_certain   = float(np.mean(uncertainty_scores[mask_certain]))   if mask_certain.any()   else float("nan")
    mean_unc_uncertain = float(np.mean(uncertainty_scores[mask_uncertain])) if mask_uncertain.any() else float("nan")

    results = {
        "ensemble_name":          ensemble_name,
        "threshold":              round(threshold, 6),
        "auroc_uncertain_detection": round(auroc, 4),
        "f1_uncertain":           round(f1_uncertain, 4),
        "n_total":                n_total,
        "n_expert_uncertain":     n_expert_uncertain,
        "n_ensemble_flagged":     n_ensemble_flagged,
        "mean_unc_certain":       round(mean_unc_certain, 6),
        "mean_unc_uncertain":     round(mean_unc_uncertain, 6),
        "confusion_matrix":       cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    # --- Wydruk ---
    print(f"\n{'=' * 65}")
    print(f"UQ WALIDACJA: {ensemble_name}")
    print(f"{'=' * 65}")
    print(f"  Próbki łącznie:                  {n_total}")
    print(f"  Expert-uncertain (ground truth): {n_expert_uncertain} ({100*n_expert_uncertain/n_total:.1f}%)")
    print(f"  Ensemble-flagged (>threshold):   {n_ensemble_flagged} ({100*n_ensemble_flagged/n_total:.1f}%)")
    print(f"\n  AUROC (uncertain detection):     {auroc:.4f}")
    print(f"  F1 (uncertain class):            {f1_uncertain:.4f}")
    print(f"\n  Średnie unc(x):")
    print(f"    Certain   (eksperci zgodni):   {mean_unc_certain:.4f}")
    print(f"    Uncertain (eksperci różni):    {mean_unc_uncertain:.4f}")
    print(f"\n  Macierz konfuzji [Certain/Uncertain]:")
    print(f"    Predicted →     Certain  Uncertain")
    print(f"    True Certain:   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"    True Uncertain: {cm[1,0]:6d}   {cm[1,1]:6d}")
    print(f"\n  Raport klasyfikacji:")
    print(classification_report(
        expert_agreement_labels, ensemble_flags,
        target_names=config.AGREEMENT_CLASS_NAMES,
        zero_division=0,
    ))

    # --- Zapis JSON ---
    uq_json_path = config.RESULTS_DIR / f"{ensemble_name}_uq_detection.json"
    with open(uq_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wyniki UQ zapisane: {uq_json_path}")

    return results


# =============================================================================
# 6. TABELE PODSUMOWUJĄCE
# =============================================================================

def print_uq_summary_table(all_kl_metrics: list[dict]) -> None:
    """Tabela wyników klasyfikacji KL dla wszystkich ensembli (jak wcześniej)."""
    print("\n\n" + "=" * 105)
    print("WYNIKI KLASYFIKACJI KL — KOMITETY NA HOLD-OUT")
    print("=" * 105)
    print(f"{'Model':<28} {'Kappa':>8} {'F1-Mac':>8} | {'UQ-Mean':>8} | "
          f"{'KL0':>6} {'KL1':>6} {'KL2':>6} {'KL3':>6} {'KL4':>6}")
    print("-" * 105)

    for m in sorted(all_kl_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        print(
            f"{m['model_name']:<28} "
            f"{m['cohen_kappa_Quadratic']:>8.4f} "
            f"{m['f1_macro']:>8.4f} | "
            f"{m.get('uq_mean_uncertainty', 0.0):>8.4f} | "
            f"{f1_c[0]:>6.4f} {f1_c[1]:>6.4f} {f1_c[2]:>6.4f} {f1_c[3]:>6.4f} {f1_c[4]:>6.4f}"
        )
    print("=" * 105)


def print_uq_detection_table(all_uq_results: list[dict]) -> None:
    """
    Centralna tabela pracy: jak dobrze każdy ensemble wykrywa próbki uncertain
    (walidacja przez etykiety ekspertów).
    """
    print("\n\n" + "=" * 95)
    print("DETEKCJA UNCERTAIN — WALIDACJA PRZEZ ZGODNOŚĆ EKSPERTÓW")
    print("=" * 95)
    print(f"{'Model':<28} {'AUROC':>7} {'F1-Unc':>7} {'Flagged':>8} {'E-Unc':>7} | "
          f"{'μ-unc(C)':>9} {'μ-unc(U)':>9}")
    print(f"{'':28} {'':7} {'':7} {'(pred)':>8} {'(true)':>7} | "
          f"{'certain':>9} {'uncertain':>9}")
    print("-" * 95)

    for r in sorted(all_uq_results, key=lambda x: x["auroc_uncertain_detection"], reverse=True):
        print(
            f"{r['ensemble_name']:<28} "
            f"{r['auroc_uncertain_detection']:>7.4f} "
            f"{r['f1_uncertain']:>7.4f} "
            f"{r['n_ensemble_flagged']:>7d}  "
            f"{r['n_expert_uncertain']:>6d}  | "
            f"{r['mean_unc_certain']:>9.4f} "
            f"{r['mean_unc_uncertain']:>9.4f}"
        )
    print("=" * 95)
    print("AUROC: zdolność separacji certain/uncertain | "
          "μ-unc(C/U): średnie unc(x) w każdej grupie")


# =============================================================================
# 7. SKRYPT GŁÓWNY
# =============================================================================

def main():
    print("=" * 65)
    print("EWALUACJA KOMITETÓW I KWANTYFIKACJA NIEPEWNOŚCI")
    print("=" * 65)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Wczytaj dane ---
    # Używamy load_dual_expert_samples, żeby mieć etykiety certain/uncertain
    all_dual_samples = load_dual_expert_samples()
    _, test_dual_samples = split_holdout(all_dual_samples)

    # DataLoader zwraca tylko (image, kl_label) — agreement_label wyciągamy osobno
    test_loader = build_test_dataloader(test_dual_samples)
    # Etykiety certain/uncertain dla próbek holdout (w tej samej kolejności co loader)
    expert_agreement_labels = np.array([s[2] for s in test_dual_samples])

    print(f"\nHold-out: {len(test_dual_samples)} próbek")
    print(f"  Expert-certain:   {(expert_agreement_labels == config.CERTAIN_LABEL).sum()}")
    print(f"  Expert-uncertain: {(expert_agreement_labels == config.UNCERTAIN_LABEL).sum()}")

    # Zapisz etykiety ekspertów — visualize.py będzie ich potrzebował
    labels_path = config.RESULTS_DIR / "expert_agreement_labels.npy"
    np.save(labels_path, expert_agreement_labels)
    print(f"  Etykiety ekspertów zapisane: {labels_path}")

    kl_metrics_all  = []
    uq_results_all  = []
    all_uncertainties: dict[str, np.ndarray] = {}

    # =========================================================================
    # BUDOWANIE I EWALUACJA KAŻDEGO ENSEMBLA
    # =========================================================================

    # 1. Homogeneous (jeden typ architektury, 5 foldów)
    for model_cfg in config.MODELS_CONFIG:
        name = f"{model_cfg['name']}_Homogeneous"
        ensemble = build_homogeneous_ensemble(model_cfg)
        metrics, uncertainty, _, _ = evaluate_ensemble(name, ensemble, test_loader)
        kl_metrics_all.append(metrics)
        all_uncertainties[name] = uncertainty
        del ensemble
        torch.cuda.empty_cache()

    # 2. Heterogeneous Avg
    het_ensemble = build_heterogeneous_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Heterogeneous_Avg", het_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Heterogeneous_Avg"] = uncertainty
    del het_ensemble
    torch.cuda.empty_cache()

    # 3. Heterogeneous Weighted (MoE)
    wei_ensemble = build_weighted_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Heterogeneous_Weighted", wei_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Heterogeneous_Weighted"] = uncertainty
    del wei_ensemble
    torch.cuda.empty_cache()

    # 4. Mega Ensemble (25 modeli)
    mega_ensemble = build_mega_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Mega_Ensemble_TypD", mega_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Mega_Ensemble_TypD"] = uncertainty
    del mega_ensemble
    torch.cuda.empty_cache()

    # =========================================================================
    # WALIDACJA UQ — próg 3σ i porównanie z etykietami ekspertów
    # =========================================================================
    print("\n\n" + "=" * 65)
    print("ANALIZA UQ — PRÓG 3σ I PORÓWNANIE Z EKSPERTAMI")
    print("=" * 65)

    for ensemble_name, uncertainty_scores in all_uncertainties.items():
        threshold, _, _ = compute_uncertainty_threshold(
            uncertainty_scores,
            sigma_multiplier=config.UNCERTAINTY_SIGMA_MULTIPLIER,
        )
        uq_result = evaluate_uncertainty_detection(
            ensemble_name=ensemble_name,
            uncertainty_scores=uncertainty_scores,
            expert_agreement_labels=expert_agreement_labels,
            threshold=threshold,
        )
        uq_results_all.append(uq_result)

    # =========================================================================
    # TEST MANN-WHITNEY U — centralny dowód statystyczny pracy (R5)
    # =========================================================================
    # Używamy Mega Ensemble jako głównego źródła sygnału UQ (największa liczba modeli)
    # Jeśli brak, używamy pierwszego dostępnego ensembla
    primary_ensemble = "Mega_Ensemble_TypD"
    if primary_ensemble not in all_uncertainties:
        primary_ensemble = next(iter(all_uncertainties))

    unc_scores = all_uncertainties[primary_ensemble]
    mask_certain   = expert_agreement_labels == config.CERTAIN_LABEL
    mask_uncertain = expert_agreement_labels == config.UNCERTAIN_LABEL

    print("\n\n" + "=" * 65)
    print(f"TEST STATYSTYCZNY MANN-WHITNEY U — {primary_ensemble}")
    print("=" * 65)
    print("H0: std ensembla jest identyczne dla grupy certain i uncertain.")
    print("H1: std ensembla jest wyższe dla grupy uncertain (p < 0.05).")

    from evaluate import mann_whitney_uncertainty_test
    mw_result = mann_whitney_uncertainty_test(
        uncertainty_certain   = unc_scores[mask_certain],
        uncertainty_uncertain = unc_scores[mask_uncertain],
    )
    mw_result["ensemble_name"] = primary_ensemble

    mw_path = config.RESULTS_DIR / f"{primary_ensemble}_mann_whitney.json"
    import json as _json
    with open(mw_path, "w") as f:
        _json.dump(mw_result, f, indent=2)
    print(f"  Wyniki Mann-Whitney zapisane: {mw_path}")

    # =========================================================================
    # TEST MANN-WHITNEY U — centralny dowód statystyczny pracy (R5)
    # =========================================================================
    # Używamy Mega Ensemble jako głównego źródła sygnału UQ
    primary_ensemble = "Mega_Ensemble_TypD"
    if primary_ensemble not in all_uncertainties:
        primary_ensemble = next(iter(all_uncertainties))

    unc_scores     = all_uncertainties[primary_ensemble]
    mask_certain   = expert_agreement_labels == config.CERTAIN_LABEL
    mask_uncertain = expert_agreement_labels == config.UNCERTAIN_LABEL

    print("\n\n" + "=" * 65)
    print(f"TEST STATYSTYCZNY MANN-WHITNEY U — {primary_ensemble}")
    print("=" * 65)
    print("H0: std ensembla jest identyczne dla grupy certain i uncertain.")
    print("H1: std ensembla jest wyższe dla grupy uncertain (p < 0.05).")

    from evaluate import mann_whitney_uncertainty_test
    mw_result = mann_whitney_uncertainty_test(
        uncertainty_certain   = unc_scores[mask_certain],
        uncertainty_uncertain = unc_scores[mask_uncertain],
    )
    mw_result["ensemble_name"] = primary_ensemble

    mw_path = config.RESULTS_DIR / f"{primary_ensemble}_mann_whitney.json"
    with open(mw_path, "w") as f:
        json.dump(mw_result, f, indent=2)
    print(f"  Wyniki Mann-Whitney zapisane: {mw_path}")

    # =========================================================================
    # TABELE PODSUMOWUJĄCE
    # =========================================================================
    print_uq_summary_table(kl_metrics_all)
    print_uq_detection_table(uq_results_all)

    print("\nWyniki zapisane w:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
