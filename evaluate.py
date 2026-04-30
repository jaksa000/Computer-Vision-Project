import json
from pathlib import Path

import torch
import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    brier_score_loss,
)
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader

import config


# =============================================================================
# ZBIERZ PREDYKCJE
# =============================================================================

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(config.DEVICE)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        _, preds = torch.max(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# =============================================================================
# OBLICZ METRYKI KLASYFIKACJI KL
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Metryki klasyfikacji KL (0–4):
      - Balanced Accuracy
      - F1 macro
      - Quadratic Cohen's Kappa
      - F1 per class
    """
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    f1_m         = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa        = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "balanced_accuracy":       round(float(bal_acc), 4),
        "f1_macro":                round(float(f1_m), 4),
        "cohen_kappa_Quadratic":   round(float(kappa), 4),
        "f1_per_class":            [round(float(f), 4) for f in f1_per_class],
    }


# =============================================================================
# OBLICZ METRYKI KALIBRACJI
# =============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Metryki kalibracji probabilistycznej:

    ECE (Expected Calibration Error)
      Średni błąd między confidence modelu a faktyczną accuracy w koszykach.
      Wada: wynik zależy od liczby i podziału koszyków.
      Niższe = lepiej skalibrowany.

    Brier Score (multiclass, uśredniony One-vs-Rest)
      Proper Scoring Rule: MSE(softmax_vector, one_hot_true).
      Nie zależy od koszyków — ocenia każdą próbkę indywidualnie.
      Silnie penalizuje overconfidence.
      Niższe = lepiej skalibrowany. Zakres [0, 2].
    """

    # --- ECE ---
    # Używamy top-1 confidence (standard w literaturze)
    confidences  = np.max(y_probs, axis=1)
    correct      = (np.argmax(y_probs, axis=1) == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    # --- Brier Score (One-vs-Rest, uśredniony po klasach) ---
    brier_per_class = []
    for cls in range(config.NUM_CLASSES):
        y_true_bin = (y_true == cls).astype(int)
        y_prob_cls = y_probs[:, cls]
        brier_per_class.append(brier_score_loss(y_true_bin, y_prob_cls))

    brier_mean = float(np.mean(brier_per_class))

    return {
        "ece":                round(float(ece), 4),
        "brier_score_mean":   round(brier_mean, 4),
        "brier_per_class":    [round(float(b), 4) for b in brier_per_class],
    }


# =============================================================================
# TEST MANN-WHITNEY U (z p-value)
# =============================================================================

def mann_whitney_uncertainty_test(
    uncertainty_certain: np.ndarray,
    uncertainty_uncertain: np.ndarray,
) -> dict:
    """
    Dwustronny test Mann-Whitney U sprawdzający, czy próbki uncertain (wg ekspertów)
    mają istotnie wyższe unc(x) niż próbki certain.

    H0: rozkłady std są identyczne dla obu grup.
    H1: std jest wyższe dla grupy uncertain.

    Raportuje:
      - U-statistic
      - p-value (jednostronny, alternative='less' → testujemy czy certain < uncertain)
      - effect size: rank-biserial correlation r = 1 - 2U / (n1 * n2)
        r ≈ 0.1 małe, 0.3 średnie, 0.5 duże (konwencja Cohen)

    WAŻNE: ten wynik musi pojawić się w R5 pracy jako osobny wynik z liczbami,
    nie tylko jako wzmianka. Format: U=..., p=..., r=...
    """
    n1 = len(uncertainty_certain)
    n2 = len(uncertainty_uncertain)

    # alternative='less': testujemy H1: certain < uncertain (jednostronny)
    stat, p_value = mannwhitneyu(
        uncertainty_certain,
        uncertainty_uncertain,
        alternative="less",
    )

    # rank-biserial correlation jako effect size
    r_effect = 1.0 - (2.0 * float(stat)) / (n1 * n2)

    significant = p_value < 0.05

    result = {
        "test":            "Mann-Whitney U (one-sided: certain < uncertain)",
        "U_statistic":     round(float(stat), 2),
        "p_value":         round(float(p_value), 6),
        "p_significant":   significant,
        "effect_size_r":   round(r_effect, 4),
        "n_certain":       n1,
        "n_uncertain":     n2,
        "interpretation":  (
            f"p={p_value:.4f} {'< 0.05 ✓ różnica istotna statystycznie' if significant else '>= 0.05 brak istotnej różnicy'}, "
            f"r={r_effect:.3f} ({'duży' if abs(r_effect) >= 0.5 else 'średni' if abs(r_effect) >= 0.3 else 'mały'} efekt)"
        ),
    }

    print(f"\n  Test Mann-Whitney U (certain vs uncertain):")
    print(f"    n_certain   = {n1}")
    print(f"    n_uncertain = {n2}")
    print(f"    U           = {stat:.2f}")
    print(f"    p-value     = {p_value:.6f}  {'✓ ISTOTNE (p < 0.05)' if significant else '✗ NIEISTOTNE'}")
    print(f"    effect r    = {r_effect:.4f}  ({'duży' if abs(r_effect) >= 0.5 else 'średni' if abs(r_effect) >= 0.3 else 'mały'} efekt)")

    return result


# =============================================================================
# GŁÓWNA FUNKCJA EWALUACJI — pojedynczy model / fold
# =============================================================================

def evaluate_model(
    model_name: str,
    model,
    test_loader,
    history,
    save_dir: Path = config.RESULTS_DIR,
) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEwaluacja modelu: {model_name}")
    print("-" * 40)

    y_true, y_pred, y_probs = get_predictions(model, test_loader)

    # --- Metryki klasyfikacji ---
    metrics = compute_metrics(y_true, y_pred)

    # --- Metryki kalibracji ---
    calibration = compute_calibration_metrics(y_true, y_probs)
    metrics.update(calibration)

    print(f"  Balanced Accuracy:       {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  F1 (macro):              {metrics['f1_macro']:.4f}")
    print(f"  Quadratic Cohen's Kappa: {metrics['cohen_kappa_Quadratic']:.4f}")
    print(f"  ECE:                     {metrics['ece']:.4f}")
    print(f"  Brier Score (mean):      {metrics['brier_score_mean']:.4f}")

    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_DISPLAY_NAMES,
        zero_division=0,
    )
    print(f"\n  Classification Report:\n{report}")

    metrics["model_name"] = model_name

    # --- Zapis JSON ---
    json_path = save_dir / f"{model_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metryki zapisane: {json_path}")

    # --- Zapis prawdopodobieństw ---
    probs_path = save_dir / f"{model_name}_test_probs.npz"
    np.savez(probs_path, y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    print(f"  Prawdopodobieństwa zapisane: {probs_path}")

    return metrics


# =============================================================================
# PODSUMOWANIE WSZYSTKICH FOLDÓW/MODELI
# =============================================================================

def print_summary_table(all_metrics: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("PODSUMOWANIE POJEDYNCZYCH FOLDÓW")
    print("=" * 120)
    print(
        f"{'Model':<25} {'Kappa':>8} {'F1-Mac':>8} {'ECE':>7} {'Brier':>7} | "
        f"{'KL0':>7} {'KL1':>7} {'KL2':>7} {'KL3':>7} {'KL4':>7}"
    )
    print("-" * 120)

    for m in sorted(all_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        ece   = m.get("ece", float("nan"))
        brier = m.get("brier_score_mean", float("nan"))
        print(
            f"{m['model_name']:<25} "
            f"{m['cohen_kappa_Quadratic']:>8.4f} "
            f"{m['f1_macro']:>8.4f} "
            f"{ece:>7.4f} "
            f"{brier:>7.4f} | "
            f"{f1_c[0]:>7.4f} {f1_c[1]:>7.4f} {f1_c[2]:>7.4f} {f1_c[3]:>7.4f} {f1_c[4]:>7.4f}"
        )
    print("=" * 120)
    print("Posortowane wg Cohen's Kappa")
