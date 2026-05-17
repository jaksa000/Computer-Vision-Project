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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    f1_macro     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa        = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "balanced_accuracy":       round(float(bal_acc), 4),
        "f1_macro":                round(float(f1_macro), 4),
        "cohen_kappa_Quadratic":   round(float(kappa), 4),
        "f1_per_class":            [round(float(f), 4) for f in f1_per_class],
    }


# =============================================================================
# FUNCTION — CALIBRATION METRICS
# =============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    ECE (Expected Calibration Error):
        Average gap between model confidence and actual accuracy across bins.
        Lower is better. Depends on bin count.

    Brier Score (multiclass, One-vs-Rest averaged):
        MSE(softmax_vector, one_hot_true). Proper Scoring Rule.
        Lower is better. Range [0, 2].
    """

    # ECE
    confidences = np.max(y_probs, axis=1)
    correct     = (np.argmax(y_probs, axis=1) == y_true).astype(float)

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

    # Brier Score (One-vs-Rest, averaged across classes)
    brier_per_class = []
    for cls in range(config.NUM_CLASSES):
        y_true_bin = (y_true == cls).astype(int)
        y_prob_cls = y_probs[:, cls]
        brier_per_class.append(brier_score_loss(y_true_bin, y_prob_cls))

    brier_mean = float(np.mean(brier_per_class))

    return {
        "ece":               round(float(ece), 4),
        "brier_score_mean":  round(brier_mean, 4),
        "brier_per_class":   [round(float(b), 4) for b in brier_per_class],
    }


# =============================================================================
# FUNCTION — MANN-WHITNEY U TEST
# =============================================================================

def mann_whitney_uncertainty_test(uncertainty_certain, uncertainty_uncertain):
    """
    One-sided Mann-Whitney U test: do expert-uncertain samples have
    significantly higher unc(x) than expert-certain samples?

    H0: distributions identical for both groups.
    H1: unc(x) stochastically higher for the uncertain group.

    Reports U-statistic, p-value, and rank-biserial correlation r as effect size.
    r ≈ 0.1 small, 0.3 medium, 0.5 large (Cohen convention).
    """
    n1 = len(uncertainty_certain)
    n2 = len(uncertainty_uncertain)

    # alternative='less': H1 is that certain < uncertain (one-sided)
    stat, p_value = mannwhitneyu(
        uncertainty_certain,
        uncertainty_uncertain,
        alternative="less",
    )

    r_effect   = 1.0 - (2.0 * float(stat)) / (n1 * n2)
    significant = p_value < 0.05

    effect_label = (
        "large" if abs(r_effect) >= 0.5 else
        "medium" if abs(r_effect) >= 0.3 else
        "small"
    )

    result = {
        "test":           "Mann-Whitney U (one-sided: certain < uncertain)",
        "U_statistic":    round(float(stat), 2),
        "p_value":        round(float(p_value), 6),
        "p_significant":  significant,
        "effect_size_r":  round(r_effect, 4),
        "n_certain":      n1,
        "n_uncertain":    n2,
        "interpretation": (
            f"p={p_value:.4f} "
            f"{'< 0.05 — statistically significant difference' if significant else '>= 0.05 — statistically insignificant difference'}, "
            f"r={r_effect:.3f} ({effect_label} effect)"
        ),
    }

    print(f"\n  Mann-Whitney U test (certain vs uncertain):")
    print(f"    n_certain   = {n1}")
    print(f"    n_uncertain = {n2}")
    print(f"    U           = {stat:.2f}")
    print(f"    p-value     = {p_value:.6f}  "
          f"{'Significant (p < 0.05)' if significant else 'Not significant (p >= 0.05)'}")
    print(f"    effect r    = {r_effect:.4f}  ({effect_label} effect)")

    return result


# =============================================================================
# Main evaluation function (used by main.py for single-model CV evaluation)
# =============================================================================

def evaluate_model(model_name, model, test_loader, history, save_dir=config.RESULTS_DIR):
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel evaluation: {model_name}")
    print("-" * 40)

    y_true, y_pred, y_probs = get_predictions(model, test_loader)

    metrics     = compute_metrics(y_true, y_pred)
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

    json_path = save_dir / f"{model_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {json_path}")

    probs_path = save_dir / f"{model_name}_test_probs.npz"
    np.savez(probs_path, y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    print(f"  Probabilities saved to: {probs_path}")

    return metrics


# =============================================================================
# Cross-validation summary
# =============================================================================

def print_summary_table(all_metrics):
    print("\n" + "=" * 120)
    print("Single Fold Summary:")
    print("=" * 120)
    print(
        f"{'Model':<25} {'Kappa':>8} {'F1-Mac':>8} {'ECE':>7} {'Brier':>7} | "
        f"{'KL0':>7} {'KL1':>7} {'KL2':>7} {'KL3':>7} {'KL4':>7}"
    )
    print("-" * 120)

    for m in sorted(all_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c  = m["f1_per_class"]
        ece   = m.get("ece",              float("nan"))
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
    print("Sorted by Cohen's Kappa")
