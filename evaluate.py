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
        probs = torch.softmax(logits, dim=1)
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

    # Ordinal metrics — appropriate because KL grades form an ordinal scale.
    # MAE penalises large ordinal errors proportionally; off-by-one accuracy
    # counts predictions within one grade as clinically acceptable.
    abs_errors   = np.abs(y_true.astype(int) - y_pred.astype(int))
    mae          = float(np.mean(abs_errors))
    off_by_one   = float(np.mean(abs_errors <= 1))

    return {
        "balanced_accuracy":    round(float(bal_acc), 4),
        "f1_macro":             round(float(f1_macro), 4),
        "cohen_kappa_Quadratic": round(float(kappa), 4),
        "f1_per_class":         [round(float(f), 4) for f in f1_per_class],
        "mae_ordinal":          round(mae, 4),
        "off_by_one_accuracy":  round(off_by_one, 4),
    }


def compute_calibration_metrics(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_bins: int = 10,
) -> dict:
    confidences = np.max(y_probs, axis=1)
    correct = (np.argmax(y_probs, axis=1) == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    brier_per_class = []
    for cls in range(config.NUM_CLASSES):
        y_true_bin = (y_true == cls).astype(int)
        y_prob_cls = y_probs[:, cls]
        brier_per_class.append(brier_score_loss(y_true_bin, y_prob_cls))

    brier_mean = float(np.mean(brier_per_class))

    return {
        "ece":             round(float(ece), 4),
        "brier_score_mean": round(brier_mean, 4),
        "brier_per_class": [round(float(b), 4) for b in brier_per_class],
    }


# =============================================================================
# Mann-Whitney U — single signal (kept for backward compatibility)
# =============================================================================

def mann_whitney_uncertainty_test(uncertainty_certain, uncertainty_uncertain):
    n1 = len(uncertainty_certain)
    n2 = len(uncertainty_uncertain)

    stat, p_value = mannwhitneyu(
        uncertainty_certain,
        uncertainty_uncertain,
        alternative="less",
    )

    r_effect  = 1.0 - (2.0 * float(stat)) / (n1 * n2)
    significant = bool(p_value < 0.05)

    effect_label = (
        "large"  if abs(r_effect) >= 0.5 else
        "medium" if abs(r_effect) >= 0.3 else
        "small"
    )

    result = {
        "test":             "Mann-Whitney U (one-sided: certain < uncertain)",
        "U_statistic":      round(float(stat), 2),
        "p_value":          round(float(p_value), 6),
        "p_significant":    significant,
        "effect_size_r":    round(r_effect, 4),
        "n_certain":        n1,
        "n_uncertain":      n2,
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
# Mann-Whitney U — all three uncertainty signals with Bonferroni correction
# =============================================================================

def mann_whitney_all_signals(unc_scores_dict: dict, expert_labels: np.ndarray,
                              alpha: float = 0.05) -> list:
    """
    Run Mann-Whitney U (one-sided: certain < uncertain) for every uncertainty
    signal and apply Bonferroni correction over the family of three tests.

    Parameters
    ----------
    unc_scores_dict : {"unc_mean": array, "unc_max": array, "entropy": array}
    expert_labels   : array of 0 (certain) / 1 (uncertain)
    alpha           : family-wise error rate (default 0.05)

    Returns
    -------
    list of result dicts, one per signal, with raw and corrected p-values.

    Note on test choice
    -------------------
    The alternative hypothesis is one-sided (certain < uncertain): we expect
    the model to assign lower uncertainty to cases where both experts agree.
    The rank-biserial correlation r = 1 − 2U/(n₁·n₂) is used as effect size.
    Bonferroni is conservative but appropriate given the small number of tests.
    """
    signals = ["unc_mean", "unc_max", "entropy"]
    n_tests = len(signals)
    mask_c  = expert_labels == config.CERTAIN_LABEL
    mask_u  = expert_labels == config.UNCERTAIN_LABEL
    n1, n2  = int(mask_c.sum()), int(mask_u.sum())

    # Collect raw results first, then apply correction
    raw = []
    for sig in signals:
        scores = unc_scores_dict[sig]
        stat, p_raw = mannwhitneyu(scores[mask_c], scores[mask_u], alternative="less")
        r_effect = 1.0 - (2.0 * float(stat)) / (n1 * n2)
        raw.append((sig, float(stat), float(p_raw), r_effect))

    results = []
    for sig, stat, p_raw, r_effect in raw:
        p_corrected  = min(p_raw * n_tests, 1.0)
        significant  = p_corrected < alpha
        effect_label = (
            "large"  if abs(r_effect) >= 0.5 else
            "medium" if abs(r_effect) >= 0.3 else
            "small"
        )
        results.append({
            "signal":                    sig,
            "test":                      "Mann-Whitney U (one-sided: certain < uncertain)",
            "n_certain":                 n1,
            "n_uncertain":               n2,
            "U_statistic":               round(stat, 2),
            "p_value_raw":               round(p_raw, 6),
            "p_value_bonferroni":        round(p_corrected, 6),
            "n_tests_bonferroni":        n_tests,
            "alpha":                     alpha,
            "p_significant_bonferroni":  significant,
            "effect_size_r":             round(r_effect, 4),
            "effect_label":              effect_label,
            "interpretation": (
                f"p_bonf={p_corrected:.4f} "
                f"{'< {:.2f} (significant)'.format(alpha) if significant else '>= {:.2f} (not significant)'.format(alpha)}, "
                f"r={r_effect:.3f} ({effect_label} effect)"
            ),
        })

    print(f"\n  Mann-Whitney U — Bonferroni-corrected (n_tests={n_tests}, α={alpha})")
    print(f"  n_certain={n1},  n_uncertain={n2}")
    header = f"  {'Signal':<12} {'U':>10} {'p_raw':>10} {'p_bonf':>10} {'Sig':>5} {'r':>8}  Effect"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in results:
        print(f"  {r['signal']:<12} "
              f"{r['U_statistic']:>10.2f} "
              f"{r['p_value_raw']:>10.6f} "
              f"{r['p_value_bonferroni']:>10.6f} "
              f"{'*' if r['p_significant_bonferroni'] else '':>5} "
              f"{r['effect_size_r']:>8.4f}  {r['effect_label']}")

    return results


# =============================================================================
# Main evaluation function (used by main.py for single-model CV evaluation)
# =============================================================================

def evaluate_model(model_name, model, val_loader, history, save_dir=None):
    if save_dir is None:
        save_dir = config.INDIVIDUAL_MODELS_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel evaluation: {model_name}")
    print("-" * 40)

    y_true, y_pred, y_probs = get_predictions(model, val_loader)

    metrics    = compute_metrics(y_true, y_pred)
    calibration = compute_calibration_metrics(y_true, y_probs)
    metrics.update(calibration)

    print(f"  Balanced Accuracy:       {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  F1 (macro):              {metrics['f1_macro']:.4f}")
    print(f"  Quadratic Cohen's Kappa: {metrics['cohen_kappa_Quadratic']:.4f}")
    print(f"  MAE (ordinal):           {metrics['mae_ordinal']:.4f}")
    print(f"  Off-by-one accuracy:     {metrics['off_by_one_accuracy'] * 100:.2f}%")
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


def print_summary_table(all_metrics):
    print("\n" + "=" * 135)
    print("Single Fold Summary:")
    print("=" * 135)
    print(
        f"{'Model':<25} {'Kappa':>8} {'F1-Mac':>8} {'MAE':>7} {'Off-1':>7} "
        f"{'ECE':>7} {'Brier':>7} | "
        f"{'KL0':>7} {'KL1':>7} {'KL2':>7} {'KL3':>7} {'KL4':>7}"
    )
    print("-" * 135)

    for m in sorted(all_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c  = m["f1_per_class"]
        ece   = m.get("ece", float("nan"))
        brier = m.get("brier_score_mean", float("nan"))
        mae   = m.get("mae_ordinal", float("nan"))
        off1  = m.get("off_by_one_accuracy", float("nan"))
        print(
            f"{m['model_name']:<25} "
            f"{m['cohen_kappa_Quadratic']:>8.4f} "
            f"{m['f1_macro']:>8.4f} "
            f"{mae:>7.4f} "
            f"{off1:>7.4f} "
            f"{ece:>7.4f} "
            f"{brier:>7.4f} | "
            f"{f1_c[0]:>7.4f} {f1_c[1]:>7.4f} {f1_c[2]:>7.4f} "
            f"{f1_c[3]:>7.4f} {f1_c[4]:>7.4f}"
        )
    print("=" * 135)
    print("Sorted by Cohen's Kappa")
