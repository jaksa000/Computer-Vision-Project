import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from collections import defaultdict
import config
from models import build_model
from dataset import load_dual_expert_samples, split_holdout, build_test_dataloader
from evaluate import compute_metrics


# =============================================================================
# Ensemble classes
# =============================================================================

class SimpleEnsemble(nn.Module):
    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)
        avg_probs = torch.mean(stacked_probs, dim=0)
        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return avg_probs, std_probs
        return avg_probs


class WeightedEnsemble(nn.Module):
    def __init__(self, models_list, weight_matrix):
        super().__init__()
        self.models  = nn.ModuleList(models_list)
        self.weights = torch.tensor(weight_matrix, dtype=torch.float32).to(config.DEVICE)
        row_sums     = self.weights.sum(dim=0, keepdim=True)
        self.weights = self.weights / (row_sums + 1e-8)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs     = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)
        w             = self.weights.unsqueeze(1)
        weighted_probs = torch.sum(stacked_probs * w, dim=0)
        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return weighted_probs, std_probs
        return weighted_probs


class ClassSpecificEnsemble(nn.Module):
    """
    Mixture-of-Experts: each KL class gets a specialist model for prediction.
    Uncertainty (std) is always computed from ALL 25 models so it is directly
    comparable with the Mega Ensemble's UQ signal.
    """

    def __init__(self, all_models_list, expert_indices):
        super().__init__()
        self.expert_indices = expert_indices
        self.models = nn.ModuleList(all_models_list)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs     = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)   # [N_models, batch, 5]

        batch_size  = x.size(0)
        num_classes = len(self.expert_indices)
        fused_probs = torch.zeros((batch_size, num_classes), device=x.device)
        for c, model_idx in enumerate(self.expert_indices):
            fused_probs[:, c] = stacked_probs[model_idx, :, c]
        fused_probs = fused_probs / (fused_probs.sum(dim=1, keepdim=True) + 1e-8)

        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return fused_probs, std_probs
        return fused_probs


# =============================================================================
# Building ensembles
# =============================================================================

def load_best_fold_for_model(model_cfg):
    model_name = model_cfg["name"]
    best_kappa, best_fold, best_metrics = -1, -1, None
    for fold_idx in range(config.NUM_FOLDS):
        json_path = config.RESULTS_DIR / f"{model_name}_fold{fold_idx + 1}_metrics.json"
        if json_path.exists():
            with open(json_path) as f:
                metrics = json.load(f)
            if metrics["cohen_kappa_Quadratic"] > best_kappa:
                best_kappa   = metrics["cohen_kappa_Quadratic"]
                best_fold    = fold_idx + 1
                best_metrics = metrics
    if best_fold == -1:
        raise FileNotFoundError(f"No checkpoint data found for {model_name}.")
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{best_fold}_best.pt"
    model = build_model(model_cfg)
    ckpt  = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, best_metrics, best_fold


def _load_checkpoint(model_cfg, fold_idx):
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_cfg['name']}_fold{fold_idx + 1}_best.pt"
    model = build_model(model_cfg)
    ckpt  = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_homogeneous_ensemble(model_cfg):
    print(f"\n Building Homogeneous Ensemble: {model_cfg['name']}")
    models   = [_load_checkpoint(model_cfg, i) for i in range(config.NUM_FOLDS)]
    ensemble = SimpleEnsemble(models)
    return ensemble.to(config.DEVICE).eval()


def build_heterogeneous_ensemble():
    print("\n Building Heterogeneous Ensemble (best fold per architecture)")
    models   = [load_best_fold_for_model(cfg)[0] for cfg in config.MODELS_CONFIG]
    ensemble = SimpleEnsemble(models)
    return ensemble.to(config.DEVICE).eval()


def build_weighted_ensemble():
    print("\n Building Weighted Ensemble (MoE, F1-based weights)")
    models, weights = [], []
    for cfg in config.MODELS_CONFIG:
        model, best_metrics, _ = load_best_fold_for_model(cfg)
        models.append(model)
        weights.append(best_metrics["f1_per_class"])
    ensemble = WeightedEnsemble(models, weights)
    return ensemble.to(config.DEVICE).eval()


def build_mega_ensemble():
    print("\n Building MEGA ENSEMBLE (Type D — 25 models)")
    all_models = []
    for cfg in config.MODELS_CONFIG:
        for fold_idx in range(config.NUM_FOLDS):
            ckpt_path = config.CHECKPOINTS_DIR / f"{cfg['name']}_fold{fold_idx + 1}_best.pt"
            if ckpt_path.exists():
                all_models.append(_load_checkpoint(cfg, fold_idx))
    ensemble = SimpleEnsemble(all_models)
    return ensemble.to(config.DEVICE).eval()


def build_class_specific_ensembles():
    print("\n Building CLASS-SPECIFIC ENSEMBLES (Type E and F)")
    all_models, f1_matrix, model_names = [], [], []

    for cfg in config.MODELS_CONFIG:
        for fold_idx in range(config.NUM_FOLDS):
            json_path = config.RESULTS_DIR / f"{cfg['name']}_fold{fold_idx + 1}_metrics.json"
            ckpt_path = config.CHECKPOINTS_DIR / f"{cfg['name']}_fold{fold_idx + 1}_best.pt"
            if json_path.exists() and ckpt_path.exists():
                with open(json_path) as f:
                    metrics = json.load(f)
                f1_matrix.append(metrics["f1_per_class"])
                model_names.append(f"{cfg['name']}_f{fold_idx + 1}")
                all_models.append(_load_checkpoint(cfg, fold_idx))

    f1_matrix = np.array(f1_matrix)   # [25, 5]

    # Type E: best model per class, repetitions allowed
    best_with_rep = np.argmax(f1_matrix, axis=0).tolist()

    # Type F: unique architecture per class (Hungarian algorithm)
    arch_to_indices = defaultdict(list)
    for idx, name in enumerate(model_names):
        arch_to_indices[name.rsplit("_f", 1)[0]].append(idx)

    arch_list           = list(arch_to_indices.keys())
    arch_f1_matrix      = np.zeros((len(arch_list), config.NUM_CLASSES))
    arch_best_model_idx = np.zeros((len(arch_list), config.NUM_CLASSES), dtype=int)

    for ai, arch_name in enumerate(arch_list):
        for c in range(config.NUM_CLASSES):
            scores = [(f1_matrix[idx, c], idx) for idx in arch_to_indices[arch_name]]
            best_f1, best_idx = max(scores)
            arch_f1_matrix[ai, c]      = best_f1
            arch_best_model_idx[ai, c] = best_idx

    row_ind, col_ind = linear_sum_assignment(-arch_f1_matrix)
    best_without_rep = [0] * config.NUM_CLASSES
    for i in range(len(col_ind)):
        best_without_rep[col_ind[i]] = int(arch_best_model_idx[row_ind[i], col_ind[i]])

    print("\n  Type E — Best specialist per class (repetitions allowed):")
    for c, idx in enumerate(best_with_rep):
        print(f"    KL{c}: {model_names[idx]:<22} (F1: {f1_matrix[idx, c]:.4f})")
    print("\n  Type F — Unique architecture per class (Hungarian):")
    for c, idx in enumerate(best_without_rep):
        print(f"    KL{c}: {model_names[idx]:<22} (F1: {f1_matrix[idx, c]:.4f})")

    v1 = ClassSpecificEnsemble(all_models, best_with_rep)
    v2 = ClassSpecificEnsemble(all_models, best_without_rep)
    return v1.to(config.DEVICE).eval(), v2.to(config.DEVICE).eval()


# =============================================================================
# Core inference helper
# =============================================================================

@torch.no_grad()
def _run_forward(ensemble_model, loader):
    all_labels, all_preds, all_unc = [], [], []
    for images, labels in loader:
        images = images.to(config.DEVICE)
        avg_probs, std_probs = ensemble_model(images, return_std=True)
        _, preds = torch.max(avg_probs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_unc.extend(std_probs.mean(dim=1).cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_unc)


# =============================================================================
# Threshold computation — calibrated on CV set (training-side, ~1402 samples)
#
# This is the methodologically correct approach: the decision boundary is set
# using data that was never part of the final holdout evaluation.  Using the
# holdout itself to set the threshold would constitute information leakage
# into the test-set reporting.
# =============================================================================

def compute_uncertainty_threshold_from_cv(ensemble_model, cv_loader,
                                          sigma_multiplier=config.UNCERTAINTY_SIGMA_MULTIPLIER):
    print(f"\n  Computing uncertainty threshold from CV set ({sigma_multiplier}σ)...")
    _, _, cv_unc = _run_forward(ensemble_model, cv_loader)

    mean_unc  = float(np.mean(cv_unc))
    std_unc   = float(np.std(cv_unc))
    threshold = mean_unc + sigma_multiplier * std_unc

    print(f"    CV samples used: {len(cv_unc)}")
    print(f"    mean(unc):       {mean_unc:.6f}")
    print(f"    std(unc):        {std_unc:.6f}")
    print(f"    threshold:       {threshold:.6f}  (mean + {sigma_multiplier}σ)")
    return threshold, mean_unc, std_unc


# =============================================================================
# Ensemble evaluation
#
#   KL classification metrics  →  holdout_loader  (clean, never seen in training)
#   Threshold calibration      →  cv_loader       (training-side, ~1402 samples)
#   UQ / uncertainty analysis  →  full_loader     (all 1650 images)
# =============================================================================

def evaluate_ensemble(ensemble_name, ensemble_model,
                      holdout_loader, full_loader, cv_loader):
    print(f"\nEvaluating: {ensemble_name}")

    # KL classification on clean holdout
    y_true_ho, y_pred_ho, _ = _run_forward(ensemble_model, holdout_loader)
    kl_metrics = compute_metrics(y_true_ho, y_pred_ho)
    kl_metrics["model_name"] = ensemble_name
    print(f"  [Holdout] Kappa: {kl_metrics['cohen_kappa_Quadratic']:.4f}  "
          f"F1: {kl_metrics['f1_macro']:.4f}  "
          f"Bal.Acc: {kl_metrics['balanced_accuracy']:.4f}")

    # Threshold from CV set
    threshold, mean_cv, std_cv = compute_uncertainty_threshold_from_cv(
        ensemble_model, cv_loader)
    kl_metrics.update({
        "threshold_cv": round(threshold, 6),
        "mean_unc_cv":  round(mean_cv, 6),
        "std_unc_cv":   round(std_cv, 6),
    })

    # Uncertainty on full dataset
    y_true_full, y_pred_full, uncertainty = _run_forward(ensemble_model, full_loader)
    kl_metrics["uq_mean_uncertainty"] = float(np.mean(uncertainty))

    npz_path = config.RESULTS_DIR / f"{ensemble_name}_uncertainty.npz"
    np.savez(npz_path, y_true=y_true_full, y_pred=y_pred_full, uncertainty=uncertainty)
    print(f"  [Full dataset] Uncertainty saved: {npz_path}")

    # Persist threshold so visualize.py can read it
    threshold_path = config.RESULTS_DIR / f"{ensemble_name}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "ensemble_name":    ensemble_name,
            "threshold":        round(threshold, 6),
            "mean_unc_cv":      round(mean_cv, 6),
            "std_unc_cv":       round(std_cv, 6),
            "sigma_multiplier": config.UNCERTAINTY_SIGMA_MULTIPLIER,
            "note": "Threshold calibrated on CV portion (~1402 samples). Holdout not used.",
        }, f, indent=2)

    return kl_metrics, uncertainty, y_true_full, y_pred_full, threshold


# =============================================================================
# UQ validation against expert agreement labels
# =============================================================================

def evaluate_uncertainty_detection(ensemble_name, uncertainty_scores,
                                   expert_agreement_labels, threshold):
    flags = (uncertainty_scores > threshold).astype(int)

    try:
        auroc = float(roc_auc_score(expert_agreement_labels, uncertainty_scores))
    except ValueError:
        auroc = float("nan")

    f1_unc = float(f1_score(expert_agreement_labels, flags,
                             pos_label=config.UNCERTAIN_LABEL, zero_division=0))
    cm     = confusion_matrix(expert_agreement_labels, flags,
                              labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL])

    n_total     = len(uncertainty_scores)
    n_exp_unc   = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())
    n_flagged   = int(flags.sum())

    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL
    mu_c   = float(np.mean(uncertainty_scores[mask_c])) if mask_c.any() else float("nan")
    mu_u   = float(np.mean(uncertainty_scores[mask_u])) if mask_u.any() else float("nan")

    results = {
        "ensemble_name":             ensemble_name,
        "threshold":                 round(threshold, 6),
        "threshold_source":          "CV set (training-side, ~1402 samples)",
        "auroc_uncertain_detection": round(auroc, 4),
        "f1_uncertain":              round(f1_unc, 4),
        "n_total":                   n_total,
        "n_expert_uncertain":        n_exp_unc,
        "n_ensemble_flagged":        n_flagged,
        "mean_unc_certain":          round(mu_c, 6),
        "mean_unc_uncertain":        round(mu_u, 6),
        "confusion_matrix":          cm.tolist(),
    }

    print(f"\n{'=' * 65}")
    print(f"UQ VALIDATION: {ensemble_name}")
    print(f"{'=' * 65}")
    print(f"  Threshold (from CV set):         {threshold:.6f}")
    print(f"  Total samples (full dataset):    {n_total}")
    print(f"  Expert-uncertain (ground truth): {n_exp_unc}  ({100*n_exp_unc/n_total:.1f}%)")
    print(f"  Ensemble-flagged (>threshold):   {n_flagged}  ({100*n_flagged/n_total:.1f}%)")
    print(f"  AUROC:                           {auroc:.4f}")
    print(f"  F1 (uncertain):                  {f1_unc:.4f}")
    print(f"  Mean unc(x) — certain:           {mu_c:.4f}")
    print(f"  Mean unc(x) — uncertain:         {mu_u:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted →     Certain  Uncertain")
    print(f"    True Certain:   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"    True Uncertain: {cm[1,0]:6d}   {cm[1,1]:6d}")
    print(classification_report(expert_agreement_labels, flags,
                                 labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL],
                                 target_names=config.AGREEMENT_CLASS_NAMES,
                                 zero_division=0))

    with open(config.RESULTS_DIR / f"{ensemble_name}_uq_detection.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Excel export
# =============================================================================

def save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_result):
    df_kl = pd.DataFrame(kl_metrics_all)
    if "f1_per_class" in df_kl.columns:
        splits = pd.DataFrame(df_kl["f1_per_class"].tolist(),
                              columns=["F1_KL0","F1_KL1","F1_KL2","F1_KL3","F1_KL4"])
        df_kl  = pd.concat([df_kl.drop("f1_per_class", axis=1), splits], axis=1)

    df_uq = pd.DataFrame(uq_results_all)
    if "confusion_matrix" in df_uq.columns:
        df_uq["TN"]  = df_uq["confusion_matrix"].apply(lambda x: x[0][0])
        df_uq["FP"]  = df_uq["confusion_matrix"].apply(lambda x: x[0][1])
        df_uq["FN"]  = df_uq["confusion_matrix"].apply(lambda x: x[1][0])
        df_uq["TP"]  = df_uq["confusion_matrix"].apply(lambda x: x[1][1])
        df_uq        = df_uq.drop("confusion_matrix", axis=1)

    excel_path = config.RESULTS_DIR / "MASTER_RESULTS_SUMMARY.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_kl.to_excel(writer, sheet_name="KL_Classification", index=False)
        df_uq.to_excel(writer, sheet_name="UQ_Detection",      index=False)
        pd.DataFrame([mw_result]).to_excel(writer, sheet_name="Mann_Whitney_Test", index=False)
    print(f"  Results saved: {excel_path}")


# =============================================================================
# Summary tables
# =============================================================================

def print_uq_summary_table(all_kl_metrics):
    print("\n" + "=" * 100)
    print(" KL Classification — Ensembles on HOLD-OUT")
    print("=" * 100)
    print(f"{'Model':<28} {'Kappa':>8} {'F1-Mac':>8} {'Threshold':>10} | "
          f"{'KL0':>6} {'KL1':>6} {'KL2':>6} {'KL3':>6} {'KL4':>6}")
    print("-" * 100)
    for m in sorted(all_kl_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        print(f"{m['model_name']:<28} {m['cohen_kappa_Quadratic']:>8.4f} "
              f"{m['f1_macro']:>8.4f} {m.get('threshold_cv', float('nan')):>10.6f} | "
              f"{f1_c[0]:>6.4f} {f1_c[1]:>6.4f} {f1_c[2]:>6.4f} {f1_c[3]:>6.4f} {f1_c[4]:>6.4f}")
    print("=" * 100)


def print_uq_detection_table(all_uq_results):
    print("\n" + "=" * 100)
    print(" UQ Detection — Full dataset, threshold from CV set")
    print("=" * 100)
    print(f"{'Model':<28} {'AUROC':>7} {'F1-Unc':>7} {'Flagged':>8} {'E-Unc':>7} | "
          f"{'μ-unc(C)':>9} {'μ-unc(U)':>9}")
    print("-" * 100)
    for r in sorted(all_uq_results, key=lambda x: x["auroc_uncertain_detection"], reverse=True):
        print(f"{r['ensemble_name']:<28} {r['auroc_uncertain_detection']:>7.4f} "
              f"{r['f1_uncertain']:>7.4f} {r['n_ensemble_flagged']:>7d}  "
              f"{r['n_expert_uncertain']:>6d}  | "
              f"{r['mean_unc_certain']:>9.4f} {r['mean_unc_uncertain']:>9.4f}")
    print("=" * 100)
    print("NOTE: KL metrics on holdout only. UQ on full dataset. Threshold from CV set.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 65)
    print("Ensemble Evaluation and Uncertainty Quantification")
    print("=" * 65)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_dual_samples = load_dual_expert_samples()
    cv_samples, test_dual_samples = split_holdout(all_dual_samples)

    # Three loaders — each with a distinct, non-overlapping purpose
    cv_loader      = build_test_dataloader(cv_samples)           # threshold calibration
    holdout_loader = build_test_dataloader(test_dual_samples)    # KL metrics (clean)
    full_loader    = build_test_dataloader(all_dual_samples)     # UQ analysis

    expert_agreement_labels = np.array([s[2] for s in all_dual_samples])
    n_certain   = int((expert_agreement_labels == config.CERTAIN_LABEL).sum())
    n_uncertain = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())
    print(f"\n  CV set (threshold):   {len(cv_samples)} samples")
    print(f"  Holdout (KL metrics): {len(test_dual_samples)} samples")
    print(f"  Full dataset (UQ):    {len(all_dual_samples)} samples")
    print(f"    Expert-certain:     {n_certain}")
    print(f"    Expert-uncertain:   {n_uncertain}")

    np.save(config.RESULTS_DIR / "expert_agreement_labels.npy", expert_agreement_labels)

    kl_metrics_all:   list       = []
    uq_results_all:   list       = []
    all_uncertainties: dict[str, np.ndarray] = {}
    all_thresholds:    dict[str, float]       = {}

    def _run(name, ensemble):
        metrics, unc, _, _, thr = evaluate_ensemble(
            name, ensemble, holdout_loader, full_loader, cv_loader)
        kl_metrics_all.append(metrics)
        all_uncertainties[name] = unc
        all_thresholds[name]    = thr
        del ensemble
        torch.cuda.empty_cache()

    for cfg in config.MODELS_CONFIG:
        _run(f"{cfg['name']}_Homogeneous", build_homogeneous_ensemble(cfg))

    _run("Heterogeneous_Avg",      build_heterogeneous_ensemble())
    _run("Heterogeneous_Weighted", build_weighted_ensemble())
    _run("Mega_Ensemble_TypD",     build_mega_ensemble())

    v1, v2 = build_class_specific_ensembles()
    _run("Class_Specific_With_Rep", v1)
    _run("Class_Specific_Unique",   v2)

    # UQ validation
    print("\n" + "=" * 65)
    print("UQ Analysis — CV-calibrated threshold vs Expert Agreement Labels")
    print("=" * 65)
    for name, unc in all_uncertainties.items():
        uq_results_all.append(evaluate_uncertainty_detection(
            name, unc, expert_agreement_labels, all_thresholds[name]))

    # Mann-Whitney U test
    primary = "Mega_Ensemble_TypD"
    if primary not in all_uncertainties:
        primary = next(iter(all_uncertainties))

    unc_scores   = all_uncertainties[primary]
    mask_c       = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u       = expert_agreement_labels == config.UNCERTAIN_LABEL
    print(f"\n  Mann-Whitney U test — {primary}")
    print(f"    n_certain={mask_c.sum()}, n_uncertain={mask_u.sum()}")

    from evaluate import mann_whitney_uncertainty_test
    mw_result = mann_whitney_uncertainty_test(unc_scores[mask_c], unc_scores[mask_u])
    mw_result["ensemble_name"] = primary
    with open(config.RESULTS_DIR / f"{primary}_mann_whitney.json", "w") as f:
        json.dump(mw_result, f, indent=2)

    save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_result)
    print_uq_summary_table(kl_metrics_all)
    print_uq_detection_table(uq_results_all)


if __name__ == "__main__":
    main()
