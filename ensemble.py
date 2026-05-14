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
from dataset import load_all_samples, load_dual_expert_samples, split_holdout, build_test_dataloader
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


class ClassSpecificEnsemble(nn.Module):
    def __init__(self, models_list, expert_indices):
        super().__init__()
        self.expert_indices = expert_indices
        unique_indices = list(set(expert_indices))
        self.models = nn.ModuleList([models_list[i] for i in unique_indices])
        # Changing global index to local one
        self.idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_indices)}

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked_probs = torch.stack(all_probs)

        batch_size = x.size(0)
        num_classes = len(self.expert_indices)
        fused_probs = torch.zeros((batch_size, num_classes), device=x.device)
        for c in range(num_classes):
            orig_idx = self.expert_indices[c]
            new_idx = self.idx_map[orig_idx]
            fused_probs[:, c] = stacked_probs[new_idx, :, c]

        row_sums = fused_probs.sum(dim=1, keepdim=True) + 1e-8
        fused_probs = fused_probs / row_sums
        if return_std:
            std_probs = torch.std(stacked_probs, dim=0, unbiased=False)
            return fused_probs, std_probs
        return fused_probs


# =============================================================================
# Building Ensembles
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
        raise FileNotFoundError(f"no data for this {model_name}.")
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{best_fold}_best.pt"
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, best_metrics, best_fold


def build_homogeneous_ensemble(model_cfg):
    model_name = model_cfg["name"]
    print(f"\n Building Homogeneous Ensemble for : {model_name}")
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
    print("\n Building Heterogeneous Ensemble (Best fold of each architecture)")
    loaded_models = []
    for model_cfg in config.MODELS_CONFIG:
        model, _, _ = load_best_fold_for_model(model_cfg)
        loaded_models.append(model)

    ensemble = SimpleEnsemble(loaded_models)
    ensemble.to(config.DEVICE)
    ensemble.eval()
    return ensemble


def build_weighted_ensemble():
    print("\n Building Weighted Ensemble (Mixture of Experts on F1 basis)")
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
    print("\nBuilding MEGA ENSEMBLE (Type D: 25 models)")
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


def build_class_specific_ensembles():
    print("\nBuilding CLASS-SPECIFIC ENSEMBLES (Version 1 and 2)")
    all_models = []
    f1_matrix = []
    model_names = []

    # Loading all models i and their f1 scores foe each class
    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]
        for fold_idx in range(config.NUM_FOLDS):
            json_path = config.RESULTS_DIR / f"{model_name}_fold{fold_idx + 1}_metrics.json"
            checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_fold{fold_idx + 1}_best.pt"

            if json_path.exists() and checkpoint_path.exists():
                with open(json_path, "r") as f:
                    metrics = json.load(f)
                f1_matrix.append(metrics["f1_per_class"])
                model_names.append(f"{model_name}_f{fold_idx + 1}")

                model = build_model(model_cfg)
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                all_models.append(model)

    f1_matrix = np.array(f1_matrix)

    # Version 1: Best of the best with allowed repetitions
    best_with_rep = np.argmax(f1_matrix, axis=0).tolist()
    arch_to_indices = defaultdict(list)

    for idx, name in enumerate(model_names):
        arch_name = name.rsplit("_f", 1)[0]
        arch_to_indices[arch_name].append(idx)

    arch_list = list(arch_to_indices.keys())
    num_archs = len(arch_list)
    arch_f1_matrix = np.zeros((num_archs, config.NUM_CLASSES))
    arch_best_model_idx = np.zeros((num_archs, config.NUM_CLASSES), dtype=int)

    for arch_idx, arch_name in enumerate(arch_list):
        indices = arch_to_indices[arch_name]
        for c in range(config.NUM_CLASSES):
            best_f1 = -1
            best_idx = -1
            for idx in indices:
                if f1_matrix[idx, c] > best_f1:
                    best_f1 = f1_matrix[idx, c]
                    best_idx = idx

            arch_f1_matrix[arch_idx, c] = best_f1
            arch_best_model_idx[arch_idx, c] = best_idx

    row_ind, col_ind = linear_sum_assignment(-arch_f1_matrix)
    best_without_rep = [0] * config.NUM_CLASSES
    for i in range(len(col_ind)):
        c = col_ind[i]
        arch_idx = row_ind[i]
        best_without_rep[c] = int(arch_best_model_idx[arch_idx, c])
    # =========================================================================

    print("\n  Version 1: Best of the best with allowed repetitions:")
    for c, idx in enumerate(best_with_rep):
        print(f"    KL{c}: {model_names[idx]:<22} (F1: {f1_matrix[idx, c]:.4f})")

    print("\n  Version 2: Diverse ensemble, unique specialists (one per architecture):")
    for c, idx in enumerate(best_without_rep):
        print(f"    KL{c}: {model_names[idx]:<22} (F1: {f1_matrix[idx, c]:.4f})")

    v1_ensemble = ClassSpecificEnsemble(all_models, best_with_rep)
    v1_ensemble.to(config.DEVICE)
    v1_ensemble.eval()

    v2_ensemble = ClassSpecificEnsemble(all_models, best_without_rep)
    v2_ensemble.to(config.DEVICE)
    v2_ensemble.eval()

    return v1_ensemble, v2_ensemble


# =============================================================================
# 3. Ensemble evaluation with uncertainty per sample
# =============================================================================

@torch.no_grad()
def evaluate_ensemble(ensemble_name, ensemble_model, test_loader):
    all_labels, all_preds = [], []
    all_uncertainty = []

    print(f"Evaluation of {ensemble_name}")

    for images, labels in test_loader:
        images = images.to(config.DEVICE)

        avg_probs, std_probs = ensemble_model(images, return_std=True)
        _, preds = torch.max(avg_probs, dim=1)

        sample_uncertainty = std_probs.mean(dim=1)
        all_uncertainty.extend(sample_uncertainty.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    uncertainty = np.array(all_uncertainty)

    metrics = compute_metrics(y_true, y_pred)
    metrics["model_name"] = ensemble_name
    metrics["uq_mean_uncertainty"] = float(np.mean(uncertainty))

    npz_path = config.RESULTS_DIR / f"{ensemble_name}_uncertainty.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, uncertainty=uncertainty)
    print(f"  Uncertainty per sample saved to: {npz_path}")

    return metrics, uncertainty, y_true, y_pred


# =============================================================================
# Threshold designation
# =============================================================================

def compute_uncertainty_threshold(uncertainty_scores, sigma_multiplier=config.UNCERTAINTY_SIGMA_MULTIPLIER, ):
    mean_unc = float(np.mean(uncertainty_scores))
    std_unc = float(np.std(uncertainty_scores))
    threshold = mean_unc + sigma_multiplier * std_unc

    print(f"\n  Uncertainty threshold ({sigma_multiplier}σ):")
    print(f"    mean(unc) = {mean_unc:.4f}")
    print(f"    std(unc)  = {std_unc:.4f}")
    print(f"    threshold = {threshold:.4f}")
    print(f"    Uncertain samples (unc > threshold): "
          f"{(uncertainty_scores > threshold).sum()} / {len(uncertainty_scores)}")

    return threshold, mean_unc, std_unc


# =============================================================================
# UQ validation
# =============================================================================

def evaluate_uncertainty_detection(ensemble_name, uncertainty_scores, expert_agreement_labels, threshold, ):
    ensemble_flags = (uncertainty_scores > threshold).astype(int)

    try:
        auroc = float(roc_auc_score(expert_agreement_labels, uncertainty_scores))
    except ValueError:
        auroc = float("nan")

    f1_uncertain = float(f1_score(
        expert_agreement_labels, ensemble_flags,
        pos_label=config.UNCERTAIN_LABEL, zero_division=0,
    ))
    cm = confusion_matrix(
        expert_agreement_labels, ensemble_flags,
        labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL],
    )

    n_expert_uncertain = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())
    n_ensemble_flagged = int(ensemble_flags.sum())
    n_total = len(uncertainty_scores)

    mask_certain = expert_agreement_labels == config.CERTAIN_LABEL
    mask_uncertain = expert_agreement_labels == config.UNCERTAIN_LABEL
    mean_unc_certain = float(np.mean(uncertainty_scores[mask_certain])) if mask_certain.any() else float("nan")
    mean_unc_uncertain = float(np.mean(uncertainty_scores[mask_uncertain])) if mask_uncertain.any() else float("nan")

    results = {
        "ensemble_name": ensemble_name,
        "threshold": round(threshold, 6),
        "auroc_uncertain_detection": round(auroc, 4),
        "f1_uncertain": round(f1_uncertain, 4),
        "n_total": n_total,
        "n_expert_uncertain": n_expert_uncertain,
        "n_ensemble_flagged": n_ensemble_flagged,
        "mean_unc_certain": round(mean_unc_certain, 6),
        "mean_unc_uncertain": round(mean_unc_uncertain, 6),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n{'=' * 65}")
    print(f"UQ VALIDATION: {ensemble_name}")
    print(f"{'=' * 65}")
    print(f"  Total Samples:                  {n_total}")
    print(f"  Expert-uncertain (ground truth): {n_expert_uncertain} ({100 * n_expert_uncertain / n_total:.1f}%)")
    print(f"  Ensemble-flagged (>threshold):   {n_ensemble_flagged} ({100 * n_ensemble_flagged / n_total:.1f}%)")
    print(f"\n  AUROC (uncertain detection):     {auroc:.4f}")
    print(f"  F1 (uncertain class):            {f1_uncertain:.4f}")
    print(f"\n  average unc(x):")
    print(f"    Certain   (experts agree):   {mean_unc_certain:.4f}")
    print(f"    Uncertain (experts disagree):    {mean_unc_uncertain:.4f}")
    print(f"\n  Confusion Matrix [Certain/Uncertain]:")
    print(f"    Predicted →     Certain  Uncertain")
    print(f"    True Certain:   {cm[0, 0]:6d}   {cm[0, 1]:6d}")
    print(f"    True Uncertain: {cm[1, 0]:6d}   {cm[1, 1]:6d}")
    print(f"\n  Classification report:")
    print(classification_report(
        expert_agreement_labels,
        ensemble_flags,
        labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL],
        target_names=config.AGREEMENT_CLASS_NAMES,
        zero_division=0,
    ))

    return results


# =============================================================================
# EXCEL EXPORT (Pandas)
# =============================================================================

def save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_result):
    print("\n" + "=" * 65)
    print(" SAVING RESULTS TO EXCEL (PANDAS)")
    print("=" * 65)

    # 1. KL Classification
    df_kl = pd.DataFrame(kl_metrics_all)
    if "f1_per_class" in df_kl.columns:
        f1_splits = pd.DataFrame(df_kl['f1_per_class'].tolist(),
                                 columns=['F1_KL0', 'F1_KL1', 'F1_KL2', 'F1_KL3', 'F1_KL4'])
        df_kl = pd.concat([df_kl.drop('f1_per_class', axis=1), f1_splits], axis=1)

    # 2. UQ Detection
    df_uq = pd.DataFrame(uq_results_all)
    if "confusion_matrix" in df_uq.columns:
        df_uq['TN (Certain_ok)'] = df_uq['confusion_matrix'].apply(
            lambda x: x[0][0] if isinstance(x, list) and len(x) > 0 else 0)
        df_uq['FP (False_Uncertain)'] = df_uq['confusion_matrix'].apply(
            lambda x: x[0][1] if isinstance(x, list) and len(x) > 0 else 0)
        df_uq['FN (Missed_Uncertain)'] = df_uq['confusion_matrix'].apply(
            lambda x: x[1][0] if isinstance(x, list) and len(x) > 1 else 0)
        df_uq['TP (True_Uncertain)'] = df_uq['confusion_matrix'].apply(
            lambda x: x[1][1] if isinstance(x, list) and len(x) > 1 else 0)
        df_uq = df_uq.drop('confusion_matrix', axis=1)

    # 3. Mann-Whitney
    df_mw = pd.DataFrame([mw_result])

    excel_path = config.RESULTS_DIR / "MASTER_RESULTS_SUMMARY.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_kl.to_excel(writer, sheet_name="KL_Classification", index=False)
        df_uq.to_excel(writer, sheet_name="UQ_Detection", index=False)
        df_mw.to_excel(writer, sheet_name="Mann_Whitney_Test", index=False)

    print(f" All tables compiled successfully to:")
    print(f"  {excel_path}")


# =============================================================================
# SUMMARY TABLES (Terminal)
# =============================================================================

def print_uq_summary_table(all_kl_metrics: list[dict]) -> None:
    print("\n\n" + "=" * 105)
    print(" KL Classification Results — Ensembles tested on HOLD-OUT")
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
    print("\n\n" + "=" * 95)
    print(" UNCERTAIN Detection — Validation by experts agreement")
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
    print("AUROC: Separation ability certain/uncertain | "
          "μ-unc(C/U): Mean unc(x) in each group")


# =============================================================================
# Main Function
# =============================================================================

def main():
    print("=" * 65)
    print("Ensembles Evaluation And Uncertainty Quantification")
    print("=" * 65)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_dual_samples = load_dual_expert_samples()
    _, test_dual_samples = split_holdout(all_dual_samples)

    test_loader = build_test_dataloader(test_dual_samples)
    expert_agreement_labels = np.array([s[2] for s in test_dual_samples])

    print(f"\nHold-out: {len(test_dual_samples)} samples")
    print(f"  Expert-certain:   {(expert_agreement_labels == config.CERTAIN_LABEL).sum()}")
    print(f"  Expert-uncertain: {(expert_agreement_labels == config.UNCERTAIN_LABEL).sum()}")

    labels_path = config.RESULTS_DIR / "expert_agreement_labels.npy"
    np.save(labels_path, expert_agreement_labels)
    print(f" Experts labels saved to: {labels_path}")

    kl_metrics_all = []
    uq_results_all = []
    all_uncertainties: dict[str, np.ndarray] = {}

    # =========================================================================
    # Building and evaluating each ensemble
    # =========================================================================

    for model_cfg in config.MODELS_CONFIG:
        name = f"{model_cfg['name']}_Homogeneous"
        ensemble = build_homogeneous_ensemble(model_cfg)
        metrics, uncertainty, _, _ = evaluate_ensemble(name, ensemble, test_loader)
        kl_metrics_all.append(metrics)
        all_uncertainties[name] = uncertainty
        del ensemble
        torch.cuda.empty_cache()

    het_ensemble = build_heterogeneous_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Heterogeneous_Avg", het_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Heterogeneous_Avg"] = uncertainty
    del het_ensemble
    torch.cuda.empty_cache()

    wei_ensemble = build_weighted_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Heterogeneous_Weighted", wei_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Heterogeneous_Weighted"] = uncertainty
    del wei_ensemble
    torch.cuda.empty_cache()

    mega_ensemble = build_mega_ensemble()
    metrics, uncertainty, _, _ = evaluate_ensemble("Mega_Ensemble_TypD", mega_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Mega_Ensemble_TypD"] = uncertainty
    del mega_ensemble
    torch.cuda.empty_cache()


    v1_ensemble, v2_ensemble = build_class_specific_ensembles()

    metrics, uncertainty, _, _ = evaluate_ensemble("Class_Specific_With_Rep", v1_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Class_Specific_With_Rep"] = uncertainty
    del v1_ensemble
    torch.cuda.empty_cache()

    metrics, uncertainty, _, _ = evaluate_ensemble("Class_Specific_Unique", v2_ensemble, test_loader)
    kl_metrics_all.append(metrics)
    all_uncertainties["Class_Specific_Unique"] = uncertainty
    del v2_ensemble
    torch.cuda.empty_cache()

    # =========================================================================
    # UQ Validation
    # =========================================================================
    print("\n\n" + "=" * 65)
    print("UQ Analysis — 3σ Threshold and Comparison with Experts")
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
    #  MANN-WHITNEY U TEST
    # =========================================================================
    primary_ensemble = "Mega_Ensemble_TypD"
    if primary_ensemble not in all_uncertainties:
        primary_ensemble = next(iter(all_uncertainties))

    unc_scores = all_uncertainties[primary_ensemble]
    mask_certain = expert_agreement_labels == config.CERTAIN_LABEL
    mask_uncertain = expert_agreement_labels == config.UNCERTAIN_LABEL

    print("\n\n" + "=" * 65)
    print(f" MANN-WHITNEY U Statistical test — {primary_ensemble}")
    print("=" * 65)
    print("H0: Ensemble std is identical for certain and uncertain.")
    print("H1: Ensemble std is higher for uncertain group (p < 0.05).")

    from evaluate import mann_whitney_uncertainty_test
    mw_result = mann_whitney_uncertainty_test(
        uncertainty_certain=unc_scores[mask_certain],
        uncertainty_uncertain=unc_scores[mask_uncertain],
    )
    mw_result["ensemble_name"] = primary_ensemble

    # =========================================================================
    # Summary EXCEL Export & Terminal Output
    # =========================================================================
    save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_result)

    print_uq_summary_table(kl_metrics_all)
    print_uq_detection_table(uq_results_all)


if __name__ == "__main__":
    main()