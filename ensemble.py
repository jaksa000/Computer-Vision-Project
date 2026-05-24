import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)
import config
from models import build_model
from dataset import load_dual_expert_samples, split_holdout, build_test_dataloader
from evaluate import compute_metrics, compute_calibration_metrics


# =============================================================================
# ENSEMBLE ARCHITECTURES
# =============================================================================

class SimpleEnsemble(nn.Module):
    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(m(x), dim=1) for m in self.models]
        stacked = torch.stack(all_probs)
        avg_probs = torch.mean(stacked, dim=0)
        if return_std:
            return avg_probs, torch.std(stacked, dim=0, unbiased=False)
        return avg_probs


class WeightedEnsemble(nn.Module):
    def __init__(self, models_list, weight_matrix):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        w = torch.tensor(weight_matrix, dtype=torch.float32).to(config.DEVICE)
        self.weights = w / (w.sum(dim=0, keepdim=True) + 1e-8)

    @torch.no_grad()
    def forward(self, x, return_std=False):
        all_probs = [torch.softmax(m(x), dim=1) for m in self.models]
        stacked = torch.stack(all_probs)
        weighted = torch.sum(stacked * self.weights.unsqueeze(1), dim=0)
        if return_std:
            return weighted, torch.std(stacked, dim=0, unbiased=False)
        return weighted


# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_checkpoint(model_cfg, fold_idx):
    model = build_model(model_cfg)
    ckpt = torch.load(
        config.CHECKPOINTS_DIR / f"{model_cfg['name']}_fold{fold_idx + 1}_best.pt",
        map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_best_fold(model_cfg):
    best_kappa, best_fold, best_metrics = -1, -1, None
    for fold_idx in range(config.NUM_FOLDS):
        jp = config.INDIVIDUAL_MODELS_DIR / f"{model_cfg['name']}_fold{fold_idx + 1}_metrics.json"
        if jp.exists():
            with open(jp) as f:
                m = json.load(f)
            if m["cohen_kappa_Quadratic"] > best_kappa:
                best_kappa, best_fold, best_metrics = (
                    m["cohen_kappa_Quadratic"], fold_idx + 1, m)
    if best_fold == -1:
        raise FileNotFoundError(f"No checkpoint found for {model_cfg['name']}.")
    return _load_checkpoint(model_cfg, best_fold - 1), best_metrics, best_fold


def build_homogeneous_ensemble(model_cfg):
    print(f"\n Building Homogeneous Ensemble: {model_cfg['name']}")
    return SimpleEnsemble(
        [_load_checkpoint(model_cfg, i) for i in range(config.NUM_FOLDS)]
    ).to(config.DEVICE).eval()


def build_heterogeneous_ensemble():
    print("\n Building Heterogeneous Ensemble (best fold per architecture)")
    return SimpleEnsemble(
        [_load_best_fold(cfg)[0] for cfg in config.MODELS_CONFIG]
    ).to(config.DEVICE).eval()


def build_weighted_ensemble():
    print("\n Building Weighted Ensemble (per-class F1 weights)")
    models, weights = [], []
    for cfg in config.MODELS_CONFIG:
        model, best_metrics, _ = _load_best_fold(cfg)
        models.append(model)
        weights.append(best_metrics["f1_per_class"])
    return WeightedEnsemble(models, weights).to(config.DEVICE).eval()


def build_mega_ensemble():
    print("\n Building Mega Ensemble (Type D — 25 models)")
    all_models = []
    for cfg in config.MODELS_CONFIG:
        for fold_idx in range(config.NUM_FOLDS):
            cp = config.CHECKPOINTS_DIR / f"{cfg['name']}_fold{fold_idx + 1}_best.pt"
            if cp.exists():
                all_models.append(_load_checkpoint(cfg, fold_idx))
    print(f"  Loaded {len(all_models)} models.")
    return SimpleEnsemble(all_models).to(config.DEVICE).eval()


# =============================================================================
# FORWARD PASS
# =============================================================================

@torch.no_grad()
def _run_forward(ensemble_model, loader):
    all_labels, all_preds, all_probs = [], [], []
    all_unc_mean, all_unc_max, all_entropy = [], [], []

    for images, labels in loader:
        images = images.to(config.DEVICE)
        avg_probs, std_probs = ensemble_model(images, return_std=True)
        _, preds = torch.max(avg_probs, dim=1)

        unc_mean = std_probs.mean(dim=1)
        unc_max = std_probs.max(dim=1)[0]
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(avg_probs.cpu().numpy())
        all_unc_mean.extend(unc_mean.cpu().numpy())
        all_unc_max.extend(unc_max.cpu().numpy())
        all_entropy.extend(entropy.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_unc_mean),
        np.array(all_unc_max),
        np.array(all_entropy),
    )


# =============================================================================
# THRESHOLD COMPUTATION
# Refactored: CV forward pass is separated from threshold computation so that
# cv_scores can be reused later for the sensitivity sweep without a second
# forward pass through the whole CV set.
# =============================================================================

def _get_cv_uncertainty_scores(ensemble_name, ensemble_model, cv_loader):
    """
    Run one forward pass over the CV set and return raw uncertainty arrays.
    These scores are used both for default threshold computation and for the
    full percentile sweep — so the model only needs to run once.
    """
    print(f"  Computing CV uncertainty scores for {ensemble_name}...")
    _, _, _, unc_mean, unc_max, entropy = _run_forward(ensemble_model, cv_loader)
    print(f"    CV samples: {len(unc_mean)}")
    return {"unc_mean": unc_mean, "unc_max": unc_max, "entropy": entropy}


def compute_thresholds_from_cv(ensemble_name, cv_scores):
    """
    Compute the default (config.UNCERTAINTY_PERCENTILE) thresholds from
    pre-computed CV uncertainty scores.
    """
    pct = config.UNCERTAINTY_PERCENTILE
    thresholds = {
        "unc_mean":      float(np.percentile(cv_scores["unc_mean"], pct)),
        "unc_max":       float(np.percentile(cv_scores["unc_max"],  pct)),
        "entropy":       float(np.percentile(cv_scores["entropy"],  pct)),
        "percentile_used": pct,
        "cv_samples":    int(len(cv_scores["unc_mean"])),
    }
    print(
        f"  [{ensemble_name}] Thresholds ({pct}th pct) -> "
        f"unc_mean: {thresholds['unc_mean']:.4f},  "
        f"unc_max: {thresholds['unc_max']:.4f},  "
        f"entropy: {thresholds['entropy']:.4f}"
    )
    return thresholds


# =============================================================================
# THRESHOLD SENSITIVITY ANALYSIS
# Sweeps percentiles 90.0–99.9 (step 0.1) plus μ+3σ for every uncertainty
# signal.  Thresholds are always calibrated on CV scores and applied to the
# full-dataset eval scores — consistent with the default 95th-percentile logic.
# =============================================================================

def _evaluate_threshold_at(ensemble_name, signal, method, percentile, threshold,
                            eval_scores, expert_labels):
    """
    Evaluate one (signal, threshold) point and return a metrics dict.
    All rate-based metrics (precision, recall, F1) are computed for the
    positive class = UNCERTAIN_LABEL, matching the primary task definition.
    """
    flags    = (eval_scores > threshold).astype(int)
    n_total  = len(flags)
    n_flagged = int(flags.sum())
    pct_flagged = round(100.0 * n_flagged / n_total, 2) if n_total > 0 else 0.0

    tp = int(((flags == 1) & (expert_labels == config.UNCERTAIN_LABEL)).sum())
    fp = int(((flags == 1) & (expert_labels == config.CERTAIN_LABEL)).sum())
    tn = int(((flags == 0) & (expert_labels == config.CERTAIN_LABEL)).sum())
    fn = int(((flags == 0) & (expert_labels == config.UNCERTAIN_LABEL)).sum())

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(
        2 * precision * recall / (precision + recall), 4
    ) if (precision + recall) > 0 else 0.0

    return {
        "ensemble_name":        ensemble_name,
        "signal":               signal,
        "method":               method,
        "percentile":           percentile,          # NaN for sigma3
        "threshold":            round(float(threshold), 6),
        "n_flagged":            n_flagged,
        "pct_flagged":          pct_flagged,
        "TP":                   tp,
        "FP":                   fp,
        "TN":                   tn,
        "FN":                   fn,
        "precision_uncertain":  precision,
        "recall_uncertain":     recall,
        "f1_uncertain":         f1,
    }


def run_threshold_sensitivity_analysis(ensemble_name, cv_scores, eval_scores, expert_labels):
    """
    For every uncertainty signal × every threshold method, evaluate UQ detection.

    Threshold methods
    -----------------
    percentile : 90.0, 90.1, ..., 99.9  (100 values, calibrated on CV scores)
    sigma3     : μ_cv + 3·σ_cv           (one value per signal)

    Returns a long-format DataFrame — one row per (signal, method, percentile).
    """
    print(f"\n  Threshold sensitivity sweep: {ensemble_name}")
    signals = ["unc_mean", "unc_max", "entropy"]
    rows = []

    for signal in signals:
        cv_sig   = cv_scores[signal]
        eval_sig = eval_scores[signal]
        mu, sigma = float(np.mean(cv_sig)), float(np.std(cv_sig))

        # --- Percentile sweep 90.0–99.9 ---
        for pct in np.round(np.arange(90.0, 100.0, 0.1), 1):
            threshold = float(np.percentile(cv_sig, pct))
            rows.append(_evaluate_threshold_at(
                ensemble_name, signal, "percentile", float(pct),
                threshold, eval_sig, expert_labels,
            ))

        # --- μ + 3σ ---
        thr_sigma = mu + config.UNCERTAINTY_SIGMA_MULTIPLIER * sigma
        rows.append(_evaluate_threshold_at(
            ensemble_name, signal, "sigma3", np.nan,
            thr_sigma, eval_sig, expert_labels,
        ))

    print(f"    {len(rows)} threshold combinations evaluated.")
    return pd.DataFrame(rows)


def select_best_thresholds(sensitivity_df):
    """
    From the full sensitivity table, extract three reference rows per
    (ensemble_name, signal):
      - default_{pct}pct  : the fixed operational threshold
      - sigma3            : μ + 3σ
      - best_f1_percentile: percentile that maximises f1_uncertain on the full dataset

    All three are needed to give an honest comparison; no single 'winner'
    is declared.  The caller can choose which to highlight.

    Returns a summary DataFrame with a 'selection_method' column.
    """
    pct_only = sensitivity_df[sensitivity_df["method"] == "percentile"]

    best_rows = (
        pct_only
        .loc[pct_only.groupby(["ensemble_name", "signal"])["f1_uncertain"].idxmax()]
        .copy()
    )
    best_rows["selection_method"] = "best_f1_percentile"

    sigma_rows = sensitivity_df[sensitivity_df["method"] == "sigma3"].copy()
    sigma_rows["selection_method"] = "sigma3"

    default_pct = float(config.UNCERTAINTY_PERCENTILE)
    default_rows = sensitivity_df[
        (sensitivity_df["method"] == "percentile") &
        (sensitivity_df["percentile"].round(1) == default_pct)
    ].copy()
    default_rows["selection_method"] = f"default_{config.UNCERTAINTY_PERCENTILE}pct"

    return pd.concat([default_rows, sigma_rows, best_rows], ignore_index=True)


# =============================================================================
# ENSEMBLE EVALUATION  (unchanged from original)
# =============================================================================

def evaluate_ensemble(ensemble_name, ensemble_model, holdout_loader, full_loader):
    print(f"\nEvaluating: {ensemble_name}")

    y_true_ho, y_pred_ho, y_probs_ho, _, _, _ = _run_forward(ensemble_model, holdout_loader)

    kl_metrics = compute_metrics(y_true_ho, y_pred_ho)
    kl_metrics.update(compute_calibration_metrics(y_true_ho, y_probs_ho))
    kl_metrics["model_name"] = ensemble_name

    print(f"  [Holdout]  Kappa: {kl_metrics['cohen_kappa_Quadratic']:.4f}  "
          f"F1: {kl_metrics['f1_macro']:.4f}  "
          f"ECE: {kl_metrics['ece']:.4f}  "
          f"Brier: {kl_metrics['brier_score_mean']:.4f}")

    y_true_full, y_pred_full, _, unc_mean, unc_max, entropy = _run_forward(
        ensemble_model, full_loader)

    kl_metrics["mean_unc_mean"] = round(float(np.mean(unc_mean)), 6)
    kl_metrics["mean_unc_max"]  = round(float(np.mean(unc_max)),  6)
    kl_metrics["mean_entropy"]  = round(float(np.mean(entropy)),  6)

    npz_path = config.ENSEMBLES_DIR / f"{ensemble_name}_uncertainty.npz"
    np.savez(npz_path,
             y_true=y_true_full, y_pred=y_pred_full,
             unc_mean=unc_mean, unc_max=unc_max, entropy=entropy)
    print(f"  [Full dataset] Saved: {npz_path}")

    return kl_metrics, unc_mean, unc_max, entropy, y_true_full


def _eval_one_signal(signal_name, unc_scores, expert_labels, threshold):
    flags = (unc_scores > threshold).astype(int)
    try:
        auroc = float(roc_auc_score(expert_labels, unc_scores))
        auprc = float(average_precision_score(expert_labels, unc_scores))
    except ValueError:
        auroc = auprc = float("nan")

    f1_unc = float(f1_score(expert_labels, flags,
                            pos_label=config.UNCERTAIN_LABEL, zero_division=0))
    cm = confusion_matrix(expert_labels, flags,
                          labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL])

    mask_c = expert_labels == config.CERTAIN_LABEL
    mask_u = expert_labels == config.UNCERTAIN_LABEL
    mu_c = float(np.mean(unc_scores[mask_c])) if mask_c.any() else float("nan")
    mu_u = float(np.mean(unc_scores[mask_u])) if mask_u.any() else float("nan")

    print(f"    [{signal_name:<9}]  "
          f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
          f"F1={f1_unc:.4f}  "
          f"flagged={flags.sum()}/{len(flags)}  "
          f"μ_certain={mu_c:.4f}  μ_uncertain={mu_u:.4f}")

    return {
        f"auroc_{signal_name}":              round(auroc,  4),
        f"auprc_{signal_name}":              round(auprc,  4),
        f"f1_uncertain_{signal_name}":       round(f1_unc, 4),
        f"n_flagged_{signal_name}":          int(flags.sum()),
        f"mean_unc_certain_{signal_name}":   round(mu_c, 6),
        f"mean_unc_uncertain_{signal_name}": round(mu_u, 6),
        f"cm_{signal_name}":                 cm.tolist(),
    }


def evaluate_uncertainty_detection(ensemble_name, unc_mean, unc_max, entropy,
                                   expert_agreement_labels, kl_labels_full, thresholds):
    n_total   = len(unc_mean)
    n_exp_unc = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())

    print(f"\n{'=' * 65}")
    print(f"UQ VALIDATION: {ensemble_name}")
    print(f"{'=' * 65}")
    print(f"  Total samples (full dataset):     {n_total}")
    print(f"  Expert-uncertain (ground truth):  {n_exp_unc}  ({100 * n_exp_unc / n_total:.1f}%)")

    results = {
        "ensemble_name":      ensemble_name,
        "n_total":            n_total,
        "n_expert_uncertain": n_exp_unc,
        "threshold_unc_mean": round(thresholds["unc_mean"], 6),
        "threshold_unc_max":  round(thresholds["unc_max"],  6),
        "threshold_entropy":  round(thresholds["entropy"],  6),
    }

    for sig_name, scores in [("unc_mean", unc_mean), ("unc_max", unc_max), ("entropy", entropy)]:
        results.update(_eval_one_signal(sig_name, scores,
                                        expert_agreement_labels, thresholds[sig_name]))

    # Best signal by AUROC
    best_signal = max(["unc_mean", "unc_max", "entropy"],
                      key=lambda s: results.get(f"auroc_{s}", 0.0))
    best_flags = (
        {"unc_mean": unc_mean, "unc_max": unc_max, "entropy": entropy}[best_signal]
        > thresholds[best_signal]
    ).astype(int)
    results["best_signal_by_auroc"] = best_signal

    # KL breakdown for flagged
    flagged_kls       = kl_labels_full[best_flags == 1]
    unique, counts    = np.unique(flagged_kls, return_counts=True)
    kl_counts         = dict(zip(unique, counts))
    kl_str = " : ".join([str(kl_counts.get(k, 0)) for k in range(config.NUM_CLASSES)])

    print(f"\n  Best signal by AUROC: {best_signal}")
    print(f"  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = {kl_str}")
    print(classification_report(expert_agreement_labels, best_flags,
                                labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL],
                                target_names=config.AGREEMENT_CLASS_NAMES,
                                zero_division=0))

    with open(config.ENSEMBLES_DIR / f"{ensemble_name}_uq_detection.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, best_flags


# =============================================================================
# CONSENSUS MATRIX BUILDING
# =============================================================================

def _build_consensus_df(all_dual_samples, flags_dict, kl_labels, expert_labels):
    """
    Build a consensus DataFrame from a mapping {ensemble_name: flag_array}.
    Rows are sorted by Total_Flags descending so the most-flagged samples
    appear first.
    """
    df_data = {
        "Image_Name":       [s[0].name for s in all_dual_samples],
        "KL_Label":         [config.CLASS_DISPLAY_NAMES[kl] for kl in kl_labels],
        "Expert_Uncertain": expert_labels,
    }
    for name, flags in flags_dict.items():
        df_data[name] = flags

    consensus_df = pd.DataFrame(df_data)
    ens_cols = list(flags_dict.keys())
    consensus_df["Total_Flags"] = consensus_df[ens_cols].sum(axis=1)
    consensus_df = consensus_df.sort_values(
        by=["Total_Flags", "Expert_Uncertain"], ascending=[False, False]
    )
    return consensus_df


def build_consensus_from_sensitivity(all_dual_samples, all_eval_unc,
                                     sensitivity_df, kl_labels, expert_labels,
                                     selection_method):
    """
    Build a consensus DataFrame where each ensemble's flags are derived from
    the threshold selected by `selection_method`.

    selection_method options
    ------------------------
    "sigma3"             : μ_cv + 3σ_cv
    "best_f1_percentile" : percentile that maximises F1 on the full dataset
    f"default_{pct}pct"  : fixed operational percentile (e.g. "default_95pct")

    For each ensemble the signal with the highest f1_uncertain among the
    selection_method rows is used.  This mirrors how best_flags is chosen
    for the default evaluation (best signal by AUROC), but is based on F1
    to remain consistent with the sensitivity analysis objective.
    """
    best_df    = select_best_thresholds(sensitivity_df)
    method_df  = best_df[best_df["selection_method"] == selection_method]

    flags_dict = {}
    for ens_name, eval_unc in all_eval_unc.items():
        ens_rows = method_df[method_df["ensemble_name"] == ens_name]
        if len(ens_rows) == 0:
            continue
        best_row  = ens_rows.loc[ens_rows["f1_uncertain"].idxmax()]
        flags     = (eval_unc[best_row["signal"]] > best_row["threshold"]).astype(int)
        flags_dict[ens_name] = flags

    return _build_consensus_df(all_dual_samples, flags_dict, kl_labels, expert_labels)


# =============================================================================
# EXCEL EXPORT  (extended with sensitivity sheets)
# =============================================================================

def save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_results,
                                  consensus_default, sensitivity_all, sensitivity_best,
                                  consensus_sigma3, consensus_best):
    df_kl = pd.DataFrame(kl_metrics_all)

    if "f1_per_class" in df_kl.columns:
        splits = pd.DataFrame(df_kl["f1_per_class"].tolist(),
                              columns=["F1_KL0", "F1_KL1", "F1_KL2", "F1_KL3", "F1_KL4"])
        df_kl  = pd.concat([df_kl.drop("f1_per_class", axis=1), splits], axis=1)
    if "brier_per_class" in df_kl.columns:
        bp    = pd.DataFrame(df_kl["brier_per_class"].tolist(),
                             columns=["Brier_KL0", "Brier_KL1", "Brier_KL2", "Brier_KL3", "Brier_KL4"])
        df_kl = pd.concat([df_kl.drop("brier_per_class", axis=1), bp], axis=1)

    df_uq = pd.DataFrame(uq_results_all)
    for sig in ["unc_mean", "unc_max", "entropy"]:
        col = f"cm_{sig}"
        if col in df_uq.columns:
            df_uq[f"TN_{sig}"] = df_uq[col].apply(lambda x: x[0][0] if x else 0)
            df_uq[f"FP_{sig}"] = df_uq[col].apply(lambda x: x[0][1] if x else 0)
            df_uq[f"FN_{sig}"] = df_uq[col].apply(lambda x: x[1][0] if x else 0)
            df_uq[f"TP_{sig}"] = df_uq[col].apply(lambda x: x[1][1] if x else 0)
            df_uq = df_uq.drop(col, axis=1)

    excel_path = config.ENSEMBLES_DIR / "MASTER_RESULTS_SUMMARY.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_kl.to_excel(writer, sheet_name="KL_Classification",   index=False)
        df_uq.to_excel(writer, sheet_name="UQ_Detection",        index=False)
        pd.DataFrame(mw_results).to_excel(writer, sheet_name="Mann_Whitney_Test", index=False)
        consensus_default.to_excel(writer, sheet_name="Consensus_95pct",  index=False)
        sensitivity_all.to_excel(writer,   sheet_name="Threshold_Sensitivity", index=False)
        sensitivity_best.to_excel(writer,  sheet_name="Best_Thresholds",  index=False)
        consensus_sigma3.to_excel(writer,  sheet_name="Consensus_3Sigma", index=False)
        consensus_best.to_excel(writer,    sheet_name="Consensus_Best",   index=False)

    print(f"\n  Master results saved: {excel_path}")


# =============================================================================
# PRINT TABLES  (unchanged)
# =============================================================================

def print_kl_summary_table(all_kl_metrics):
    print("\n" + "=" * 110)
    print(" KL Classification — all ensembles on HOLD-OUT (clean)")
    print("=" * 110)
    print(f"{'Model':<28} {'Kappa':>8} {'F1-Mac':>8} {'ECE':>7} {'Brier':>7} | "
          f"{'KL0':>6} {'KL1':>6} {'KL2':>6} {'KL3':>6} {'KL4':>6}")
    print("-" * 110)
    for m in sorted(all_kl_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        print(f"{m['model_name']:<28} "
              f"{m['cohen_kappa_Quadratic']:>8.4f} "
              f"{m['f1_macro']:>8.4f} "
              f"{m.get('ece', float('nan')):>7.4f} "
              f"{m.get('brier_score_mean', float('nan')):>7.4f} | "
              f"{f1_c[0]:>6.4f} {f1_c[1]:>6.4f} "
              f"{f1_c[2]:>6.4f} {f1_c[3]:>6.4f} {f1_c[4]:>6.4f}")
    print("=" * 110)


def print_uq_detection_table(all_uq_results):
    print("\n" + "=" * 120)
    print(" UQ Detection — full dataset (Independent CV thresholds)")
    print("=" * 120)
    print(f"{'Model':<28} "
          f"{'AUROC_mn':>9} {'AUPRC_mn':>9} "
          f"{'AUROC_mx':>9} {'AUPRC_mx':>9} "
          f"{'AUROC_ent':>10} {'AUPRC_ent':>10} | "
          f"{'E-Unc':>6} {'Best':>9}")
    print("-" * 120)
    for r in sorted(all_uq_results,
                    key=lambda x: max(
                        x.get("auroc_unc_mean", 0),
                        x.get("auroc_unc_max",  0),
                        x.get("auroc_entropy",  0),
                    ), reverse=True):
        print(f"{r['ensemble_name']:<28} "
              f"{r.get('auroc_unc_mean', float('nan')):>9.4f} "
              f"{r.get('auprc_unc_mean', float('nan')):>9.4f} "
              f"{r.get('auroc_unc_max',  float('nan')):>9.4f} "
              f"{r.get('auprc_unc_max',  float('nan')):>9.4f} "
              f"{r.get('auroc_entropy',  float('nan')):>10.4f} "
              f"{r.get('auprc_entropy',  float('nan')):>10.4f} | "
              f"{r['n_expert_uncertain']:>6} "
              f"{r.get('best_signal_by_auroc', '?'):>9}")
    print("=" * 120)
    print("mn=unc_mean  mx=unc_max  ent=entropy  |  Best = signal with highest AUROC")


def print_sensitivity_summary(sensitivity_best):
    """Print a compact table of the three reference thresholds for Mega_Ensemble."""
    mega = sensitivity_best[sensitivity_best["ensemble_name"] == "Mega_Ensemble"]
    if mega.empty:
        return

    print("\n" + "=" * 100)
    print(" Threshold Sensitivity Summary — Mega_Ensemble")
    print("=" * 100)
    print(f"{'Method':<22} {'Signal':<12} {'Percentile':>11} {'Threshold':>11} "
          f"{'n_flag':>7} {'%flag':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 100)

    order = [f"default_{config.UNCERTAINTY_PERCENTILE}pct", "sigma3", "best_f1_percentile"]
    for method in order:
        rows = mega[mega["selection_method"] == method]
        for _, row in rows.iterrows():
            pct_str = f"{row['percentile']:.1f}" if not pd.isna(row["percentile"]) else "μ+3σ"
            print(f"{method:<22} {row['signal']:<12} {pct_str:>11} "
                  f"{row['threshold']:>11.4f} "
                  f"{row['n_flagged']:>7} {row['pct_flagged']:>7.1f} "
                  f"{row['precision_uncertain']:>7.4f} "
                  f"{row['recall_uncertain']:>7.4f} "
                  f"{row['f1_uncertain']:>7.4f}")
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("Ensemble Evaluation and Uncertainty Quantification")
    print("=" * 65)
    config.ENSEMBLES_DIR.mkdir(parents=True, exist_ok=True)

    all_dual_samples = load_dual_expert_samples()
    cv_samples, test_dual_samples = split_holdout(all_dual_samples)

    cv_loader       = build_test_dataloader(cv_samples)
    holdout_loader  = build_test_dataloader(test_dual_samples)
    full_loader     = build_test_dataloader(all_dual_samples)

    expert_agreement_labels = np.array([s[2] for s in all_dual_samples])
    kl_labels_full          = np.array([s[1] for s in all_dual_samples])

    n_c = int((expert_agreement_labels == config.CERTAIN_LABEL).sum())
    n_u = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())

    print(f"\n  CV set (threshold):   {len(cv_samples)} samples")
    print(f"  Holdout (KL):         {len(test_dual_samples)} samples")
    print(f"  Full dataset (UQ):    {len(all_dual_samples)} samples")
    print(f"    Expert-certain:     {n_c}")
    print(f"    Expert-uncertain:   {n_u}")

    np.save(config.ENSEMBLES_DIR / "expert_agreement_labels.npy", expert_agreement_labels)

    # Placeholder thresholds file — updated after the first ensemble runs.
    # visualize.py reads this to draw threshold lines on histograms.
    _thresholds_file = config.ENSEMBLES_DIR / "uq_thresholds.json"
    if not _thresholds_file.exists():
        _thresholds_file.write_text(
            json.dumps({"unc_mean": 0.0, "unc_max": 0.0, "entropy": 0.0}, indent=2)
        )

    # ------------------------------------------------------------------
    # STEP 1 & 2: Evaluate all ensembles with default CV thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" STEP 1 & 2: Evaluate Ensembles (with specific CV thresholds)")
    print("=" * 65)

    kl_metrics_all: list[dict]            = []
    uq_results_all: list[dict]            = []
    all_flags_dict: dict[str, np.ndarray] = {}
    # Stored for later sensitivity analysis — avoid second forward pass
    all_cv_unc:   dict[str, dict]         = {}
    all_eval_unc: dict[str, dict]         = {}

    def _run(name, ensemble):
        # 1. CV scores — single forward pass, reused for sweep
        cv_unc = _get_cv_uncertainty_scores(name, ensemble, cv_loader)
        all_cv_unc[name] = cv_unc

        # 2. Default thresholds
        thresholds = compute_thresholds_from_cv(name, cv_unc)

        # 3. KL classification metrics + full-dataset uncertainty scores
        metrics, unc_mean, unc_max, entropy, _ = evaluate_ensemble(
            name, ensemble, holdout_loader, full_loader)
        kl_metrics_all.append(metrics)

        eval_unc = {"unc_mean": unc_mean, "unc_max": unc_max, "entropy": entropy}
        all_eval_unc[name] = eval_unc

        # 4. UQ detection with default threshold
        uq_res, best_flags = evaluate_uncertainty_detection(
            name, unc_mean, unc_max, entropy,
            expert_agreement_labels, kl_labels_full, thresholds)
        uq_results_all.append(uq_res)
        all_flags_dict[name] = best_flags

        # Update per-ensemble thresholds file.
        # visualize.py reads this dict to draw the correct threshold line
        # on each ensemble's histogram.
        _thr_file = config.ENSEMBLES_DIR / "uq_thresholds.json"
        if _thr_file.exists():
            with open(_thr_file) as _f:
                _all_thr = json.load(_f)
        else:
            _all_thr = {}
        _all_thr[name] = {
            "unc_mean": thresholds["unc_mean"],
            "unc_max":  thresholds["unc_max"],
            "entropy":  thresholds["entropy"],
        }
        with open(_thr_file, "w") as _f:
            json.dump(_all_thr, _f, indent=2)

        del ensemble
        torch.cuda.empty_cache()

    for cfg in config.MODELS_CONFIG:
        _run(f"{cfg['name']}_Homogeneous", build_homogeneous_ensemble(cfg))

    _run("Heterogeneous_Avg",      build_heterogeneous_ensemble())
    _run("Heterogeneous_Weighted", build_weighted_ensemble())
    _run("Mega_Ensemble",          build_mega_ensemble())

    # ------------------------------------------------------------------
    # STEP 3: Threshold Sensitivity Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" STEP 3: Threshold Sensitivity Analysis (percentile sweep + 3σ)")
    print("=" * 65)

    sensitivity_frames = []
    for ens_name in all_eval_unc:
        df_sens = run_threshold_sensitivity_analysis(
            ens_name,
            all_cv_unc[ens_name],
            all_eval_unc[ens_name],
            expert_agreement_labels,
        )
        sensitivity_frames.append(df_sens)

    sensitivity_all  = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity_best = select_best_thresholds(sensitivity_all)

    # Save CSVs
    sensitivity_all.to_csv(
        config.ENSEMBLES_DIR / "threshold_sensitivity_all.csv",  index=False)
    sensitivity_best.to_csv(
        config.ENSEMBLES_DIR / "threshold_sensitivity_best.csv", index=False)
    print(f"  Sensitivity CSVs saved to {config.ENSEMBLES_DIR}")

    # ------------------------------------------------------------------
    # STEP 4: Build consensus matrices for all three threshold methods
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" STEP 4: Building Consensus Matrices")
    print("=" * 65)

    # Default 95th percentile — uses flags already computed in Step 2
    consensus_default = _build_consensus_df(
        all_dual_samples, all_flags_dict, kl_labels_full, expert_agreement_labels)

    # μ + 3σ
    consensus_sigma3 = build_consensus_from_sensitivity(
        all_dual_samples, all_eval_unc, sensitivity_all,
        kl_labels_full, expert_agreement_labels, "sigma3")

    # Best percentile by F1
    consensus_best = build_consensus_from_sensitivity(
        all_dual_samples, all_eval_unc, sensitivity_all,
        kl_labels_full, expert_agreement_labels, "best_f1_percentile")

    # Save CSVs
    consensus_default.to_csv(
        config.ENSEMBLES_DIR / "consensus_default_95pct.csv", index=False)
    consensus_sigma3.to_csv(
        config.ENSEMBLES_DIR / "consensus_sigma3.csv",        index=False)
    consensus_best.to_csv(
        config.ENSEMBLES_DIR / "consensus_best_threshold.csv", index=False)
    print(f"  Consensus CSVs saved to {config.ENSEMBLES_DIR}")

    # ------------------------------------------------------------------
    # Mann-Whitney U test (Mega_Ensemble, best signal by AUROC)
    # ------------------------------------------------------------------
    from evaluate import mann_whitney_uncertainty_test

    mega_uq  = next(r for r in uq_results_all if r["ensemble_name"] == "Mega_Ensemble")
    best_sig = mega_uq.get("best_signal_by_auroc", "unc_max")
    unc_for_mw = all_eval_unc["Mega_Ensemble"][best_sig]

    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    print(f"\n  Mann-Whitney U test — Mega Ensemble  [{best_sig}]")
    print(f"    n_certain={mask_c.sum()},  n_uncertain={mask_u.sum()}")

    mw_result = mann_whitney_uncertainty_test(unc_for_mw[mask_c], unc_for_mw[mask_u])
    mw_result["ensemble_name"] = "Mega_Ensemble"
    mw_result["unc_signal"]    = best_sig
    mw_result["p_significant"] = bool(mw_result["p_value"] < 0.05)

    with open(config.ENSEMBLES_DIR / "Mega_Ensemble_mann_whitney.json", "w") as f:
        json.dump(mw_result, f, indent=2)

    # ------------------------------------------------------------------
    # Save master Excel (extended)
    # ------------------------------------------------------------------
    save_master_results_to_excel(
        kl_metrics_all, uq_results_all, [mw_result],
        consensus_default, sensitivity_all, sensitivity_best,
        consensus_sigma3, consensus_best,
    )

    # ------------------------------------------------------------------
    # Print summaries
    # ------------------------------------------------------------------
    print_kl_summary_table(kl_metrics_all)
    print_uq_detection_table(uq_results_all)
    print_sensitivity_summary(sensitivity_best)

    print("\nResults saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
