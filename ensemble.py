import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
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


def compute_thresholds_from_cv(ensemble_name, ensemble_model, cv_loader):
    pct = config.UNCERTAINTY_PERCENTILE
    print(f"  Computing uncertainty thresholds from CV set ({pct}th percentile) for {ensemble_name}...")

    _, _, _, unc_mean_cv, unc_max_cv, entropy_cv = _run_forward(ensemble_model, cv_loader)

    thresholds = {
        "unc_mean": float(np.percentile(unc_mean_cv, pct)),
        "unc_max": float(np.percentile(unc_max_cv, pct)),
        "entropy": float(np.percentile(entropy_cv, pct)),
        "percentile_used": pct,
        "cv_samples": len(unc_mean_cv)
    }

    print(
        f"    Thresholds -> unc_mean: {thresholds['unc_mean']:.4f}, unc_max: {thresholds['unc_max']:.4f}, entropy: {thresholds['entropy']:.4f}")
    return thresholds


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
    kl_metrics["mean_unc_max"] = round(float(np.mean(unc_max)), 6)
    kl_metrics["mean_entropy"] = round(float(np.mean(entropy)), 6)

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

    f1_unc = float(f1_score(expert_labels, flags, pos_label=config.UNCERTAIN_LABEL, zero_division=0))
    cm = confusion_matrix(expert_labels, flags, labels=[config.CERTAIN_LABEL, config.UNCERTAIN_LABEL])

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
        f"auroc_{signal_name}": round(auroc, 4),
        f"auprc_{signal_name}": round(auprc, 4),
        f"f1_uncertain_{signal_name}": round(f1_unc, 4),
        f"n_flagged_{signal_name}": int(flags.sum()),
        f"mean_unc_certain_{signal_name}": round(mu_c, 6),
        f"mean_unc_uncertain_{signal_name}": round(mu_u, 6),
        f"cm_{signal_name}": cm.tolist(),
    }


def evaluate_uncertainty_detection(ensemble_name, unc_mean, unc_max, entropy,
                                   expert_agreement_labels, kl_labels_full, thresholds):
    n_total = len(unc_mean)
    n_exp_unc = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())

    print(f"\n{'=' * 65}")
    print(f"UQ VALIDATION: {ensemble_name}")
    print(f"{'=' * 65}")
    print(f"  Total samples (full dataset):     {n_total}")
    print(f"  Expert-uncertain (ground truth):  {n_exp_unc}  ({100 * n_exp_unc / n_total:.1f}%)")

    results = {
        "ensemble_name": ensemble_name,
        "n_total": n_total,
        "n_expert_uncertain": n_exp_unc,
        "threshold_unc_mean": round(thresholds["unc_mean"], 6),
        "threshold_unc_max": round(thresholds["unc_max"], 6),
        "threshold_entropy": round(thresholds["entropy"], 6),
    }

    for sig_name, scores in [("unc_mean", unc_mean), ("unc_max", unc_max), ("entropy", entropy)]:
        results.update(_eval_one_signal(sig_name, scores, expert_agreement_labels, thresholds[sig_name]))

    # Wybór najlepszego sygnału by AUROC
    best_signal = max(["unc_mean", "unc_max", "entropy"], key=lambda s: results.get(f"auroc_{s}", 0.0))
    best_flags = ({"unc_mean": unc_mean, "unc_max": unc_max, "entropy": entropy}[best_signal] > thresholds[
        best_signal]).astype(int)
    results["best_signal_by_auroc"] = best_signal

    # Breakdown KL dla oflagowanych
    flagged_kls = kl_labels_full[best_flags == 1]
    unique, counts = np.unique(flagged_kls, return_counts=True)
    kl_counts = dict(zip(unique, counts))
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


def save_master_results_to_excel(kl_metrics_all, uq_results_all, mw_results, consensus_df):
    df_kl = pd.DataFrame(kl_metrics_all)

    if "f1_per_class" in df_kl.columns:
        splits = pd.DataFrame(df_kl["f1_per_class"].tolist(),
                              columns=["F1_KL0", "F1_KL1", "F1_KL2", "F1_KL3", "F1_KL4"])
        df_kl = pd.concat([df_kl.drop("f1_per_class", axis=1), splits], axis=1)
    if "brier_per_class" in df_kl.columns:
        bp = pd.DataFrame(df_kl["brier_per_class"].tolist(),
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
        df_kl.to_excel(writer, sheet_name="KL_Classification", index=False)
        df_uq.to_excel(writer, sheet_name="UQ_Detection", index=False)
        pd.DataFrame(mw_results).to_excel(writer, sheet_name="Mann_Whitney_Test", index=False)
        consensus_df.to_excel(writer, sheet_name="Consensus_Matrix", index=False)
    print(f"\n  Master results saved: {excel_path}")


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
                        x.get("auroc_unc_max", 0),
                        x.get("auroc_entropy", 0),
                    ), reverse=True):
        print(f"{r['ensemble_name']:<28} "
              f"{r.get('auroc_unc_mean', float('nan')):>9.4f} "
              f"{r.get('auprc_unc_mean', float('nan')):>9.4f} "
              f"{r.get('auroc_unc_max', float('nan')):>9.4f} "
              f"{r.get('auprc_unc_max', float('nan')):>9.4f} "
              f"{r.get('auroc_entropy', float('nan')):>10.4f} "
              f"{r.get('auprc_entropy', float('nan')):>10.4f} | "
              f"{r['n_expert_uncertain']:>6} "
              f"{r.get('best_signal_by_auroc', '?'):>9}")
    print("=" * 120)
    print("mn=unc_mean  mx=unc_max  ent=entropy  |  Best = signal with highest AUROC")


def main():
    print("=" * 65)
    print("Ensemble Evaluation and Uncertainty Quantification")
    print("=" * 65)
    config.ENSEMBLES_DIR.mkdir(parents=True, exist_ok=True)

    all_dual_samples = load_dual_expert_samples()
    cv_samples, test_dual_samples = split_holdout(all_dual_samples)

    cv_loader = build_test_dataloader(cv_samples)
    holdout_loader = build_test_dataloader(test_dual_samples)
    full_loader = build_test_dataloader(all_dual_samples)

    expert_agreement_labels = np.array([s[2] for s in all_dual_samples])
    kl_labels_full = np.array([s[1] for s in all_dual_samples])

    n_c = int((expert_agreement_labels == config.CERTAIN_LABEL).sum())
    n_u = int((expert_agreement_labels == config.UNCERTAIN_LABEL).sum())

    print(f"\n  CV set (threshold):   {len(cv_samples)} samples")
    print(f"  Holdout (KL):         {len(test_dual_samples)} samples")
    print(f"  Full dataset (UQ):    {len(all_dual_samples)} samples")
    print(f"    Expert-certain:     {n_c}")
    print(f"    Expert-uncertain:   {n_u}")

    np.save(config.ENSEMBLES_DIR / "expert_agreement_labels.npy", expert_agreement_labels)

    print("\n" + "=" * 65)
    print(" STEP 1 & 2: Evaluate Ensembles (with specific CV thresholds)")
    print("=" * 65)

    kl_metrics_all = []
    uq_results_all = []
    all_unc_max_dict: dict[str, np.ndarray] = {}
    all_flags_dict = {}

    def _run(name, ensemble):
        # Najpierw wyliczamy próg tylko dla TEGO ensembla
        thresholds = compute_thresholds_from_cv(name, ensemble, cv_loader)

        # Oceniamy KL i pobieramy sygnały niepewności
        metrics, unc_mean, unc_max, entropy, _ = evaluate_ensemble(
            name, ensemble, holdout_loader, full_loader)
        kl_metrics_all.append(metrics)

        # Oceniamy sygnały UQ względem wyliczonych progów
        uq_res, best_flags = evaluate_uncertainty_detection(
            name, unc_mean, unc_max, entropy,
            expert_agreement_labels, kl_labels_full, thresholds)

        uq_results_all.append(uq_res)
        all_unc_max_dict[name] = unc_max
        all_flags_dict[name] = best_flags

        del ensemble
        torch.cuda.empty_cache()

    for cfg in config.MODELS_CONFIG:
        _run(f"{cfg['name']}_Homogeneous", build_homogeneous_ensemble(cfg))

    _run("Heterogeneous_Avg", build_heterogeneous_ensemble())
    _run("Heterogeneous_Weighted", build_weighted_ensemble())
    _run("Mega_Ensemble", build_mega_ensemble())

    # Generowanie Macierzy Konsensusu
    df_data = {
        "Image_Name": [s[0].name for s in all_dual_samples],
        "KL_Label": [config.CLASS_DISPLAY_NAMES[kl] for kl in kl_labels_full],
        "Expert_Uncertain": expert_agreement_labels
    }
    for name, flags in all_flags_dict.items():
        df_data[name] = flags

    consensus_df = pd.DataFrame(df_data)
    ensemble_cols = list(all_flags_dict.keys())
    consensus_df["Total_Flags"] = consensus_df[ensemble_cols].sum(axis=1)
    consensus_df = consensus_df.sort_values(by=["Total_Flags", "Expert_Uncertain"], ascending=[False, False])

    # Mann-Whitney U Test
    from evaluate import mann_whitney_uncertainty_test

    mega_uq = next(r for r in uq_results_all if r["ensemble_name"] == "Mega_Ensemble")
    best_sig = mega_uq.get("best_signal_by_auroc", "unc_max")
    unc_for_mw = all_unc_max_dict["Mega_Ensemble"]

    if best_sig != "unc_max":
        npz = np.load(config.ENSEMBLES_DIR / "Mega_Ensemble_uncertainty.npz")
        unc_for_mw = npz[best_sig]

    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    print(f"\n  Mann-Whitney U test — Mega Ensemble  [{best_sig}]")
    print(f"    n_certain={mask_c.sum()},  n_uncertain={mask_u.sum()}")

    mw_result = mann_whitney_uncertainty_test(unc_for_mw[mask_c], unc_for_mw[mask_u])
    mw_result["ensemble_name"] = "Mega_Ensemble"
    mw_result["unc_signal"] = best_sig
    # Zabezpieczenie p_value na wypadek typu numpy.bool_ z scipy (wymuszamy pythonowy bool)
    mw_result["p_significant"] = bool(mw_result["p_value"] < 0.05)

    with open(config.ENSEMBLES_DIR / "Mega_Ensemble_mann_whitney.json", "w") as f:
        json.dump(mw_result, f, indent=2)

    save_master_results_to_excel(kl_metrics_all, uq_results_all, [mw_result], consensus_df)
    print_kl_summary_table(kl_metrics_all)
    print_uq_detection_table(uq_results_all)


if __name__ == "__main__":
    main()