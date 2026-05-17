from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
import config

FIGURES_DIR = config.FIGURES_DIR

COLORS = {
    "certain":   "#2196F3",
    "uncertain": "#F44336",
    "neutral":   "#90CAF9",
}

def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _get_cmap(name, n):
    return matplotlib.colormaps[name].resampled(n)

def plot_uncertainty_histogram(ensemble_name, uncertainty_scores,
                               expert_agreement_labels, threshold,
                               signal_label="unc_max"):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncertainty_scores[mask_c], bins=40, alpha=0.6,
            color=COLORS["certain"], label=f"Certain (n={mask_c.sum()})", density=True)
    ax.hist(uncertainty_scores[mask_u], bins=40, alpha=0.6,
            color=COLORS["uncertain"], label=f"Uncertain (n={mask_u.sum()})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold (CV 95th pct) = {threshold:.4f}")
    ax.set_xlabel(f"Uncertainty signal: {signal_label}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Uncertainty distribution [{signal_label}]: {ensemble_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_{signal_label}_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved: {out_path}")
    return out_path

def plot_uncertainty_boxplot(ensemble_names, uncertainties_dict,
                             expert_agreement_labels, signal_key="unc_max"):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    n   = len(ensemble_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 6))

    positions_c = [i * 3     for i in range(n)]
    positions_u = [i * 3 + 1 for i in range(n)]

    ax.boxplot([uncertainties_dict[nm][signal_key][mask_c] for nm in ensemble_names],
               positions=positions_c, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["certain"], alpha=0.7))
    ax.boxplot([uncertainties_dict[nm][signal_key][mask_u] for nm in ensemble_names],
               positions=positions_u, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["uncertain"], alpha=0.7))

    tick_pos    = [(positions_c[i] + positions_u[i]) / 2 for i in range(n)]
    short_names = [nm.replace("_Homogeneous", "\nHom.").replace("_", "\n") for nm in ensemble_names]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel(signal_key, fontsize=11)
    ax.set_title(f"Ensemble uncertainty [{signal_key}]: certain vs uncertain", fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color=COLORS["certain"],   label="Certain (experts agree)"),
        mpatches.Patch(color=COLORS["uncertain"], label="Uncertain (experts disagree)"),
    ], fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"all_ensembles_{signal_key}_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Box plot saved: {out_path}")
    return out_path


def plot_roc_curves(ensemble_names, uncertainties_dict,
                    expert_agreement_labels, signal_key="unc_max"):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = _get_cmap("tab10", len(ensemble_names))

    for i, name in enumerate(ensemble_names):
        unc = uncertainties_dict[name][signal_key]
        try:
            fpr, tpr, _ = roc_curve(expert_agreement_labels, unc, pos_label=config.UNCERTAIN_LABEL)
            auc   = roc_auc_score(expert_agreement_labels, unc)
            auprc = average_precision_score(expert_agreement_labels, unc)
            ax.plot(fpr, tpr, color=cmap(i), linewidth=1.8,
                    label=f"{name.replace('_', ' ')} (AUC={auc:.3f}, AP={auprc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC [{signal_key}]: uncertainty detection vs expert agreement", fontsize=12)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"roc_{signal_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved: {out_path}")
    return out_path

def plot_sigma_analysis(ensemble_name, uncertainty_scores,
                        expert_agreement_labels, sigma_range=None,
                        signal_label="unc_max"):
    _ensure_dir()
    if sigma_range is None:
        sigma_range = np.arange(0.5, 5.1, 0.25)

    mean_unc = np.mean(uncertainty_scores)
    std_unc  = np.std(uncertainty_scores)
    f1_list, pct_list = [], []

    for sigma in sigma_range:
        thr   = mean_unc + sigma * std_unc
        flags = (uncertainty_scores > thr).astype(int)
        f1_list.append(f1_score(expert_agreement_labels, flags,
                                pos_label=config.UNCERTAIN_LABEL, zero_division=0))
        pct_list.append(100 * flags.mean())

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(sigma_range, f1_list,  color="#E91E63", linewidth=2, label="F1 (uncertain)")
    ax2.plot(sigma_range, pct_list, color="#3F51B5", linewidth=2, linestyle="--", label="% flagged")
    ax1.axvline(config.UNCERTAINTY_SIGMA_MULTIPLIER, color="black", linestyle=":", linewidth=1.5,
                label=f"Reference {config.UNCERTAINTY_SIGMA_MULTIPLIER}σ")
    ax1.set_xlabel("Sigma multiplier", fontsize=11)
    ax1.set_ylabel("F1 (uncertain)", fontsize=11, color="#E91E63")
    ax2.set_ylabel("% flagged samples", fontsize=11, color="#3F51B5")
    ax1.set_title(
        f"σ-threshold sensitivity [{signal_label}]: {ensemble_name}\n"
        "(Illustrative — operational threshold is CV 95th percentile)",
        fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_{signal_label}_sigma_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Sigma analysis saved: {out_path}")
    return out_path


def plot_reliability_diagram(model_name, y_probs, y_true, n_bins=10):
    _ensure_dir()
    confidences = np.max(y_probs, axis=1)
    correct     = (np.argmax(y_probs, axis=1) == y_true).astype(float)

    bins, bin_confs, bin_accs = np.linspace(0.0, 1.0, n_bins + 1), [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(float(confidences[mask].mean()))
        bin_accs.append(float(correct[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.65,
           color=COLORS["certain"], align="center", label="Model")
    ax.set_xlabel("Mean confidence (max softmax)", fontsize=11)
    ax.set_ylabel("Fraction correct", fontsize=11)
    ax.set_title(f"Reliability Diagram: {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{model_name}_reliability_diagram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Reliability diagram saved: {out_path}")
    return out_path


def plot_confidence_histogram(model_name, y_probs, bins=30):
    _ensure_dir()
    confidences = np.max(y_probs, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidences, bins=bins, color=COLORS["certain"], alpha=0.8, edgecolor="white")
    ax.axvline(confidences.mean(), color="black", linestyle="--",
               linewidth=1.5, label=f"Mean = {confidences.mean():.3f}")
    ax.set_xlabel("Max softmax probability (confidence)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Confidence Distribution: {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{model_name}_confidence_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confidence histogram saved: {out_path}")
    return out_path


def main():
    _ensure_dir()

    expert_labels_path = config.ENSEMBLES_DIR / "expert_agreement_labels.npy"
    if not expert_labels_path.exists():
        print(f"Missing: {expert_labels_path}. Run ensemble.py first.")
        return
    expert_agreement_labels = np.load(expert_labels_path)
    print(f"Expert labels: {len(expert_agreement_labels)} samples")

    thresholds_path = config.ENSEMBLES_DIR / "uq_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        print("WARNING: uq_thresholds.json not found. Using 0 for all thresholds.")
        thresholds = {"unc_mean": 0.0, "unc_max": 0.0, "entropy": 0.0}

    uncertainties: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(config.ENSEMBLES_DIR.glob("*_uncertainty.npz")):
        name = npz_path.stem.replace("_uncertainty", "")
        data = np.load(npz_path)
        if "unc_max" not in data:
            print(f"  Skipping {name} — old format (no unc_max).")
            continue
        uncertainties[name] = {
            "unc_mean": data["unc_mean"],
            "unc_max":  data["unc_max"],
            "entropy":  data["entropy"],
        }
        print(f"  Loaded: {name}  ({len(data['unc_max'])} samples)")

    ensemble_names = list(uncertainties.keys())

    if not ensemble_names:
        print("No uncertainty files found. Run ensemble.py first.")
        return

    for sig in ["unc_max", "entropy"]:
        plot_uncertainty_boxplot(ensemble_names, uncertainties,
                                 expert_agreement_labels, signal_key=sig)
        plot_roc_curves(ensemble_names, uncertainties,
                        expert_agreement_labels, signal_key=sig)

    for name, sigs in uncertainties.items():
        for sig in ["unc_mean", "unc_max", "entropy"]:
            plot_uncertainty_histogram(
                name, sigs[sig], expert_agreement_labels,
                thresholds.get(sig, 0.0), signal_label=sig)

        plot_sigma_analysis(name, sigs["unc_max"],
                            expert_agreement_labels, signal_label="unc_max")

    print("\n--- Reliability Diagrams & Confidence Histograms ---")
    for probs_path in sorted(config.INDIVIDUAL_MODELS_DIR.glob("*_test_probs.npz")):
        model_name = probs_path.stem.replace("_test_probs", "")
        data       = np.load(probs_path)
        if "y_probs" not in data or "y_true" not in data:
            print(f"  Skipping {model_name}.")
            continue
        plot_reliability_diagram(model_name, data["y_probs"], data["y_true"])
        plot_confidence_histogram(model_name, data["y_probs"])

    print(f"\nAll figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()