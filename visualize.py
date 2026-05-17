from pathlib import Path

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import config


FIGURES_DIR = config.RESULTS_DIR / "figures"

COLORS = {
    "certain":   "#2196F3",
    "uncertain": "#F44336",
    "neutral":   "#90CAF9",
}


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_cmap(name, n):
    """
    Colormap helper compatible with matplotlib >= 3.7.
    matplotlib.colormaps[] replaces the deprecated cm.get_cmap().
    """
    return matplotlib.colormaps[name].resampled(n)


# =============================================================================
# HISTOGRAM — distribution of unc(x): certain vs uncertain
# =============================================================================

def plot_uncertainty_histogram(ensemble_name, uncertainty_scores,
                               expert_agreement_labels, threshold):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncertainty_scores[mask_c], bins=40, alpha=0.6, color=COLORS["certain"],
            label=f"Certain (n={mask_c.sum()})", density=True)
    ax.hist(uncertainty_scores[mask_u], bins=40, alpha=0.6, color=COLORS["uncertain"],
            label=f"Uncertain (n={mask_u.sum()})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold {config.UNCERTAINTY_SIGMA_MULTIPLIER}σ = {threshold:.4f}")
    ax.set_xlabel("Uncertainty  unc(x) = mean std of ensemble probabilities", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Uncertainty distribution: {ensemble_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_uncertainty_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved: {out_path}")
    return out_path


# =============================================================================
# BOX PLOT — comparison of unc(x) across all ensembles
# =============================================================================

def plot_uncertainty_boxplot(ensemble_names, uncertainties_dict, expert_agreement_labels):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    n = len(ensemble_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 6))

    positions_c = [i * 3     for i in range(n)]
    positions_u = [i * 3 + 1 for i in range(n)]

    ax.boxplot([uncertainties_dict[name][mask_c] for name in ensemble_names],
               positions=positions_c, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["certain"], alpha=0.7))
    ax.boxplot([uncertainties_dict[name][mask_u] for name in ensemble_names],
               positions=positions_u, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["uncertain"], alpha=0.7))

    tick_pos   = [(positions_c[i] + positions_u[i]) / 2 for i in range(n)]
    short_names = [name.replace("_Homogeneous", "\nHom.").replace("_", "\n")
                   for name in ensemble_names]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("unc(x)", fontsize=11)
    ax.set_title("Ensemble Uncertainty: certain vs uncertain (expert labels)", fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color=COLORS["certain"],   label="Certain (experts agree)"),
        mpatches.Patch(color=COLORS["uncertain"], label="Uncertain (experts disagree)"),
    ], fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / "all_ensembles_uncertainty_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Box plot saved: {out_path}")
    return out_path


# =============================================================================
# ROC curves — uncertain detection
# =============================================================================

def plot_roc_curves(ensemble_names, uncertainties_dict, expert_agreement_labels):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = _get_cmap("tab10", len(ensemble_names))

    for i, name in enumerate(ensemble_names):
        unc = uncertainties_dict[name]
        try:
            fpr, tpr, _ = roc_curve(expert_agreement_labels, unc,
                                     pos_label=config.UNCERTAIN_LABEL)
            auc = roc_auc_score(expert_agreement_labels, unc)
            ax.plot(fpr, tpr, color=cmap(i), linewidth=1.8,
                    label=f"{name.replace('_', ' ')} (AUC={auc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC: uncertainty detection (validated by expert agreement)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "roc_uncertain_detection.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved: {out_path}")
    return out_path


# =============================================================================
# Sigma analysis — effect of threshold multiplier on F1 and flagging rate
# =============================================================================

def plot_sigma_analysis(ensemble_name, uncertainty_scores,
                        expert_agreement_labels, sigma_range=None):
    _ensure_dir()
    if sigma_range is None:
        sigma_range = np.arange(0.5, 5.1, 0.25)

    mean_unc = np.mean(uncertainty_scores)
    std_unc  = np.std(uncertainty_scores)
    f1_scores, pct_flagged = [], []

    for sigma in sigma_range:
        threshold = mean_unc + sigma * std_unc
        flags = (uncertainty_scores > threshold).astype(int)
        f1_scores.append(f1_score(expert_agreement_labels, flags,
                                  pos_label=config.UNCERTAIN_LABEL, zero_division=0))
        pct_flagged.append(100 * flags.mean())

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(sigma_range, f1_scores,  color="#E91E63", linewidth=2, label="F1 (uncertain)")
    ax2.plot(sigma_range, pct_flagged, color="#3F51B5", linewidth=2,
             linestyle="--", label="% flagged")
    ax1.axvline(config.UNCERTAINTY_SIGMA_MULTIPLIER, color="black", linestyle=":",
                linewidth=1.5, label=f"Chosen: {config.UNCERTAINTY_SIGMA_MULTIPLIER}σ")
    ax1.set_xlabel("Sigma multiplier", fontsize=11)
    ax1.set_ylabel("F1 (uncertain)", fontsize=11, color="#E91E63")
    ax2.set_ylabel("% flagged samples", fontsize=11, color="#3F51B5")
    ax1.set_title(f"Effect of σ threshold on uncertain detection — {ensemble_name}", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_sigma_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Sigma analysis saved: {out_path}")
    return out_path


# =============================================================================
# Reliability diagram — calibration visualisation
#
# Shows whether the model's confidence matches its actual accuracy.
# A perfectly calibrated model lies on the diagonal.
# Reads from *_test_probs.npz files saved by evaluate.py during main.py's run.
# =============================================================================

def plot_reliability_diagram(model_name, y_probs, y_true, n_bins=10):
    _ensure_dir()

    confidences = np.max(y_probs, axis=1)
    correct     = (np.argmax(y_probs, axis=1) == y_true).astype(float)

    bins      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs  = []

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


# =============================================================================
# Main — called after ensemble.py has written all result files
# =============================================================================

def main():
    _ensure_dir()

    # ---- Expert agreement labels ----------------------------------------
    expert_labels_path = config.RESULTS_DIR / "expert_agreement_labels.npy"
    if not expert_labels_path.exists():
        print(f"Missing: {expert_labels_path}. Run ensemble.py first.")
        return

    expert_agreement_labels = np.load(expert_labels_path)
    print(f"Expert labels loaded: {len(expert_agreement_labels)} samples")
    print(f"  Certain:   {(expert_agreement_labels == config.CERTAIN_LABEL).sum()}")
    print(f"  Uncertain: {(expert_agreement_labels == config.UNCERTAIN_LABEL).sum()}")

    # ---- Ensemble uncertainties -----------------------------------------
    uncertainties: dict[str, np.ndarray] = {}
    for npz_path in sorted(config.RESULTS_DIR.glob("*_uncertainty.npz")):
        name = npz_path.stem.replace("_uncertainty", "")
        data = np.load(npz_path)
        uncertainties[name] = data["uncertainty"]
        print(f"  Loaded unc(x): {name}  ({len(data['uncertainty'])} samples)")

    ensemble_names = list(uncertainties.keys())

    # Ensemble-level plots
    plot_uncertainty_boxplot(ensemble_names, uncertainties, expert_agreement_labels)
    plot_roc_curves(ensemble_names, uncertainties, expert_agreement_labels)

    for name, unc in uncertainties.items():
        # Prefer the CV-calibrated threshold saved by ensemble.py
        threshold_json = config.RESULTS_DIR / f"{name}_threshold.json"
        uq_json        = config.RESULTS_DIR / f"{name}_uq_detection.json"

        if threshold_json.exists():
            with open(threshold_json) as f:
                threshold = json.load(f)["threshold"]
        elif uq_json.exists():
            with open(uq_json) as f:
                threshold = json.load(f)["threshold"]
        else:
            mean_unc  = np.mean(unc)
            std_unc   = np.std(unc)
            threshold = mean_unc + config.UNCERTAINTY_SIGMA_MULTIPLIER * std_unc

        plot_uncertainty_histogram(name, unc, expert_agreement_labels, threshold)
        plot_sigma_analysis(name, unc, expert_agreement_labels)

    # ---- Reliability diagrams — one per individual model ----------------
    # evaluate.py saves y_true / y_probs to results/*_test_probs.npz
    # during main.py's cross-validation run.
    print("\n--- Reliability Diagrams ---")
    probs_files = sorted(config.RESULTS_DIR.glob("*_test_probs.npz"))
    if not probs_files:
        print("  No *_test_probs.npz files found. Run main.py first.")
    else:
        for probs_path in probs_files:
            model_name = probs_path.stem.replace("_test_probs", "")
            data       = np.load(probs_path)
            if "y_probs" not in data or "y_true" not in data:
                print(f"  Skipping {model_name} — missing y_probs or y_true.")
                continue
            plot_reliability_diagram(model_name, data["y_probs"], data["y_true"])

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
