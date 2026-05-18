"""
visualize.py
============
Generates all thesis figures.  Must be run after ensemble.py.

Figures produced
----------------
Per-ensemble:
  *_unc_mean_histogram.png
  *_unc_max_histogram.png
  *_entropy_histogram.png
  *_unc_max_sigma_analysis.png   (methodological illustration only)

Across all ensembles:
  all_ensembles_unc_max_boxplot.png
  all_ensembles_entropy_boxplot.png
  roc_unc_max.png
  roc_entropy.png
  kl_breakdown_stacked_bar.png   (NEW - KL breakdown of flagged samples)
  consensus_histogram.png        (NEW - distribution of ensemble consensus)

Per individual model (from main.py OOF files):
  *_reliability_diagram.png
  *_confidence_histogram.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
import config

FIGURES_DIR = config.RESULTS_DIR / "figures"

COLORS = {
    "certain": "#2196F3",
    "uncertain": "#F44336",
    "neutral": "#90CAF9",
}


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_cmap(name, n):
    return matplotlib.colormaps[name].resampled(n)


# =============================================================================
# HISTOGRAM — unc(x) distribution: certain vs uncertain
# =============================================================================

def plot_uncertainty_histogram(ensemble_name, uncertainty_scores,
                               expert_agreement_labels, threshold,
                               signal_label="unc_max"):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncertainty_scores[mask_c], bins=40, alpha=0.6,
            color=COLORS["certain"],
            label=f"Certain (n={mask_c.sum()})", density=True)
    ax.hist(uncertainty_scores[mask_u], bins=40, alpha=0.6,
            color=COLORS["uncertain"],
            label=f"Uncertain (n={mask_u.sum()})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold (CV 95th pct) = {threshold:.4f}")
    ax.set_xlabel(f"Uncertainty signal: {signal_label}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Uncertainty distribution [{signal_label}]: {ensemble_name}",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_{signal_label}_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved: {out_path}")
    return out_path


# =============================================================================
# BOX PLOT — comparison across all ensembles for one signal
# =============================================================================

def plot_uncertainty_boxplot(ensemble_names, uncertainties_dict,
                             expert_agreement_labels, signal_key="unc_max"):
    _ensure_dir()
    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    n = len(ensemble_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 6))

    positions_c = [i * 3 for i in range(n)]
    positions_u = [i * 3 + 1 for i in range(n)]

    ax.boxplot([uncertainties_dict[nm][signal_key][mask_c] for nm in ensemble_names],
               positions=positions_c, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["certain"], alpha=0.7))
    ax.boxplot([uncertainties_dict[nm][signal_key][mask_u] for nm in ensemble_names],
               positions=positions_u, widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=COLORS["uncertain"], alpha=0.7))

    tick_pos = [(positions_c[i] + positions_u[i]) / 2 for i in range(n)]
    short_names = [nm.replace("_Homogeneous", "\nHom.").replace("_", "\n")
                   for nm in ensemble_names]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel(signal_key, fontsize=11)
    ax.set_title(f"Ensemble uncertainty [{signal_key}]: certain vs uncertain",
                 fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color=COLORS["certain"], label="Certain (experts agree)"),
        mpatches.Patch(color=COLORS["uncertain"], label="Uncertain (experts disagree)"),
    ], fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"all_ensembles_{signal_key}_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Box plot saved: {out_path}")
    return out_path


# =============================================================================
# ROC + AUPRC curves — one curve per ensemble for one signal
# =============================================================================

def plot_roc_curves(ensemble_names, uncertainties_dict,
                    expert_agreement_labels, signal_key="unc_max"):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = _get_cmap("tab10", len(ensemble_names))

    for i, name in enumerate(ensemble_names):
        unc = uncertainties_dict[name][signal_key]
        try:
            fpr, tpr, _ = roc_curve(expert_agreement_labels, unc,
                                    pos_label=config.UNCERTAIN_LABEL)
            auc = roc_auc_score(expert_agreement_labels, unc)
            auprc = average_precision_score(expert_agreement_labels, unc)
            ax.plot(fpr, tpr, color=cmap(i), linewidth=1.8,
                    label=f"{name.replace('_', ' ')} "
                          f"(AUC={auc:.3f}, AP={auprc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC [{signal_key}]: uncertainty detection vs expert agreement",
                 fontsize=12)
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
    std_unc = np.std(uncertainty_scores)
    f1_list, pct_list = [], []

    for sigma in sigma_range:
        thr = mean_unc + sigma * std_unc
        flags = (uncertainty_scores > thr).astype(int)
        f1_list.append(f1_score(expert_agreement_labels, flags,
                                pos_label=config.UNCERTAIN_LABEL, zero_division=0))
        pct_list.append(100 * flags.mean())

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(sigma_range, f1_list, color="#E91E63", linewidth=2,
             label="F1 (uncertain)")
    ax2.plot(sigma_range, pct_list, color="#3F51B5", linewidth=2,
             linestyle="--", label="% flagged")
    ax1.axvline(3.0, color="black", linestyle=":", linewidth=1.5,
                label=f"Reference 3.0σ")
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


# =============================================================================
# Reliability diagram & Confidence Histogram
# =============================================================================

def plot_reliability_diagram(model_name, y_probs, y_true, n_bins=10):
    _ensure_dir()
    confidences = np.max(y_probs, axis=1)
    correct = (np.argmax(y_probs, axis=1) == y_true).astype(float)

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
    return out_path


def plot_confidence_histogram(model_name, y_probs, bins=30):
    _ensure_dir()
    confidences = np.max(y_probs, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidences, bins=bins, color=COLORS["certain"],
            alpha=0.8, edgecolor="white")
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
    return out_path


# =============================================================================
# NEW: Consensus and KL Breakdown Plots
# =============================================================================

def plot_kl_breakdown_of_flagged(consensus_df):
    """ Tworzy skumulowany wykres słupkowy pokazujący z jakich klas KL składają się flagowane próbki """
    _ensure_dir()

    # Wyciągnij nazwy kolumn z ensemblami (z pominięciem metadanych)
    skip_cols = {"Image_Name", "KL_Label", "Expert_Uncertain", "Is_Holdout", "Total_Flags"}
    ensemble_cols = [c for c in consensus_df.columns if c not in skip_cols]

    kl_order = config.CLASS_DISPLAY_NAMES

    # Zlicz wystąpienia klas KL dla oflagowanych próbek w każdym ensemblu
    breakdown = {kl: [] for kl in kl_order}
    for ens in ensemble_cols:
        flagged_df = consensus_df[consensus_df[ens] == 1]
        counts = flagged_df["KL_Label"].value_counts()
        for kl in kl_order:
            breakdown[kl].append(counts.get(kl, 0))

    # Rysowanie skumulowanego wykresu
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(ensemble_cols))

    # Intuicyjne medyczne kolory: Od zielonego (KL0) do Czerwonego (KL4)
    kl_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

    for kl, color in zip(kl_order, kl_colors):
        ax.bar(ensemble_cols, breakdown[kl], bottom=bottom, label=kl, color=color, alpha=0.85, edgecolor='white')
        bottom += np.array(breakdown[kl])

    ax.set_xticks(range(len(ensemble_cols)))
    short_names = [nm.replace("_Homogeneous", "\nHom.").replace("Heterogeneous", "Het.").replace("_", " ") for nm in
                   ensemble_cols]
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)

    ax.set_ylabel("Number of flagged samples", fontsize=11)
    ax.set_title("KL Class Breakdown of 'Uncertain' Samples per Ensemble", fontsize=13, fontweight='bold')

    # Odwróć legendę żeby KL4 było na górze tak jak na wykresie
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="KL Class", fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = FIGURES_DIR / "kl_breakdown_stacked_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Stacked Bar Chart saved: {out_path}")


def plot_consensus_histogram(consensus_df):
    """ Tworzy histogram pokazujący, przez ile modeli flagowane były dane próbki """
    _ensure_dir()

    max_flags = int(consensus_df["Total_Flags"].max())
    if max_flags == 0:
        print("  Brak flagowanych próbek - pomijam wykres konsensusu.")
        return

    # Zakres od 1 (oflagowane przez przynajmniej 1 model) do max_flags
    x = np.arange(1, max_flags + 1)
    counts_certain = []
    counts_uncertain = []

    for i in x:
        subset = consensus_df[consensus_df["Total_Flags"] == i]
        counts_certain.append(len(subset[subset["Expert_Uncertain"] == 0]))
        counts_uncertain.append(len(subset[subset["Expert_Uncertain"] == 1]))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, counts_certain, color=COLORS["certain"], label="Certain (Experts Agree)", alpha=0.8, edgecolor='white')
    ax.bar(x, counts_uncertain, bottom=counts_certain, color=COLORS["uncertain"], label="Uncertain (Experts Disagree)",
           alpha=0.8, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xlabel("Number of Ensembles Flagging the Sample", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Consensus Distribution: How many ensembles flagged the same samples?", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Dodaj tekst z liczbą na czubku słupka
    for i, (c1, c2) in enumerate(zip(counts_certain, counts_uncertain)):
        total = c1 + c2
        if total > 0:
            ax.text(x[i], total + max(counts_certain) * 0.02, str(total), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "consensus_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Consensus Histogram saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    _ensure_dir()

    # Expert labels
    expert_labels_path = config.ENSEMBLES_DIR / "expert_agreement_labels.npy"
    if not expert_labels_path.exists():
        print(f"Missing: {expert_labels_path}. Run ensemble.py first.")
        return
    expert_agreement_labels = np.load(expert_labels_path)
    print(f"Expert labels: {len(expert_agreement_labels)} samples")

    # Per-signal thresholds
    thresholds_path = config.ENSEMBLES_DIR / "uq_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {"unc_mean": 0.0, "unc_max": 0.0, "entropy": 0.0}

    # Load all uncertainty signals
    uncertainties: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(config.ENSEMBLES_DIR.glob("*_uncertainty.npz")):
        name = npz_path.stem.replace("_uncertainty", "")
        data = np.load(npz_path)
        if "unc_max" not in data:
            continue
        uncertainties[name] = {
            "unc_mean": data["unc_mean"],
            "unc_max": data["unc_max"],
            "entropy": data["entropy"],
        }

    ensemble_names = list(uncertainties.keys())

    if not ensemble_names:
        print("No uncertainty files found. Run ensemble.py first.")
        return

    # Ensemble-level comparison plots
    for sig in ["unc_max", "entropy"]:
        plot_uncertainty_boxplot(ensemble_names, uncertainties,
                                 expert_agreement_labels, signal_key=sig)
        plot_roc_curves(ensemble_names, uncertainties,
                        expert_agreement_labels, signal_key=sig)

    # Per-ensemble histograms
    for name, sigs in uncertainties.items():
        for sig in ["unc_mean", "unc_max", "entropy"]:
            plot_uncertainty_histogram(
                name, sigs[sig], expert_agreement_labels,
                thresholds.get(sig, 0.0), signal_label=sig)

        plot_sigma_analysis(name, sigs["unc_max"],
                            expert_agreement_labels, signal_label="unc_max")

    # Per-model reliability
    print("\n--- Reliability Diagrams & Confidence Histograms ---")
    for probs_path in sorted(config.INDIVIDUAL_MODELS_DIR.glob("*_test_probs.npz")):
        model_name = probs_path.stem.replace("_test_probs", "")
        data = np.load(probs_path)
        if "y_probs" not in data or "y_true" not in data:
            continue
        plot_reliability_diagram(model_name, data["y_probs"], data["y_true"])
        plot_confidence_histogram(model_name, data["y_probs"])

    # =========================================================================
    # NEW: Draw Consensus and KL Breakdown Plots from Excel
    # =========================================================================
    print("\n--- Consensus & KL Breakdown Plots ---")
    consensus_path = config.ENSEMBLES_DIR / "MASTER_RESULTS_SUMMARY.xlsx"
    if consensus_path.exists():
        try:
            # Read the specific sheet we generated in ensemble.py
            consensus_df = pd.read_excel(consensus_path, sheet_name="Consensus_Matrix")
            plot_kl_breakdown_of_flagged(consensus_df)
            plot_consensus_histogram(consensus_df)
        except Exception as e:
            print(f"  Could not read 'Consensus_Matrix' sheet from Excel: {e}")
    else:
        print("  MASTER_RESULTS_SUMMARY.xlsx not found. Ensure ensemble.py ran successfully.")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()