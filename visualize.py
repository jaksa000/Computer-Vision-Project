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
  *_unc_max_sigma_analysis.png

Across all ensembles:
  all_ensembles_unc_max_boxplot.png
  all_ensembles_entropy_boxplot.png
  roc_unc_max.png
  roc_entropy.png
  kl_breakdown_default_95pct.png
  kl_breakdown_sigma3.png
  kl_breakdown_best_threshold.png
  kl_breakdown_sigma3_vs_best.png   (side-by-side comparison)
  consensus_default_95pct.png
  consensus_sigma3.png
  consensus_best_threshold.png

Threshold sensitivity (Mega_Ensemble + all others):
  {name}_threshold_sensitivity_f1.png
  {name}_threshold_sensitivity_recall_vs_flagged.png

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
    "certain":   "#2196F3",
    "uncertain": "#F44336",
    "neutral":   "#90CAF9",
}

# Colours used for the three uncertainty signals across all sensitivity plots
SIGNAL_COLORS = {
    "unc_mean": "#E91E63",
    "unc_max":  "#9C27B0",
    "entropy":  "#FF9800",
}

# KL class colours: green (healthy) → red (severe)
KL_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_cmap(name, n):
    return matplotlib.colormaps[name].resampled(n)


# =============================================================================
# HISTOGRAM — uncertainty distribution: certain vs uncertain
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
               label=f"Threshold (CV {config.UNCERTAINTY_PERCENTILE}th pct) = {threshold:.4f}")
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


# =============================================================================
# BOX PLOT — comparison across all ensembles for one signal
# =============================================================================

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

    tick_pos = [(positions_c[i] + positions_u[i]) / 2 for i in range(n)]
    short_names = [nm.replace("_Homogeneous", "\nHom.").replace("_", "\n")
                   for nm in ensemble_names]
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


# =============================================================================
# ROC curves — one curve per ensemble for one signal
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
            auc  = roc_auc_score(expert_agreement_labels, unc)
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


# =============================================================================
# SIGMA ANALYSIS — F1 and % flagged vs sigma multiplier
# =============================================================================

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
    ax1.plot(sigma_range, f1_list, color="#E91E63", linewidth=2, label="F1 (uncertain)")
    ax2.plot(sigma_range, pct_list, color="#3F51B5", linewidth=2,
             linestyle="--", label="% flagged")
    ax1.axvline(config.UNCERTAINTY_SIGMA_MULTIPLIER, color="black", linestyle=":",
                linewidth=1.5, label=f"Reference {config.UNCERTAINTY_SIGMA_MULTIPLIER:.1f}σ")
    ax1.set_xlabel("Sigma multiplier", fontsize=11)
    ax1.set_ylabel("F1 (uncertain)", fontsize=11, color="#E91E63")
    ax2.set_ylabel("% flagged samples", fontsize=11, color="#3F51B5")
    ax1.set_title(
        f"σ-threshold sensitivity [{signal_label}]: {ensemble_name}\n"
        "(Illustrative — operational threshold is CV "
        f"{config.UNCERTAINTY_PERCENTILE}th percentile)",
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
# RELIABILITY DIAGRAM & CONFIDENCE HISTOGRAM
# =============================================================================

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
    return out_path


# =============================================================================
# KL BREAKDOWN — stacked bar chart of flagged samples per ensemble
# Generalised: accepts an out_name and title so it can render all three
# threshold variants (default 95pct, sigma3, best) with one function.
# =============================================================================

def plot_kl_breakdown_of_flagged(consensus_df,
                                  out_name="kl_breakdown_stacked_bar.png",
                                  title="KL Class Breakdown of 'Uncertain' Samples per Ensemble"):
    _ensure_dir()

    skip_cols = {"Image_Name", "KL_Label", "Expert_Uncertain", "Is_Holdout", "Total_Flags"}
    ensemble_cols = [c for c in consensus_df.columns if c not in skip_cols]
    if not ensemble_cols:
        print(f"  No ensemble columns found — skipping {out_name}")
        return

    kl_order = config.CLASS_DISPLAY_NAMES
    breakdown = {kl: [] for kl in kl_order}
    for ens in ensemble_cols:
        flagged_df = consensus_df[consensus_df[ens] == 1]
        counts     = flagged_df["KL_Label"].value_counts()
        for kl in kl_order:
            breakdown[kl].append(counts.get(kl, 0))

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom  = np.zeros(len(ensemble_cols))

    for kl, color in zip(kl_order, KL_COLORS):
        ax.bar(ensemble_cols, breakdown[kl], bottom=bottom,
               label=kl, color=color, alpha=0.85, edgecolor="white")
        bottom += np.array(breakdown[kl])

    ax.set_xticks(range(len(ensemble_cols)))
    short_names = [nm.replace("_Homogeneous", "\nHom.")
                     .replace("Heterogeneous", "Het.")
                     .replace("_", " ")
                   for nm in ensemble_cols]
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of flagged samples", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="KL Class", fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  KL breakdown saved: {out_path}")
    return out_path


# =============================================================================
# KL BREAKDOWN COMPARISON — 3σ vs best threshold, side-by-side per ensemble
# =============================================================================

def plot_kl_breakdown_comparison(consensus_sigma3, consensus_best):
    """
    Side-by-side stacked bar chart.
    For every ensemble there are two adjacent bars: one for the μ+3σ threshold
    and one for the best-F1 percentile threshold.  The bars are stacked by
    KL class so you can see both *how many* and *which grades* each method flags.
    """
    _ensure_dir()

    skip_cols = {"Image_Name", "KL_Label", "Expert_Uncertain", "Is_Holdout", "Total_Flags"}

    def _get_breakdown(df):
        ens_cols = [c for c in df.columns if c not in skip_cols]
        bd = {kl: [] for kl in config.CLASS_DISPLAY_NAMES}
        for ens in ens_cols:
            counts = df[df[ens] == 1]["KL_Label"].value_counts()
            for kl in config.CLASS_DISPLAY_NAMES:
                bd[kl].append(counts.get(kl, 0))
        return ens_cols, bd

    ens_cols_s3, breakdown_s3   = _get_breakdown(consensus_sigma3)
    ens_cols_best, breakdown_best = _get_breakdown(consensus_best)

    # Use whichever ensemble list is available (should be identical)
    ens_cols = ens_cols_s3 if ens_cols_s3 else ens_cols_best
    n = len(ens_cols)
    if n == 0:
        print("  No ensemble columns — skipping comparison chart.")
        return

    bar_width = 0.35
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n * 2.2), 7))

    # Stacked bars: sigma3 (left) and best (right)
    for source_label, breakdown, x_offset in [
        ("μ+3σ",           breakdown_s3,   x - bar_width / 2),
        ("Best Percentile", breakdown_best, x + bar_width / 2),
    ]:
        bottom = np.zeros(n)
        for kl, color in zip(config.CLASS_DISPLAY_NAMES, KL_COLORS):
            heights = np.array(breakdown.get(kl, [0] * n))
            ax.bar(x_offset, heights, bar_width, bottom=bottom,
                   color=color, alpha=0.85, edgecolor="white")
            bottom += heights

        # Total annotation above each bar group
        totals = np.array([
            sum(breakdown.get(kl, [0] * n)[i] for kl in config.CLASS_DISPLAY_NAMES)
            for i in range(n)
        ])
        for xi, tot in zip(x_offset, totals):
            if tot > 0:
                ax.text(xi, tot + 0.5, str(int(tot)),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Legend: KL classes + method hatching labels
    kl_patches  = [mpatches.Patch(facecolor=c, label=kl, alpha=0.85)
                   for c, kl in zip(KL_COLORS, config.CLASS_DISPLAY_NAMES)]
    method_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.4, label="Left bar  = μ+3σ"),
        mpatches.Patch(facecolor="grey", alpha=0.8, label="Right bar = Best Percentile"),
    ]
    ax.legend(handles=kl_patches[::-1] + method_patches,
              title="KL Class", fontsize=9, loc="upper right")

    short_names = [nm.replace("_Homogeneous", "\nHom.")
                     .replace("Heterogeneous", "Het.")
                     .replace("_", " ")
                   for nm in ens_cols]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of flagged samples", fontsize=11)
    ax.set_title("KL Breakdown of Flagged Samples: μ+3σ vs Best Percentile Threshold",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "kl_breakdown_sigma3_vs_best.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison KL breakdown saved: {out_path}")
    return out_path


# =============================================================================
# CONSENSUS HISTOGRAM
# Generalised: accepts an out_name and title.
# =============================================================================

def plot_consensus_histogram(consensus_df,
                              out_name="consensus_histogram.png",
                              title="Consensus Distribution: How many ensembles flagged the same samples?"):
    _ensure_dir()

    max_flags = int(consensus_df["Total_Flags"].max()) if "Total_Flags" in consensus_df.columns else 0
    if max_flags == 0:
        print(f"  No flagged samples — skipping {out_name}")
        return

    x = np.arange(1, max_flags + 1)
    counts_certain   = []
    counts_uncertain = []

    for i in x:
        subset = consensus_df[consensus_df["Total_Flags"] == i]
        counts_certain.append(len(subset[subset["Expert_Uncertain"] == config.CERTAIN_LABEL]))
        counts_uncertain.append(len(subset[subset["Expert_Uncertain"] == config.UNCERTAIN_LABEL]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, counts_certain, color=COLORS["certain"],
           label="Certain (Experts Agree)", alpha=0.8, edgecolor="white")
    ax.bar(x, counts_uncertain, bottom=counts_certain, color=COLORS["uncertain"],
           label="Uncertain (Experts Disagree)", alpha=0.8, edgecolor="white")

    # Total count on top of each bar
    for i, (c1, c2) in enumerate(zip(counts_certain, counts_uncertain)):
        total = c1 + c2
        if total > 0:
            ax.text(x[i], total + max(counts_certain + [1]) * 0.02,
                    str(total), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xlabel("Number of Ensembles Flagging the Sample", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Consensus histogram saved: {out_path}")
    return out_path


# =============================================================================
# THRESHOLD SENSITIVITY — F1 vs percentile
# =============================================================================

def plot_threshold_sensitivity_f1(ensemble_name, sensitivity_df):
    """
    Line chart: F1 uncertain vs percentile for each uncertainty signal.

    Annotations
    -----------
    - Vertical dashed line : default config.UNCERTAINTY_PERCENTILE
    - Vertical dotted line : best percentile per signal (one per signal colour)
    - Horizontal marker    : F1 achieved by the μ+3σ method (shown as a ★ scatter)
    """
    _ensure_dir()

    ens_df  = sensitivity_df[sensitivity_df["ensemble_name"] == ensemble_name]
    pct_df  = ens_df[ens_df["method"] == "percentile"].copy()
    sig3_df = ens_df[ens_df["method"] == "sigma3"].copy()

    if pct_df.empty:
        print(f"  No sensitivity data for {ensemble_name} — skipping F1 plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    signals = ["unc_mean", "unc_max", "entropy"]

    for signal in signals:
        sig_pct = pct_df[pct_df["signal"] == signal].sort_values("percentile")
        if sig_pct.empty:
            continue

        color = SIGNAL_COLORS[signal]
        ax.plot(sig_pct["percentile"], sig_pct["f1_uncertain"],
                color=color, linewidth=2, label=signal)

        # Dotted vertical at best percentile for this signal
        best_row = sig_pct.loc[sig_pct["f1_uncertain"].idxmax()]
        ax.axvline(best_row["percentile"], color=color, linestyle=":",
                   linewidth=1.2, alpha=0.7)

        # Star marker for σ3 F1
        sig_sigma = sig3_df[sig3_df["signal"] == signal]
        if not sig_sigma.empty:
            f1_sigma = float(sig_sigma["f1_uncertain"].iloc[0])
            ax.scatter([], [], marker="*", color=color, s=150, zorder=5,
                       label=f"{signal} 3σ: F1={f1_sigma:.3f}")
            # Draw as a horizontal dotted line spanning the plot width
            ax.axhline(f1_sigma, color=color, linestyle=(0, (3, 6)),
                       linewidth=1.0, alpha=0.5)

    # Default percentile reference
    ax.axvline(config.UNCERTAINTY_PERCENTILE, color="black", linestyle="--",
               linewidth=1.8, label=f"Default ({config.UNCERTAINTY_PERCENTILE}th pct)")

    ax.set_xlabel("Percentile threshold (calibrated on CV set)", fontsize=11)
    ax.set_ylabel("F1 (uncertain class)", fontsize=11)
    ax.set_title(f"Threshold Sensitivity — F1 Uncertain: {ensemble_name}\n"
                 "Dotted vertical = best pct per signal  |  Dashed horizontal = μ+3σ F1",
                 fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.set_xlim(90.0, 99.9)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_threshold_sensitivity_f1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Sensitivity F1 plot saved: {out_path}")
    return out_path


# =============================================================================
# THRESHOLD SENSITIVITY — recall vs % flagged  (operating-point curve)
# =============================================================================

def plot_threshold_sensitivity_recall_vs_flagged(ensemble_name, sensitivity_df):
    """
    Operating-point curve: recall_uncertain (y) vs pct_flagged (x).

    Each point on a line corresponds to one percentile threshold.  The curve
    moves from the bottom-left (high percentile → few flagged, low recall) to
    the top-right (low percentile → many flagged, high recall).

    Markers highlight the default 95th-pct and the μ+3σ operating points.

    This is the clinically intuitive view: how much expert workload (x-axis)
    do we need to catch a given fraction of the contested cases (y-axis)?
    """
    _ensure_dir()

    ens_df  = sensitivity_df[sensitivity_df["ensemble_name"] == ensemble_name]
    pct_df  = ens_df[ens_df["method"] == "percentile"].copy()
    sig3_df = ens_df[ens_df["method"] == "sigma3"].copy()

    if pct_df.empty:
        print(f"  No sensitivity data for {ensemble_name} — skipping recall-vs-flagged plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    signals = ["unc_mean", "unc_max", "entropy"]

    for signal in signals:
        sig_pct = pct_df[pct_df["signal"] == signal].sort_values("pct_flagged")
        if sig_pct.empty:
            continue

        color = SIGNAL_COLORS[signal]
        ax.plot(sig_pct["pct_flagged"], sig_pct["recall_uncertain"],
                color=color, linewidth=2, label=signal, alpha=0.9)

        # Default 95th-pct operating point
        default_row = sig_pct[sig_pct["percentile"].round(1) == float(config.UNCERTAINTY_PERCENTILE)]
        if not default_row.empty:
            r = default_row.iloc[0]
            ax.scatter(r["pct_flagged"], r["recall_uncertain"],
                       color=color, marker="o", s=80, zorder=6,
                       label=f"{signal} {config.UNCERTAINTY_PERCENTILE}th pct")

        # μ + 3σ operating point
        sig_sigma = sig3_df[sig3_df["signal"] == signal]
        if not sig_sigma.empty:
            r = sig_sigma.iloc[0]
            ax.scatter(r["pct_flagged"], r["recall_uncertain"],
                       color=color, marker="*", s=160, zorder=7,
                       label=f"{signal} μ+3σ")

    ax.set_xlabel("Flagged samples (%)", fontsize=11)
    ax.set_ylabel("Recall — uncertain class", fontsize=11)
    ax.set_title(f"Recall vs Expert Workload: {ensemble_name}\n"
                 "Circle = default 95th pct  |  Star = μ+3σ",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_threshold_sensitivity_recall_vs_flagged.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Recall vs flagged plot saved: {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    _ensure_dir()

    # ------------------------------------------------------------------
    # Expert agreement labels
    # ------------------------------------------------------------------
    expert_labels_path = config.ENSEMBLES_DIR / "expert_agreement_labels.npy"
    if not expert_labels_path.exists():
        print(f"Missing: {expert_labels_path}. Run ensemble.py first.")
        return
    expert_agreement_labels = np.load(expert_labels_path)
    print(f"Expert labels: {len(expert_agreement_labels)} samples")

    # ------------------------------------------------------------------
    # Per-signal thresholds (default 95th pct, stored by ensemble.py)
    # Fall back to zeros if file is absent.
    # ------------------------------------------------------------------
    thresholds_path = config.ENSEMBLES_DIR / "uq_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            all_thresholds = json.load(f)
    else:
        all_thresholds = {}

    # ------------------------------------------------------------------
    # Load all per-ensemble uncertainty arrays
    # ------------------------------------------------------------------
    uncertainties: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(config.ENSEMBLES_DIR.glob("*_uncertainty.npz")):
        name = npz_path.stem.replace("_uncertainty", "")
        data = np.load(npz_path)
        if "unc_max" not in data:
            continue
        uncertainties[name] = {
            "unc_mean": data["unc_mean"],
            "unc_max":  data["unc_max"],
            "entropy":  data["entropy"],
        }

    ensemble_names = list(uncertainties.keys())
    if not ensemble_names:
        print("No uncertainty .npz files found. Run ensemble.py first.")
        return

    # ------------------------------------------------------------------
    # Ensemble-level comparison plots  (unchanged)
    # ------------------------------------------------------------------
    for sig in ["unc_max", "entropy"]:
        plot_uncertainty_boxplot(ensemble_names, uncertainties,
                                 expert_agreement_labels, signal_key=sig)
        plot_roc_curves(ensemble_names, uncertainties,
                        expert_agreement_labels, signal_key=sig)

    # ------------------------------------------------------------------
    # Per-ensemble histograms + sigma analysis  (unchanged)
    # ------------------------------------------------------------------
    for name, sigs in uncertainties.items():
        ens_thr = all_thresholds.get(name, {"unc_mean": 0.0, "unc_max": 0.0, "entropy": 0.0})
        for sig in ["unc_mean", "unc_max", "entropy"]:
            plot_uncertainty_histogram(
                name, sigs[sig], expert_agreement_labels,
                ens_thr.get(sig, 0.0), signal_label=sig)

        plot_sigma_analysis(name, sigs["unc_max"],
                            expert_agreement_labels, signal_label="unc_max")

    # ------------------------------------------------------------------
    # Per-model reliability diagrams & confidence histograms  (unchanged)
    # ------------------------------------------------------------------
    print("\n--- Reliability Diagrams & Confidence Histograms ---")
    for probs_path in sorted(config.INDIVIDUAL_MODELS_DIR.glob("*_test_probs.npz")):
        model_name = probs_path.stem.replace("_test_probs", "")
        data = np.load(probs_path)
        if "y_probs" not in data or "y_true" not in data:
            continue
        plot_reliability_diagram(model_name, data["y_probs"], data["y_true"])
        plot_confidence_histogram(model_name, data["y_probs"])

    # ------------------------------------------------------------------
    # Threshold sensitivity plots
    # ------------------------------------------------------------------
    print("\n--- Threshold Sensitivity Plots ---")
    sensitivity_path = config.ENSEMBLES_DIR / "threshold_sensitivity_all.csv"
    if sensitivity_path.exists():
        sensitivity_df = pd.read_csv(sensitivity_path)

        # Generate for every ensemble; Mega_Ensemble is the primary figure
        for ens_name in sensitivity_df["ensemble_name"].unique():
            plot_threshold_sensitivity_f1(ens_name, sensitivity_df)
            plot_threshold_sensitivity_recall_vs_flagged(ens_name, sensitivity_df)
    else:
        print(f"  {sensitivity_path} not found — skipping sensitivity plots.")

    # ------------------------------------------------------------------
    # KL breakdown for all three threshold methods
    # ------------------------------------------------------------------
    print("\n--- KL Breakdown & Consensus Plots ---")

    threshold_variants = [
        (
            "consensus_default_95pct.csv",
            f"kl_breakdown_default_{config.UNCERTAINTY_PERCENTILE}pct.png",
            f"KL Class Breakdown — Default {config.UNCERTAINTY_PERCENTILE}th Percentile",
            f"consensus_default_{config.UNCERTAINTY_PERCENTILE}pct.png",
            f"Consensus Distribution — Default {config.UNCERTAINTY_PERCENTILE}th Percentile",
        ),
        (
            "consensus_sigma3.csv",
            "kl_breakdown_sigma3.png",
            "KL Class Breakdown — μ+3σ Threshold",
            "consensus_sigma3.png",
            "Consensus Distribution — μ+3σ Threshold",
        ),
        (
            "consensus_best_threshold.csv",
            "kl_breakdown_best_threshold.png",
            "KL Class Breakdown — Best F1 Percentile Threshold",
            "consensus_best_threshold.png",
            "Consensus Distribution — Best F1 Percentile Threshold",
        ),
    ]

    loaded_consensus = {}
    for csv_name, kl_out, kl_title, cons_out, cons_title in threshold_variants:
        csv_path = config.ENSEMBLES_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            loaded_consensus[csv_name] = df
            plot_kl_breakdown_of_flagged(df, out_name=kl_out, title=kl_title)
            plot_consensus_histogram(df, out_name=cons_out, title=cons_title)
        else:
            print(f"  {csv_path} not found — skipping.")

    # Side-by-side comparison: 3σ vs best threshold
    sigma3_df = loaded_consensus.get("consensus_sigma3.csv")
    best_df   = loaded_consensus.get("consensus_best_threshold.csv")
    if sigma3_df is not None and best_df is not None:
        plot_kl_breakdown_comparison(sigma3_df, best_df)

    # Legacy fallback: if only the Excel file exists (pre-sensitivity run)
    if not any(not df.empty for df in loaded_consensus.values()):
        excel_path = config.ENSEMBLES_DIR / "MASTER_RESULTS_SUMMARY.xlsx"
        if excel_path.exists():
            try:
                consensus_df = pd.read_excel(excel_path, sheet_name="Consensus_95pct")
                plot_kl_breakdown_of_flagged(
                    consensus_df,
                    out_name=f"kl_breakdown_default_{config.UNCERTAINTY_PERCENTILE}pct.png",
                    title=f"KL Class Breakdown — Default {config.UNCERTAINTY_PERCENTILE}th Percentile",
                )
                plot_consensus_histogram(
                    consensus_df,
                    out_name=f"consensus_default_{config.UNCERTAINTY_PERCENTILE}pct.png",
                    title=f"Consensus Distribution — Default {config.UNCERTAINTY_PERCENTILE}th Percentile",
                )
            except Exception as e:
                print(f"  Could not read Excel consensus sheet: {e}")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
