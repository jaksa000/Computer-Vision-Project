"""
visualize.py — wykresy do analizy UQ

Generuje:
  1. Histogram unc(x): certain vs uncertain (expert labels)
  2. Box plot unc(x): certain vs uncertain dla każdego ensembla
  3. Krzywa ROC dla detekcji uncertain
  4. Wykres wpływu multiplikatora σ (1σ, 2σ, 3σ) na F1 uncertain

Użycie:
    python visualize.py
"""

from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, f1_score

import config


FIGURES_DIR = config.RESULTS_DIR / "figures"

COLORS = {
    "certain":   "#2196F3",
    "uncertain": "#F44336",
    "neutral":   "#90CAF9",
}


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. HISTOGRAM — rozkład unc(x): certain vs uncertain
# =============================================================================

def plot_uncertainty_histogram(
    ensemble_name: str,
    uncertainty_scores: np.ndarray,
    expert_agreement_labels: np.ndarray,
    threshold: float,
) -> Path:
    """
    Nakładkowy histogram unc(x) dla grupy certain i uncertain.
    Rysuje pionową linię progu 3σ.
    """
    _ensure_dir()

    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        uncertainty_scores[mask_c],
        bins=40, alpha=0.6, color=COLORS["certain"],
        label=f"Certain (n={mask_c.sum()})", density=True,
    )
    ax.hist(
        uncertainty_scores[mask_u],
        bins=40, alpha=0.6, color=COLORS["uncertain"],
        label=f"Uncertain (n={mask_u.sum()})", density=True,
    )
    ax.axvline(
        threshold, color="black", linestyle="--", linewidth=1.5,
        label=f"Próg {config.UNCERTAINTY_SIGMA_MULTIPLIER}σ = {threshold:.4f}",
    )

    ax.set_xlabel("Niepewność unc(x) = mean std prawdopodobieństw ensembla", fontsize=11)
    ax.set_ylabel("Gęstość", fontsize=11)
    ax.set_title(f"Rozkład niepewności: {ensemble_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"{ensemble_name}_uncertainty_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram zapisany: {out_path}")
    return out_path


# =============================================================================
# 2. BOX PLOT — porównanie unc(x) dla wszystkich ensembli
# =============================================================================

def plot_uncertainty_boxplot(
    ensemble_names: list[str],
    uncertainties_dict: dict[str, np.ndarray],
    expert_agreement_labels: np.ndarray,
) -> Path:
    """
    Box plot pokazujący unc(x) osobno dla grupy certain i uncertain
    dla każdego ensembla — jeden wykres porównawczy.
    """
    _ensure_dir()

    mask_c = expert_agreement_labels == config.CERTAIN_LABEL
    mask_u = expert_agreement_labels == config.UNCERTAIN_LABEL

    n = len(ensemble_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 6))

    positions_c = [i * 3     for i in range(n)]
    positions_u = [i * 3 + 1 for i in range(n)]

    data_c = [uncertainties_dict[name][mask_c] for name in ensemble_names]
    data_u = [uncertainties_dict[name][mask_u] for name in ensemble_names]

    bp_c = ax.boxplot(data_c, positions=positions_c, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor=COLORS["certain"], alpha=0.7))
    bp_u = ax.boxplot(data_u, positions=positions_u, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor=COLORS["uncertain"], alpha=0.7))

    tick_positions = [(positions_c[i] + positions_u[i]) / 2 for i in range(n)]
    short_names = [name.replace("_Homogeneous", "\nHom.").replace("_", "\n") for name in ensemble_names]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(short_names, fontsize=9)

    ax.set_ylabel("unc(x)", fontsize=11)
    ax.set_title("Niepewność ensembla: certain vs uncertain (eksperci)", fontsize=12)

    patch_c = mpatches.Patch(color=COLORS["certain"],   label="Certain (eksperci zgodni)")
    patch_u = mpatches.Patch(color=COLORS["uncertain"], label="Uncertain (eksperci różni)")
    ax.legend(handles=[patch_c, patch_u], fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "all_ensembles_uncertainty_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Box plot zapisany: {out_path}")
    return out_path


# =============================================================================
# 3. KRZYWA ROC — detekcja uncertain
# =============================================================================

def plot_roc_curves(
    ensemble_names: list[str],
    uncertainties_dict: dict[str, np.ndarray],
    expert_agreement_labels: np.ndarray,
) -> Path:
    """
    Krzywe ROC dla każdego ensembla — unc(x) jako score, expert-uncertain jako GT.
    """
    from sklearn.metrics import roc_auc_score
    _ensure_dir()

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab10", len(ensemble_names))

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

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Losowy klasyfikator")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC: detekcja uncertain (walidacja przez ekspertów)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "roc_uncertain_detection.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Krzywa ROC zapisana: {out_path}")
    return out_path


# =============================================================================
# 4. ANALIZA PROGU σ — wpływ multiplikatora na F1 uncertain
# =============================================================================

def plot_sigma_analysis(
    ensemble_name: str,
    uncertainty_scores: np.ndarray,
    expert_agreement_labels: np.ndarray,
    sigma_range: np.ndarray | None = None,
) -> Path:
    """
    Pokazuje jak zmiana multiplikatora σ wpływa na F1 uncertain i odsetek flagowanych.
    Pomaga uzasadnić wybór 3σ.
    """
    _ensure_dir()

    if sigma_range is None:
        sigma_range = np.arange(0.5, 5.1, 0.25)

    mean_unc = np.mean(uncertainty_scores)
    std_unc  = np.std(uncertainty_scores)

    f1_scores  = []
    pct_flagged = []

    for sigma in sigma_range:
        threshold = mean_unc + sigma * std_unc
        flags = (uncertainty_scores > threshold).astype(int)
        f1 = f1_score(expert_agreement_labels, flags,
                      pos_label=config.UNCERTAIN_LABEL, zero_division=0)
        f1_scores.append(f1)
        pct_flagged.append(100 * flags.mean())

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(sigma_range, f1_scores, color="#E91E63", linewidth=2, label="F1 (uncertain)")
    ax2.plot(sigma_range, pct_flagged, color="#3F51B5", linewidth=2,
             linestyle="--", label="% flagowanych")

    ax1.axvline(config.UNCERTAINTY_SIGMA_MULTIPLIER, color="black", linestyle=":",
                linewidth=1.5, label=f"Wybrany: {config.UNCERTAINTY_SIGMA_MULTIPLIER}σ")

    ax1.set_xlabel("Multiplikator σ", fontsize=11)
    ax1.set_ylabel("F1 (uncertain)", fontsize=11, color="#E91E63")
    ax2.set_ylabel("% próbek flagowanych", fontsize=11, color="#3F51B5")
    ax1.set_title(f"Wpływ progu σ na detekcję uncertain — {ensemble_name}", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = FIGURES_DIR / f"{ensemble_name}_sigma_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Analiza σ zapisana: {out_path}")
    return out_path


# =============================================================================
# SKRYPT GŁÓWNY — generuj wszystkie wykresy z zapisanych plików .npz
# =============================================================================

def main():
    _ensure_dir()

    # Wczytaj etykiety ekspertów z hold-outu
    # Zakładamy, że ensemble.py zapisał je jako expert_labels.npy
    expert_labels_path = config.RESULTS_DIR / "expert_agreement_labels.npy"
    if not expert_labels_path.exists():
        print(f"Brak pliku {expert_labels_path}.")
        print("Uruchom najpierw ensemble.py, który zapisuje etykiety ekspertów.")
        return

    expert_agreement_labels = np.load(expert_labels_path)
    print(f"Wczytano {len(expert_agreement_labels)} etykiet ekspertów.")
    print(f"  Certain:   {(expert_agreement_labels == config.CERTAIN_LABEL).sum()}")
    print(f"  Uncertain: {(expert_agreement_labels == config.UNCERTAIN_LABEL).sum()}")

    # Wczytaj niepewności dla każdego ensembla
    uncertainties: dict[str, np.ndarray] = {}
    for npz_path in sorted(config.RESULTS_DIR.glob("*_uncertainty.npz")):
        ensemble_name = npz_path.stem.replace("_uncertainty", "")
        data = np.load(npz_path)
        uncertainties[ensemble_name] = data["uncertainty"]
        print(f"  Wczytano unc(x) dla: {ensemble_name} ({len(data['uncertainty'])} próbek)")

    if not uncertainties:
        print("Brak plików *_uncertainty.npz. Uruchom ensemble.py.")
        return

    ensemble_names = list(uncertainties.keys())

    # --- Zbiorczy box plot ---
    plot_uncertainty_boxplot(ensemble_names, uncertainties, expert_agreement_labels)

    # --- Zbiorczy ROC ---
    plot_roc_curves(ensemble_names, uncertainties, expert_agreement_labels)

    # --- Per-ensemble: histogram + analiza σ ---
    for name, unc in uncertainties.items():
        # Wczytaj próg z JSON
        uq_json = config.RESULTS_DIR / f"{name}_uq_detection.json"
        if uq_json.exists():
            with open(uq_json) as f:
                threshold = json.load(f)["threshold"]
        else:
            mean_unc = np.mean(unc)
            std_unc  = np.std(unc)
            threshold = mean_unc + config.UNCERTAINTY_SIGMA_MULTIPLIER * std_unc

        plot_uncertainty_histogram(name, unc, expert_agreement_labels, threshold)
        plot_sigma_analysis(name, unc, expert_agreement_labels)

    print(f"\nWszystkie wykresy zapisane w: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
