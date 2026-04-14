import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
)
from torch.utils.data import DataLoader

import config


# =============================================================================
# ZBIERZ PREDYKCJE
# =============================================================================

@torch.no_grad()
def get_predictions(model,loader,):
    model.eval()

    all_labels = []
    all_preds  = []
    all_probs  = []
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


# =============================================================================
# OBLICZ METRYKI
# =============================================================================

def compute_metrics(y_true, y_pred):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_m, 4),
        "cohen_kappa_Quadratic": round(kappa, 4),
        "f1_per_class": [round(f, 4) for f in f1_per_class],
    }

    return metrics


# =============================================================================
# GŁÓWNA FUNKCJA EWALUACJI
# =============================================================================

def evaluate_model(model_name,model,test_loader,history,save_dir=config.RESULTS_DIR,):
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEwaluacja modelu: {model_name}")
    print("-" * 40)


    y_true, y_pred, y_probs = get_predictions(model, test_loader)

    metrics = compute_metrics(y_true, y_pred)

    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  Quadratic Cohen's Kappa: {metrics['cohen_kappa_Quadratic']:.4f}")

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
    print(f"  Metryki zapisane: {json_path}")


    probs_path = save_dir / f"{model_name}_test_probs.npz"
    np.savez(
        probs_path,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
    )
    print(f"  Prawdopodobieństwa zapisane: {probs_path}")

    return metrics


# =============================================================================
# PODSUMOWANIE WSZYSTKICH MODELI
# =============================================================================

def print_summary_table(all_metrics: list[dict]) -> None:
    print("\n" + "=" * 105)
    print("PODSUMOWANIE POJEDYNCZYCH FOLDÓW")
    print("=" * 105)
    print(f"{'Model':<25} {'Kappa':>8} {'F1-Mac':>8} | {'KL0':>8} {'KL1':>8} {'KL2':>8} {'KL3':>8} {'KL4':>8}")
    print("-" * 105)

    for m in sorted(all_metrics, key=lambda x: x["cohen_kappa_Quadratic"], reverse=True):
        f1_c = m["f1_per_class"]
        print(
            f"{m['model_name']:<25} "
            f"{m['cohen_kappa_Quadratic']:>8.4f} "
            f"{m['f1_macro']:>8.4f} | "
            f"{f1_c[0]:>8.4f} {f1_c[1]:>8.4f} {f1_c[2]:>8.4f} {f1_c[3]:>8.4f} {f1_c[4]:>8.4f}"
        )
    print("=" * 105)
    print("Posortowane wg Cohen's Kappa ")
