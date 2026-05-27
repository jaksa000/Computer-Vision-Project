import gc
import json

import numpy as np
import torch

import config
from dataset import load_all_samples, build_fold_dataloaders, split_holdout, build_test_dataloader
from models import build_model
from train import train_model
from evaluate import (
    evaluate_model,
    print_summary_table,
    get_predictions,
    compute_metrics,
    compute_calibration_metrics,
)


def print_cv_summary(cv_results):
    print("\n" + "=" * 125)
    print(" CROSS-VALIDATION SUMMARY — AVERAGE OUT OF 5 FOLDs")
    print("=" * 125)
    print(
        f"{'Model':<20} {'Kappa':>14} {'F1-Mac':>14} {'MAE':>10} {'Off-1':>10} | "
        f"{'KL0':>8} {'KL1':>8} {'KL2':>8} {'KL3':>8} {'KL4':>8}"
    )
    print("-" * 125)

    for model_name, folds in cv_results.items():
        kappas           = [m["cohen_kappa_Quadratic"] for m in folds]
        f1s              = [m["f1_macro"] for m in folds]
        maes             = [m.get("mae_ordinal", float("nan")) for m in folds]
        off1s            = [m.get("off_by_one_accuracy", float("nan")) for m in folds]
        f1_classes_array = np.array([m["f1_per_class"] for m in folds])
        mean_f1_classes  = np.mean(f1_classes_array, axis=0)
        print(
            f"{model_name:<20} "
            f"{np.mean(kappas):>7.4f} ±{np.std(kappas):>4.4f} "
            f"{np.mean(f1s):>7.4f} ±{np.std(f1s):>4.4f} "
            f"{np.nanmean(maes):>7.4f} ±{np.nanstd(maes):>2.4f} "
            f"{np.nanmean(off1s):>7.4f} ±{np.nanstd(off1s):>2.4f} | "
            f"{mean_f1_classes[0]:>8.4f} {mean_f1_classes[1]:>8.4f} "
            f"{mean_f1_classes[2]:>8.4f} {mean_f1_classes[3]:>8.4f} "
            f"{mean_f1_classes[4]:>8.4f}"
        )
    print("=" * 125)


def main():
    print(f"Device: {config.DEVICE}")
    print(f"Models to run: {[m['name'] for m in config.MODELS_CONFIG]}")
    print(f"Folds number: {config.NUM_FOLDS}")

    all_samples = load_all_samples()
    cv_samples, test_samples = split_holdout(all_samples, save_manifest=True)
    test_loader = build_test_dataloader(test_samples)

    # -------------------------------------------------------------------------
    # Phase 1: 5-fold cross-validation training
    # -------------------------------------------------------------------------
    cv_results = {}
    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]

        print(f"\n{'=' * 80}")
        print(f"MODEL TRAINING START: {model_name}")
        print(f"{'=' * 80}")

        model_metrics = []
        for fold_idx in range(config.NUM_FOLDS):
            print(f"\n--- {model_name} | FOLD {fold_idx + 1}/{config.NUM_FOLDS} ---")
            train_loader, val_loader, class_weights = build_fold_dataloaders(
                cv_samples, fold_idx
            )
            run_name = f"{model_name}_fold{fold_idx + 1}"
            model    = build_model(model_cfg)

            history = train_model(
                model_name=run_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
            )

            metrics = evaluate_model(
                model_name=run_name,
                model=model,
                val_loader=val_loader,
                history=history,
            )

            model_metrics.append(metrics)
            del model
            torch.cuda.empty_cache()
            gc.collect()

        cv_results[model_name] = model_metrics
        kappas = [m["cohen_kappa_Quadratic"] for m in model_metrics]
        print(
            f"\n FINISHED: {model_name}. "
            f"Average kappa out of 5 folds: {np.mean(kappas):.4f} ±{np.std(kappas):.4f}"
        )

    all_metrics = [m for folds in cv_results.values() for m in folds]
    print_summary_table(all_metrics)
    print_cv_summary(cv_results)

    # -------------------------------------------------------------------------
    # Phase 2: Best-fold holdout evaluation — one forward pass per architecture.
    # For each architecture, the fold with the highest CV Kappa is evaluated on
    # the held-out test set.  Results are saved as
    #   results/individual_models/{model}_best_fold_holdout_metrics.json
    # so that ensemble.py can build a baseline comparison table without
    # requiring models to be re-run.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" BEST-FOLD HOLDOUT EVALUATION — Individual Models")
    print("=" * 80)

    config.INDIVIDUAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_cfg in config.MODELS_CONFIG:
        model_name   = model_cfg["name"]
        fold_metrics = cv_results.get(model_name, [])
        if not fold_metrics:
            continue

        best_fold_idx = int(np.argmax([m["cohen_kappa_Quadratic"] for m in fold_metrics]))
        ckpt_path     = config.CHECKPOINTS_DIR / f"{model_name}_fold{best_fold_idx + 1}_best.pt"

        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found — {ckpt_path.name}")
            continue

        print(f"\n  {model_name}  (best CV fold: {best_fold_idx + 1})")
        model = build_model(model_cfg)
        ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        y_true, y_pred, y_probs = get_predictions(model, test_loader)
        metrics = compute_metrics(y_true, y_pred)
        metrics.update(compute_calibration_metrics(y_true, y_probs))
        metrics["model_name"] = model_name
        metrics["fold_used"]  = best_fold_idx + 1

        print(f"  Holdout Kappa: {metrics['cohen_kappa_Quadratic']:.4f}  "
              f"F1: {metrics['f1_macro']:.4f}  "
              f"MAE: {metrics['mae_ordinal']:.4f}  "
              f"Off-1: {metrics['off_by_one_accuracy']:.4f}  "
              f"ECE: {metrics['ece']:.4f}")

        json_path = config.INDIVIDUAL_MODELS_DIR / f"{model_name}_best_fold_holdout_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {json_path.name}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\nResults saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
