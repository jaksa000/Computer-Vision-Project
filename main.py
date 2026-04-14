import gc

import numpy as np
import torch

import config
from dataset import load_all_samples, build_fold_dataloaders
from models  import build_model
from train   import train_model
from evaluate import evaluate_model, print_summary_table


def print_cv_summary(cv_results: dict) -> None:
    print("\n" + "=" * 115)
    print("PODSUMOWANIE CROSS-VALIDATION — ŚREDNIA Z 5 FOLDÓW")
    print("=" * 115)
    print(f"{'Model':<20} {'Kappa':>14} {'F1-Mac':>14} | {'KL0':>8} {'KL1':>8} {'KL2':>8} {'KL3':>8} {'KL4':>8}")
    print("-" * 115)

    for model_name, folds in cv_results.items():
        kappas = [m["cohen_kappa_Quadratic"] for m in folds]
        f1s = [m["f1_macro"] for m in folds]

        # Wyciągamy tablicę (5 foldów x 5 klas) i liczymy średnią w pionie (dla każdej klasy)
        f1_classes_array = np.array([m["f1_per_class"] for m in folds])
        mean_f1_classes = np.mean(f1_classes_array, axis=0)

        print(
            f"{model_name:<20} "
            f"{np.mean(kappas):>7.4f} ±{np.std(kappas):>4.4f} "
            f"{np.mean(f1s):>7.4f} ±{np.std(f1s):>4.4f} | "
            f"{mean_f1_classes[0]:>8.4f} {mean_f1_classes[1]:>8.4f} {mean_f1_classes[2]:>8.4f} {mean_f1_classes[3]:>8.4f} {mean_f1_classes[4]:>8.4f}"
        )

    print("=" * 115)


def main():
    print(f"Urządzenie: {config.DEVICE}")
    print(f"Modele do uruchomienia: {[m['name'] for m in config.MODELS_CONFIG]}")
    print(f"Liczba foldów: {config.NUM_FOLDS}")

    all_samples = load_all_samples()

    # cv_results[model_name] = [metrics_fold1, metrics_fold2, ...]
    cv_results = {m["name"]: [] for m in config.MODELS_CONFIG}

    for fold_idx in range(config.NUM_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{config.NUM_FOLDS}")
        print(f"{'=' * 60}")

        train_loader, val_loader, class_weights = build_fold_dataloaders(
            all_samples, fold_idx
        )

        for model_cfg in config.MODELS_CONFIG:
            model_name = model_cfg["name"]
            run_name   = f"{model_name}_fold{fold_idx + 1}"

            print(f"\n{'#' * 60}")
            print(f"# MODEL: {model_name}  |  Fold {fold_idx + 1}")
            print(f"{'#' * 60}")

            model = build_model(model_cfg)

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
                test_loader=val_loader,   # brak hold-out: ewaluacja na val folda
                history=history,
            )
            cv_results[model_name].append(metrics)

            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Podsumowanie per-fold (wszystkie runy)
    all_metrics = [m for folds in cv_results.values() for m in folds]
    print_summary_table(all_metrics)

    # Podsumowanie CV ze średnią ± std
    print_cv_summary(cv_results)

    print("\n✓ Gotowe! Wyniki w folderze:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
