import gc

import numpy as np
import torch

import config
from dataset import load_all_samples, build_fold_dataloaders, split_holdout, build_test_dataloader
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

    # --- Odcinamy sejf (hold-out) ---
    cv_samples, test_samples = split_holdout(all_samples)
    test_loader = build_test_dataloader(test_samples)

    cv_results = {}

    # =========================================================================
    # PĘTLA ZEWNĘTRZNA: MODELE (Tak jak chciałeś!)
    # =========================================================================
    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]

        print(f"\n{'=' * 80}")
        print(f"🚀 ROZPOCZĘCIE TRENINGU MODELU: {model_name}")
        print(f"{'=' * 80}")

        model_metrics = []

        # =====================================================================
        # PĘTLA WEWNĘTRZNA: FOLDY DLA DANEGO MODELU
        # =====================================================================
        for fold_idx in range(config.NUM_FOLDS):
            print(f"\n--- {model_name} | FOLD {fold_idx + 1}/{config.NUM_FOLDS} ---")

            # Generujemy split na nowo (jest bezpieczny i identyczny dzięki RANDOM_SEED)
            train_loader, val_loader, class_weights = build_fold_dataloaders(
                cv_samples, fold_idx
            )

            run_name = f"{model_name}_fold{fold_idx + 1}"
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
                test_loader=val_loader,
                history=history,
            )

            model_metrics.append(metrics)

            # Czyszczenie pamięci karty graficznej po tym konkretnym foldzie
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Po zakończeniu 5 foldów, zapisujemy wyniki tego modelu
        cv_results[model_name] = model_metrics

        # Opcjonalnie: Szybkie wyświetlenie średniej tylko dla TEGO modelu od razu!
        kappas = [m["cohen_kappa_Quadratic"] for m in model_metrics]
        print(f"\n✅ ZAKOŃCZONO: {model_name}. Średnia Kappa z 5 foldów: {np.mean(kappas):.4f} ±{np.std(kappas):.4f}")

    # Po przeliczeniu wszystkich modeli wyświetlamy tabele podsumowujące
    all_metrics = [m for folds in cv_results.values() for m in folds]
    print_summary_table(all_metrics)
    print_cv_summary(cv_results)

    print("Wyniki w folderze:", config.RESULTS_DIR)

if __name__ == "__main__":
    main()
