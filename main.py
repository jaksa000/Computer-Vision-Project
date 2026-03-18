import gc

import torch

import config
from dataset import build_dataloaders
from models  import build_model
from train   import train_model
from evaluate import evaluate_model, print_summary_table


def main():
    print(f"Urządzenie: {config.DEVICE}")
    print(f"Modele do uruchomienia: {[m['name'] for m in config.MODELS_CONFIG]}")

    train_loader, val_loader, test_loader, class_weights = build_dataloaders()
    all_metrics = []

    for model_cfg in config.MODELS_CONFIG:
        model_name = model_cfg["name"]

        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model_name}")
        print(f"{'#' * 60}")

        model = build_model(model_cfg)

        history = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
        )

        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            test_loader=test_loader,
            history=history,
        )
        all_metrics.append(metrics)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print_summary_table(all_metrics)
    print("\n✓ Gotowe! Wyniki w folderze:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
