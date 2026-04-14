import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score

import config


# =============================================================================
# PĘTLA JEDNEJ EPOKI — TRENING
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total      = 0
    all_labels = []
    all_preds  = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        logits = model(images)

        loss = criterion(logits, labels)

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)  # loss * batch_size

        _, predicted = torch.max(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        total += labels.size(0)

    avg_loss     = total_loss / total
    bal_accuracy = balanced_accuracy_score(all_labels, all_preds)
    f1_macro     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, bal_accuracy, f1_macro


# =============================================================================
# PĘTLA EWALUACJI — VAL lub TEST
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total      = 0
    all_labels = []
    all_preds  = []

    for images, labels in loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        total += labels.size(0)

    avg_loss     = total_loss / total
    bal_accuracy = balanced_accuracy_score(all_labels, all_preds)
    f1_macro     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, bal_accuracy, f1_macro


# =============================================================================
# TRENING KOMPLETNEGO MODELU
# =============================================================================

def train_model(model_name, model,train_loader,val_loader,class_weights,):

    print("\n" + "=" * 60)
    print(f"TRENING: {model_name}")
    print(f"Urządzenie: {config.DEVICE}")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(config.DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )


    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
    log_path        = config.RESULTS_DIR / f"{model_name}_training_log.csv"


    history = {
        "train_loss":    [], "train_bal_acc": [], "train_f1": [],
        "val_loss":      [], "val_bal_acc":   [], "val_f1":   [],
    }

    best_val_loss    = float("inf")
    epochs_no_improve = 0  # Licznik do early stopping

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bal_acc", "train_f1", "val_loss", "val_bal_acc", "val_f1", "lr"])

        # =====================================================================
        # GŁÓWNA PĘTLA TRENINGOWA
        # =====================================================================
        for epoch in range(1, config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            # --- Trening ---
            train_loss, train_bal_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer
            )

            # --- Walidacja ---
            val_loss, val_bal_acc, val_f1 = evaluate(
                model, val_loader, criterion
            )

            # --- Aktualizuj scheduler ---
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # --- Zapisz do historii ---
            history["train_loss"].append(train_loss)
            history["train_bal_acc"].append(train_bal_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_bal_acc"].append(val_bal_acc)
            history["val_f1"].append(val_f1)

            # --- Zaloguj do CSV ---
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",    f"{train_bal_acc:.4f}", f"{train_f1:.4f}",
                f"{val_loss:.4f}",      f"{val_bal_acc:.4f}",   f"{val_f1:.4f}",
                f"{current_lr:.6f}",
            ])
            f.flush()  # Zapisz od razu (nie czekaj na zamknięcie pliku)

            # --- Print do konsoli ---
            elapsed = time.time() - epoch_start
            print(
                f"Epoka [{epoch:3d}/{config.NUM_EPOCHS}]  "
                f"Train Loss: {train_loss:.4f}  Bal.Acc: {train_bal_acc*100:.1f}%  F1: {train_f1:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Bal.Acc: {val_bal_acc*100:.1f}%  F1: {val_f1:.4f}  |  "
                f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # --- Checkpoint: zapisz jeśli najlepsza walidacja ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                # Zapisz pełny stan modelu
                torch.save({
                    "epoch":      epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":     val_loss,
                    "val_bal_acc":  val_bal_acc,
                    "val_f1":       val_f1,
                }, checkpoint_path)

                print(f"  ✓ Zapisano najlepszy checkpoint (val_loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # --- Early stopping ---
            if epochs_no_improve >= config.PATIENCE:
                print(f"\n  Early stopping: brak poprawy przez {config.PATIENCE} epok.")
                print(f"  Najlepsza val_loss: {best_val_loss:.4f}")
                break

    print(f"\nTrening zakończony. Checkpoint: {checkpoint_path}")
    print(f"Log CSV: {log_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Załadowano najlepsze wagi z epoki {checkpoint['epoch']}")

    return history
