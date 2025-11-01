import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils import (
    DualImageResNet18Gray,
    DualImageResNet34Gray,
    DualImageResNet50Gray,
    MobileNetV2Modified,
    classifier_dataloader_cropped,
)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Return an instantiated model for the requested architecture."""
    if model_name == "mobilenet":
        width_mult = kwargs.get("width_mult", 1.0)
        return MobileNetV2Modified(num_classes=3, width_mult=width_mult)
    if model_name == "resnet18":
        return DualImageResNet18Gray(num_classes=3)
    if model_name == "resnet34":
        return DualImageResNet34Gray(num_classes=3)
    if model_name == "resnet50":
        return DualImageResNet50Gray(num_classes=3)
    raise ValueError(f"Unsupported model: {model_name}")


def _get_optimizer_and_scheduler(model: nn.Module, model_name: str) -> Tuple[optim.Optimizer, object]:
    """Configure optimizer and scheduler for the chosen model."""
    import torch.optim.lr_scheduler as lr_scheduler

    if model_name == "mobilenet":
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0002)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=2e-06)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    return optimizer, scheduler


def unified_train(args) -> None:
    """Unified training loop for different model architectures."""
    model_name = args.model
    batch_size = args.batch_size

    torch.cuda.empty_cache()

    train_loader, validation_loader = classifier_dataloader_cropped(batch_size, True)

    output_dir = f"./{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on: {str(device)}")
    print(f"Output directory: {output_dir}")
    print(f"Resize images: {args.resize}")

    model = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _get_optimizer_and_scheduler(model, model_name)

    model.to(device)

    scaler = GradScaler()

    best_val_loss = float("inf")
    early_stop_patience = args.early_stop
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Training {model_name} (LR: {current_lr:.2e})",
            unit="batch",
        )

        for image0s, image1s, labels in train_loader_tqdm:
            image0s = image0s.float().to(device)
            image1s = image1s.float().to(device)
            labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64).to(
                device
            )

            with autocast():
                output = model(image0s, image1s)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.empty_cache()

            running_loss += loss.item() * image0s.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * train_correct / train_total:.2f}%"}
            )

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        preds_list = []
        labels_list = []

        with torch.no_grad():
            validation_loader_tqdm = tqdm(
                validation_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
                unit="batch",
            )

            for image0s, image1s, labels in validation_loader_tqdm:
                image0s = image0s.float().to(device)
                image1s = image1s.float().to(device)
                labels = torch.tensor([label.to(device) for label in labels], dtype=torch.int64).to(
                    device
                )

                output = model(image0s, image1s)
                loss = criterion(output, labels)

                val_loss += loss.item() * image0s.size(0)
                _, predicted = torch.max(output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                preds_list.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss / len(validation_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if model_name == "mobilenet":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            print(
                f"\nValidation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model..."
            )
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                f"{output_dir}/{model_name}_best.pth",
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered!")
            break

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

    total_training_time = time.time() - total_start_time
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory = 0.0

    print(f"\n=== Training Summary ===")
    print(f"Total training time: {total_training_time:.2f} sec")
    if device.type == "cuda":
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output saved to: {output_dir}")
