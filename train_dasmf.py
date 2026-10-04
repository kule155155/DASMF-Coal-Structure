"""Public DASMF training demonstration using the synthetic folder dataset.

The synthetic images check that the pipeline runs. They cannot reproduce the
field-well performance reported in the manuscript.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dasmf import DASMFResNet18, make_criterion


CLASS_NAMES = ("0_primary", "1_cataclastic", "2_granulated", "3_mylonitic")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "demo_synthetic_dataset",
        help="Directory containing train, val, and test subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--loss", choices=("ce", "focal", "focal_ordinal"), default="ce",
        help="Cross-entropy is the objective used for the final DASMF comparisons.",
    )
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--ordinal-lambda", type=float, default=0.1)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    # The public demo reads prepared patches. It does not perform the private
    # well/depth split or create the field-data augmentations.
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    loaders = []
    for split in ("train", "val", "test"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {split_dir}. Run 'python make_demo_dataset.py' first."
            )
        dataset = datasets.ImageFolder(split_dir, transform=transform)
        if dataset.class_to_idx != CLASS_TO_INDEX:
            raise ValueError(
                f"Unexpected class folders in {split_dir}: {dataset.class_to_idx}"
            )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                num_workers=args.num_workers,
            )
        )
        print(f"{split}: {len(dataset)} images")
    return tuple(loaders)


def run_epoch(
    model: DASMFResNet18,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        true_labels.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return total_loss / len(loader.dataset), true_labels, predictions


def main() -> None:
    args = parse_args()
    train_loader, val_loader, test_loader = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; training objective: {args.loss}")

    # The supplied DASMFResNet18 class loads ImageNet weights on construction.
    # Its first use may therefore download the weights from PyTorch.
    model = DASMFResNet18().to(device)
    criterion = make_criterion(
        args.loss, gamma=args.gamma, ordinal_lambda=args.ordinal_lambda
    )
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_true, val_pred = run_epoch(
            model, val_loader, criterion, device
        )
        train_f1 = f1_score(
            train_true, train_pred, labels=range(4), average="macro", zero_division=0
        )
        val_f1 = f1_score(
            val_true, val_pred, labels=range(4), average="macro", zero_division=0
        )
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, Macro-F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, Macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print("Early stopping: validation Macro-F1 did not improve.")
                break

    if best_weights is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "DASMF_best.pth"
    torch.save(best_weights, checkpoint_path)
    print(f"Best validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Saved checkpoint: {checkpoint_path}")

    test_loss, test_true, test_pred = run_epoch(
        model, test_loader, criterion, device
    )
    print("\nSynthetic demonstration test results (not field-data results):")
    print(f"Loss: {test_loss:.4f}")
    print(f"Correct: {sum(a == b for a, b in zip(test_true, test_pred))}/{len(test_true)}")
    print(f"Accuracy: {accuracy_score(test_true, test_pred):.4f}")
    print(
        f"Macro-Precision: {precision_score(test_true, test_pred, labels=range(4), average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro-Recall: {recall_score(test_true, test_pred, labels=range(4), average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro-F1: {f1_score(test_true, test_pred, labels=range(4), average='macro', zero_division=0):.4f}"
    )
    print(
        classification_report(
            test_true, test_pred, labels=range(4),
            target_names=CLASS_NAMES, zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_true, test_pred, labels=range(4)))


if __name__ == "__main__":
    main()

