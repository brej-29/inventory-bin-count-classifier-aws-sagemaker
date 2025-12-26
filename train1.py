import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


def str2bool(v):
    if isinstance(v, bool):
        return v
    # SageMaker HPO sometimes wraps booleans in quotes like '"false"'
    v = str(v).strip().strip('"').strip("'").lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confusion_matrix_numpy(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    hook=None,
    is_train: bool = True,
    log_every: int = 0,
) -> Dict:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    y_true = []
    y_pred = []

    start = time.time()

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        total_correct += int((preds == labels).sum().item())
        total += int(images.size(0))

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        if log_every and step % log_every == 0:
            logger.info(
                f"{'train' if is_train else 'val'} step={step} "
                f"loss={loss.item():.6f} "
                f"elapsed_sec={time.time() - start:.2f}"
            )

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "y_true": np.array(y_true, dtype=np.int64),
        "y_pred": np.array(y_pred, dtype=np.int64),
    }


def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )
    logger.info(f"✅ Saved checkpoint -> {path}")


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    num_classes: int,
    hook=None,
    early_stop_patience: int = 3,
    checkpoint_dir: Optional[str] = None,
    log_every: int = 0,
) -> nn.Module:
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_one_epoch(
            model, train_loader, criterion, optimizer, device, hook, is_train=True, log_every=log_every
        )
        val_metrics = run_one_epoch(
            model, val_loader, criterion, None, device, hook, is_train=False, log_every=log_every
        )

        cm = confusion_matrix_numpy(val_metrics["y_true"], val_metrics["y_pred"], num_classes=num_classes)
        _, _, f1 = precision_recall_f1_from_cm(cm)
        val_macro_f1 = float(np.mean(f1))

        logger.info(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} "
            f"val_macro_f1={val_macro_f1:.6f} "
            f"elapsed_sec={time.time() - t0:.2f}"
        )

        if checkpoint_dir:
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                logger.info(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def build_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    logger.info(f"Classes (ImageFolder): {train_ds.classes}")
    logger.info(f"Class->index mapping: {train_ds.class_to_idx}")
    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = models.resnet50(pretrained=True)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device, num_classes: int) -> Dict:
    metrics = run_one_epoch(model, loader, criterion, None, device, hook=None, is_train=False)
    cm = confusion_matrix_numpy(metrics["y_true"], metrics["y_pred"], num_classes=num_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)

    return {
        "loss": float(metrics["loss"]),
        "acc": float(metrics["accuracy"]),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "confusion_matrix": cm.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # ✅ MUST accept a value because SageMaker passes "--freeze_backbone False"
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)

    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)

    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--checkpoint_dir", type=str, default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))

    # ✅ Make these robust too (prevents the same “False” issue later)
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--smoke_test", type=str2bool, default=False)

    parser.add_argument("--log_every", type=int, default=0)

    return parser.parse_args()


def main(args):
    logger.info(f"Args: {args}")
    device = get_device()
    logger.info(f"Using device: {device}")

    train_dir = args.train_dir or os.environ.get("SM_CHANNEL_TRAINING")
    val_dir = args.val_dir or os.environ.get("SM_CHANNEL_VALIDATION")
    test_dir = args.test_dir or os.environ.get("SM_CHANNEL_TEST")

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(train_ds.classes)
    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        num_classes=num_classes,
        early_stop_patience=args.early_stop_patience,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
    )

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes=num_classes)
    logger.info(
        f"TEST: loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.6f} macro_f1={test_metrics['macro_f1']:.6f}"
    )
    logger.info(
        f"TEST: macro_precision={test_metrics['macro_precision']:.6f} macro_recall={test_metrics['macro_recall']:.6f}"
    )
    logger.info(f"TEST: confusion_matrix=\n{np.array(test_metrics['confusion_matrix'])}")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Saved model state_dict to: {model_path}")

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f)
    logger.info(f"✅ Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)