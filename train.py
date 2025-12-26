import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ----------------------------
# Optional: SageMaker Debugger (smdebug)
# ----------------------------
def try_create_smdebug_hook():
    """
    Creates an smdebug hook if configuration is present in the container.
    If smdebug isn't available or not configured, returns None.
    """
    try:
        import smdebug.pytorch as smd
        hook = smd.Hook.create_from_json_file()
        logger.info("✅ SMDebug hook created from JSON config.")
        return hook
    except Exception as e:
        logger.info(f"smdebug not enabled/available (this is OK). Reason: {type(e).__name__}: {e}")
        return None

def _ckpt_path(checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pt")

def load_checkpoint_if_available(model, optimizer, checkpoint_dir: str):
    path = _ckpt_path(checkpoint_dir)

    # Always return 4 values
    if not os.path.exists(path):
        return 1, float("inf"), None, 0

    ckpt = torch.load(path, map_location="cpu")

    # Resume optimizer + model
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    if "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    # Support both "new" and "legacy" checkpoint formats
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("best_val", float("inf"))))
    best_state = ckpt.get("best_state", None)
    no_improve = int(ckpt.get("no_improve", 0))

    logger.info(f"✅ Resuming from checkpoint {path} (start_epoch={start_epoch}, best_val_loss={best_val_loss})")
    return start_epoch, best_val_loss, best_state, no_improve

def save_checkpoint(model, optimizer, epoch: int, best_val_loss: float, best_state, no_improve: int, checkpoint_dir: str):
    """
    Save checkpoint into SageMaker checkpoint folder.
    IMPORTANT: When using Managed Spot Training, SageMaker syncs checkpoint_local_path (e.g. /opt/ml/checkpoints)
    to checkpoint_s3_uri, and restores it on restart so training can resume. :contentReference[oaicite:0]{index=0}
    """
    path = _ckpt_path(checkpoint_dir)

    # best_state can be None early on; that's OK.
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": float(best_val_loss),
            "no_improve": int(no_improve),
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_state": best_state,  # can be None
        },
        path,
    )
    logger.info(f"✅ Saved checkpoint -> {path}")

# ----------------------------
# Utility: Metrics from confusion matrix (no sklearn dependency)
# ----------------------------
def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # per-class precision, recall, f1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1        = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return precision, recall, f1

# ----------------------------
# Data
# ----------------------------
def create_data_loaders(train_dir: str, val_dir: str, test_dir: str,
                        batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Creates ImageFolder datasets and data loaders.
    Expects directory structure:
      train/<class>/*.jpg
      val/<class>/*.jpg
      test/<class>/*.jpg
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = torchvision.datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds   = torchvision.datasets.ImageFolder(root=val_dir, transform=eval_tfms)
    test_ds  = torchvision.datasets.ImageFolder(root=test_dir, transform=eval_tfms)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    logger.info(f"Classes (ImageFolder): {train_ds.classes}")
    logger.info(f"Class->index mapping: {train_ds.class_to_idx}")
    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_loader, val_loader, test_loader, num_classes

# ----------------------------
# Model
# ----------------------------
def net(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Initializes a pretrained ResNet-50 and replaces the classifier head for our classes.
    """
    model = models.resnet50(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features  # typically 2048 for resnet50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model

# ----------------------------
# Train/Eval loops
# ----------------------------
def run_one_epoch(model: nn.Module,
                  loader: DataLoader,
                  criterion: nn.Module,
                  optimizer: Optional[optim.Optimizer],
                  device: torch.device,
                  hook=None,
                  is_train: bool = True) -> Dict[str, float]:
    """
    Runs one epoch. If is_train=False, optimizer can be None.
    Returns dict with loss and accuracy.
    """
    if is_train:
        model.train()
        if hook is not None:
            try:
                import smdebug.pytorch as smd
                hook.set_mode(smd.modes.TRAIN)
            except Exception:
                pass
    else:
        model.eval()
        if hook is not None:
            try:
                import smdebug.pytorch as smd
                hook.set_mode(smd.modes.EVAL)
            except Exception:
                pass

    running_loss = 0.0
    running_correct = 0
    total = 0

    all_true = []
    all_pred = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        running_correct += (preds == labels).sum().item()
        total += bs

        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_correct / max(1, total)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return {"loss": float(epoch_loss), "accuracy": float(epoch_acc), "y_true": y_true, "y_pred": y_pred}

def train_and_validate(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       criterion: nn.Module,
                       optimizer: optim.Optimizer,
                       device: torch.device,
                       epochs: int,
                       num_classes: int,
                       hook=None,
                       early_stop_patience: int = 3,
                       checkpoint_dir: str = "/opt/ml/checkpoints") -> nn.Module:
    """
    Train with simple early stopping on validation loss.
    Logs metrics in a regex-friendly format for SageMaker metric capture.
    """
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    start_epoch, best_val_loss, best_state, no_improve = load_checkpoint_if_available(
    model, optimizer, checkpoint_dir
    )

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, hook, is_train=True)
        val_metrics   = run_one_epoch(model, val_loader, criterion, None, device, hook, is_train=False)

        # Compute macro F1 on validation
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

        # Early stopping
        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                logger.info(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

        # For Udacity rubric: keep runtime reasonable in demo runs
        if epoch == 1 and os.environ.get("UDACITY_FAST_RUN", "0") == "1":
            logger.info("UDACITY_FAST_RUN enabled -> stopping after 1 epoch.")
            break
        save_checkpoint(model, optimizer, epoch, best_val_loss, best_state, no_improve, checkpoint_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def test(model: nn.Module,
         test_loader: DataLoader,
         criterion: nn.Module,
         device: torch.device,
         num_classes: int,
         hook=None) -> Dict[str, float]:
    """
    Evaluate on test set and print metrics (accuracy + macro F1 + confusion matrix).
    """
    test_metrics = run_one_epoch(model, test_loader, criterion, None, device, hook, is_train=False)

    cm = confusion_matrix_numpy(test_metrics["y_true"], test_metrics["y_pred"], num_classes=num_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    logger.info(f"TEST: loss={test_metrics['loss']:.6f} acc={test_metrics['accuracy']:.6f} macro_f1={macro_f1:.6f}")
    logger.info(f"TEST: macro_precision={macro_precision:.6f} macro_recall={macro_recall:.6f}")
    logger.info(f"TEST: confusion_matrix=\n{cm}")

    return {
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "test_macro_f1": macro_f1,
    }


def str2bool(v):
    """
    Robust bool parser for SageMaker hyperparameters.
    Accepts True/False as strings.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

# ----------------------------
# Main
# ----------------------------
def main(args):
    try:
        # SageMaker env vars
        train_dir = args.train_dir or os.environ.get("SM_CHANNEL_TRAINING")
        val_dir   = args.val_dir   or os.environ.get("SM_CHANNEL_VALIDATION")
        test_dir  = args.test_dir  or os.environ.get("SM_CHANNEL_TEST")
        model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)

        if not train_dir or not val_dir or not test_dir:
            raise ValueError(
                "Training/validation/test directories not set. "
                "Provide --train_dir/--val_dir/--test_dir or use SageMaker channels."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Loaders
        train_loader, val_loader, test_loader, num_classes = create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Model
        model = net(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
        model = model.to(device)

        # Loss + optimizer
        criterion = nn.CrossEntropyLoss()
        params = model.parameters() if not args.freeze_backbone else model.fc.parameters()
        optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

        # Optional debugger hook
        hook = try_create_smdebug_hook()
        if hook is not None:
            try:
                hook.register_hook(model)
                hook.register_loss(criterion)
                logger.info("✅ SMDebug hook registered to model and loss.")
            except Exception as e:
                logger.info(f"Could not register smdebug hook (continuing). Reason: {e}")
                hook = None

        # Smoke test mode (quick verification)
        if args.smoke_test:
            logger.info("Running SMOKE TEST: one forward/backward step on a single batch.")
            model.train()
            inputs, labels = next(iter(train_loader))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"SMOKE TEST OK. loss={loss.item():.6f}")
            return

        # Train + validate
        model = train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            num_classes=num_classes,
            hook=hook,
            early_stop_patience=args.early_stop_patience,
            checkpoint_dir="/opt/ml/checkpoints"
        )

        # Test
        metrics = test(model, test_loader, criterion, device, num_classes=num_classes, hook=hook)

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Saved model state_dict to: {model_path}")

        # Save metrics (handy for reports)
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Saved metrics to: {metrics_path}")

    except Exception as e:
        logger.error(f"Training script failed: {type(e).__name__}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters (used later by HPO)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Training behavior
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)

    # Paths (optional locally; SageMaker usually supplies SM_CHANNEL_* env vars)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument('--checkpoint_dir', type=str, default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    
    # Smoke test
    parser.add_argument("--smoke_test", action="store_true", default=False)

    args = parser.parse_args()
    logger.info(f"Args: {args}")
    main(args)