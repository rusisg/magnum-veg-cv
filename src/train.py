"""Phase 2: Transfer learning of MobileNetV3-Large with store-realistic augmentations.

Features:
- Hardware auto-selection: `cuda` -> `mps` (Apple Silicon) -> `cpu`.
- Deterministic seeding across Python, NumPy, PyTorch, and DataLoader workers (`seed=42`).
- Inverse-frequency class-weighted `CrossEntropyLoss` with `label_smoothing=0.05`
  to handle the 4.7x class imbalance (`Squash` 1,343 vs `Raddish` 286).
- Differential-LR `AdamW` + `CosineAnnealingLR` schedule.
- Checkpoints best model to `models/best_model.pt` selected strictly by leak-free
  validation **Macro-F1**, and logs epoch history to `artifacts/train_history.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

# Cache pretrained weights inside workspace artifacts/ so downstream scripts work offline/sandboxed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "artifacts" / "torch_cache"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from src.dataset import (
    CANONICAL_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VegetableDataset,
    build_eval_transforms,
    build_store_train_transforms,
)


def select_device() -> torch.device:
    """Auto-select compute device in priority order: `cuda` -> `mps` -> `cpu`."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(num_classes: int = 6, pretrained: bool = True, dropout: float = 0.25) -> nn.Module:
    """Instantiate MobileNetV3-Large replacing the final classifier layer for 6 classes."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(train_csv: Path, num_classes: int = 6) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights: N / (K * count_c)."""
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label_idx"].to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float64)
    inv_freq = len(df) / (num_classes * np.maximum(counts, 1.0))
    inv_freq = inv_freq / inv_freq.mean()
    return torch.tensor(inv_freq, dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run deterministic evaluation pass returning loss, accuracy, and macro-F1."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_sz = targets.size(0)
        total_loss += float(loss.item()) * batch_sz
        total_samples += batch_sz

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large on leak-free vegetable splits.")
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"), help="Directory containing train.csv and val.csv.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base backbone learning rate (head uses 3x).")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="CrossEntropy label smoothing.")
    parser.add_argument("--img-size", type=int, default=224, help="Input image resolution.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0 = main process, deterministic).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("models/best_model.pt"), help="Path to save best checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    print(f"[INFO] Selected device: {device} | Seed: {args.seed}", flush=True)

    train_csv = args.splits_dir / "train.csv"
    val_csv = args.splits_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Expected {train_csv} and {val_csv}. Run src/eda_and_split.py first.")

    train_ds = VegetableDataset(
        train_csv,
        repo_root=REPO_ROOT,
        transform=build_store_train_transforms(args.img_size),
        cache_images=True,
    )
    val_ds = VegetableDataset(
        val_csv,
        repo_root=REPO_ROOT,
        transform=build_eval_transforms(args.img_size),
        cache_images=True,
        max_cache_short_side=256,
    )

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        worker_init_fn=_seed_worker,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    class_weights = compute_class_weights(train_csv, num_classes=len(CANONICAL_CLASSES)).to(device)
    print(
        "[INFO] Class weights: "
        + ", ".join(f"{cls_name}={w:.3f}" for cls_name, w in zip(CANONICAL_CLASSES, class_weights.cpu().tolist()))
    )

    model = build_model(num_classes=len(CANONICAL_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier.") and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 3.0},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        total_seen = 0
        train_preds: list[int] = []
        train_targets: list[int] = []

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bsz = targets.size(0)
            running_loss += float(loss.item()) * bsz
            total_seen += bsz
            train_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            train_targets.extend(targets.detach().cpu().tolist())

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        train_loss = running_loss / max(total_seen, 1)
        train_acc = float(accuracy_score(train_targets, train_preds))
        train_f1 = float(f1_score(train_targets, train_preds, average="macro", zero_division=0))
        val_metrics = evaluate_split(model, val_loader, val_criterion, device)
        ep_sec = time.perf_counter() - ep_t0

        improved = val_metrics["macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": "mobilenet_v3_large",
                    "classes": CANONICAL_CLASSES,
                    "img_size": args.img_size,
                    "imagenet_mean": list(IMAGENET_MEAN),
                    "imagenet_std": list(IMAGENET_STD),
                    "val_macro_f1": best_val_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "model_state_dict": model.state_dict(),
                },
                args.checkpoint_out,
            )

        row = {
            "epoch": epoch,
            "lr": round(current_lr, 7),
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "epoch_sec": round(ep_sec, 2),
            "is_best": improved,
        }
        history.append(row)
        marker = " *BEST*" if improved else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} ({ep_sec:.1f}s){marker}",
            flush=True,
        )

    total_sec = time.perf_counter() - t_start
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_summary = {
        "architecture": "MobileNetV3-Large (IMAGENET1K_V2)",
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_lr": args.lr,
        "head_lr": args.lr * 3.0,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weights": {c: round(float(w), 4) for c, w in zip(CANONICAL_CLASSES, class_weights.cpu().tolist())},
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(history[best_epoch - 1]["val_accuracy"], 4),
        "total_train_time_sec": round(total_sec, 2),
        "history": history,
    }
    (artifacts_dir / "train_history.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")

    # Initialize or update models/config.json with defaults prior to OOD calibration
    config_path = REPO_ROOT / "models" / "config.json"
    existing_cfg: dict[str, Any] = {}
    if config_path.exists():
        existing_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = {
        "architecture": "mobilenet_v3_large",
        "classes": CANONICAL_CLASSES,
        "unknown_label": "unknown",
        "img_size": args.img_size,
        "resize_short": int(round(args.img_size * 256 / 224)),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "temperature": existing_cfg.get("temperature", 1.0),
        "unknown_threshold": existing_cfg.get("unknown_threshold", 0.5),
    }
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    metrics_path = artifacts_dir / "metrics.json"
    metrics_data: dict[str, Any] = {}
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_data["training"] = {
        k: v for k, v in train_summary.items() if k != "history"
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    print(
        f"\n[OK] Training complete in {total_sec:.1f}s. Best checkpoint at epoch {best_epoch} "
        f"(Val Macro-F1 = {best_val_f1:.4f}) -> {args.checkpoint_out}"
    )


if __name__ == "__main__":
    main()
