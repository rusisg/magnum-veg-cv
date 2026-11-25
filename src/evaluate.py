"""Phase 3.1: Leak-free test set evaluation, confusion matrix, and stress-test audit.

Computes overall Accuracy, Macro-F1, Weighted-F1, per-class Precision/Recall/F1,
confusion matrix (`assets/confusion_matrix.png`), hard/low-confidence test samples,
and a simulated retail-shelf domain-shift stress evaluation. Writes real metrics
to `artifacts/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    CANONICAL_CLASSES,
    VegetableDataset,
    build_eval_transforms,
    build_store_train_transforms,
)
from src.train import build_model, seed_everything, select_device


def load_trained_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load trained MobileNetV3-Large checkpoint onto `device`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt.get("classes", CANONICAL_CLASSES)
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: VegetableDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run inference returning targets, predictions, probabilities, logits, and filepaths."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits: list[np.ndarray] = []
    all_targets: list[int] = []
    all_paths: list[str] = []

    for images, targets, paths in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_targets.extend(targets.tolist())
        all_paths.extend(list(paths))

    logits_arr = np.concatenate(all_logits, axis=0)
    shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs_arr = exp_l / exp_l.sum(axis=1, keepdims=True)
    preds_arr = probs_arr.argmax(axis=1)
    confs_arr = probs_arr.max(axis=1)

    return {
        "logits": logits_arr,
        "probs": probs_arr,
        "preds": preds_arr,
        "confs": confs_arr,
        "targets": np.array(all_targets, dtype=np.int64),
        "paths": all_paths,
    }


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    """Save dual-panel (raw counts + row-normalized recall %) confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=180)

    # Panel 1: Raw counts
    ax = axes[0]
    im0 = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Test Confusion Matrix (Counts)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    thresh0 = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh0 else "#1a202c",
                fontsize=9.5,
                fontweight="bold" if i == j else "normal",
            )

    # Panel 2: Row-normalized Recall (%)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(np.float64) / row_sums * 100.0
    ax = axes[1]
    im1 = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=100.0)
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Test Confusion Matrix (Row Recall %)", fontsize=11.5, fontweight="bold")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            txt = f"{val:.1f}%" if val > 0 else "0%"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > 55.0 else "#1a202c",
                fontsize=9.0,
                fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on leak-free test split.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--test-csv", type=Path, default=Path("splits/test.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cm-out", type=Path, default=Path("assets/confusion_matrix.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    print(f"[INFO] Evaluating {args.checkpoint} on {args.test_csv} (device={device})")

    model, ckpt = load_trained_model(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 224))

    test_ds = VegetableDataset(
        args.test_csv,
        repo_root=REPO_ROOT,
        transform=build_eval_transforms(img_size),
    )
    res = run_inference(model, test_ds, device, batch_size=args.batch_size)

    targets = res["targets"]
    preds = res["preds"]
    confs = res["confs"]
    probs = res["probs"]
    paths = res["paths"]

    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
    prec, rec, f1_arr, supp = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(CANONICAL_CLASSES))), zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(CANONICAL_CLASSES))))

    per_class_metrics: dict[str, dict[str, Any]] = {}
    for idx, cls_name in enumerate(CANONICAL_CLASSES):
        per_class_metrics[cls_name] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1_arr[idx]), 4),
            "support": int(supp[idx]),
        }

    # Simulated retail store stress test (seed=42) with heavy lighting/angle/occlusion transforms
    seed_everything(args.seed)
    stress_ds = VegetableDataset(
        args.test_csv,
        repo_root=REPO_ROOT,
        transform=build_store_train_transforms(img_size),
    )
    stress_res = run_inference(model, stress_ds, device, batch_size=args.batch_size)
    stress_acc = float(accuracy_score(stress_res["targets"], stress_res["preds"]))
    stress_f1 = float(f1_score(stress_res["targets"], stress_res["preds"], average="macro", zero_division=0))

    # Record misclassifications and lowest-confidence edge cases
    misclassified: list[dict[str, Any]] = []
    for i in np.where(preds != targets)[0]:
        misclassified.append(
            {
                "filepath": paths[i],
                "true_label": CANONICAL_CLASSES[int(targets[i])],
                "pred_label": CANONICAL_CLASSES[int(preds[i])],
                "confidence": round(float(confs[i]), 4),
                "true_class_prob": round(float(probs[i, int(targets[i])]), 4),
            }
        )

    low_conf_indices = np.argsort(confs)[:10]
    hard_examples = [
        {
            "filepath": paths[int(i)],
            "true_label": CANONICAL_CLASSES[int(targets[int(i)])],
            "pred_label": CANONICAL_CLASSES[int(preds[int(i)])],
            "correct": bool(preds[int(i)] == targets[int(i)]),
            "confidence": round(float(confs[int(i)]), 4),
        }
        for i in low_conf_indices
    ]

    plot_confusion_matrix(cm, CANONICAL_CLASSES, args.cm_out)
    print(f"[OK] Saved confusion matrix -> {args.cm_out}")
    print("\n" + classification_report(targets, preds, target_names=CANONICAL_CLASSES, digits=4))
    print(f"Test Accuracy:          {acc:.4f}")
    print(f"Test Macro-F1:          {macro_f1:.4f}")
    print(f"Store-Stress Accuracy:  {stress_acc:.4f} | Store-Stress Macro-F1: {stress_f1:.4f}")
    print(f"Misclassified samples:  {len(misclassified)} / {len(targets)}")

    metrics_path = REPO_ROOT / "artifacts" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data: dict[str, Any] = {}
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_data["test_evaluation"] = {
        "test_csv": str(args.test_csv),
        "num_samples": int(len(targets)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "misclassified_count": len(misclassified),
        "store_augmented_stress_accuracy": round(stress_acc, 4),
        "store_augmented_stress_macro_f1": round(stress_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "misclassified_examples": misclassified,
        "lowest_confidence_examples": hard_examples,
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"[OK] Updated {metrics_path}")


if __name__ == "__main__":
    main()
