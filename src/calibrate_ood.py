"""Phase 3.3 (Bonus 1): Temperature Scaling & `unknown` OOD threshold calibration.

1. Fits post-hoc Temperature Scaling (`T > 0`) on `splits/val.csv` (evaluating calibration
   on clean + store-domain-perturbed validation views so probabilities reflect real retail
   uncertainty without over-sharpening from label smoothing).
2. Builds a deterministic 300-image Out-of-Distribution (OOD) validation benchmark saved in
   `artifacts/ood_benchmark/` spanning 6 retail non-vegetable categories:
   - Real empty countertop/margin background patches cropped from 10MP `val.csv` stills
   - Simulated empty wooden/plastic retail shelf crates & dividers
   - Retail price tags, barcode stickers & printed labels
   - Specular polythene/plastic wrap glare without produce
   - Camera faults (covered lens, heavy motion streak, overexposed fluorescent glare)
   - Non-target packaged goods / synthetic color-texture distractors
3. Sweeps confidence thresholds (`max_prob < threshold -> "unknown"`), selects the optimal
   threshold maximizing balanced `(ID_Retention + OOD_Rejection) / 2`, saves
   `assets/ood_threshold_curve.png`, and writes `temperature` + `unknown_threshold`
   to `models/config.json` and `artifacts/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
import torch

from src.dataset import (
    CANONICAL_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VegetableDataset,
    build_eval_transforms,
    build_store_train_transforms,
    deterministic_eval_preprocess_pil,
)
from src.evaluate import load_trained_model, run_inference
from src.train import seed_everything, select_device


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute temperature-scaled softmax probabilities."""
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) across `n_bins` confidence intervals."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == targets).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for b_idx in range(n_bins):
        lo, hi = bins[b_idx], bins[b_idx + 1]
        mask = (confs > lo) & (confs <= hi) if b_idx > 0 else (confs >= lo) & (confs <= hi)
        if np.any(mask):
            ece += (mask.sum() / n) * abs(float(accs[mask].mean() - confs[mask].mean()))
    return float(ece)


def fit_temperature_scaling(
    clean_logits: np.ndarray,
    clean_targets: np.ndarray,
    aug_logits: np.ndarray,
    aug_targets: np.ndarray,
    label_smoothing_prior: float = 0.03,
) -> tuple[float, dict[str, float]]:
    """Optimize scalar temperature `T` on validation logits via smoothed cross-entropy NLL.

    Using combined clean + store-perturbed validation views with a small label-smoothing
    target (`0.03`) prevents degenerate `T -> 0` collapse when clean validation accuracy
    is ~100% after label-smoothed training, yielding well-calibrated probabilities.
    """
    all_logits = np.concatenate([clean_logits, aug_logits], axis=0)
    all_targets = np.concatenate([clean_targets, aug_targets], axis=0)
    num_classes = all_logits.shape[1]

    one_hot = np.zeros_like(all_logits, dtype=np.float64)
    one_hot[np.arange(len(all_targets)), all_targets] = 1.0
    smooth_targets = (1.0 - label_smoothing_prior) * one_hot + (label_smoothing_prior / num_classes)

    def nll_objective(temp: float) -> float:
        probs = softmax_np(all_logits, temperature=temp)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        return float(-(smooth_targets * log_probs).sum(axis=1).mean())

    opt = minimize_scalar(nll_objective, bounds=(0.5, 4.0), method="bounded")
    best_t = float(opt.x)

    ece_before = compute_ece(softmax_np(all_logits, 1.0), all_targets)
    ece_after = compute_ece(softmax_np(all_logits, best_t), all_targets)
    nll_before = nll_objective(1.0)
    nll_after = nll_objective(best_t)

    return best_t, {
        "nll_before": round(nll_before, 4),
        "nll_after": round(nll_after, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
    }


def generate_ood_benchmark(
    val_csv: Path,
    out_dir: Path,
    seed: int = 42,
    per_category: int = 50,
) -> list[tuple[Path, str]]:
    """Construct 300 deterministic retail-shelf OOD images across 6 non-vegetable categories."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[Path, str]] = []

    val_df = pd.read_csv(val_csv)
    stills_df = val_df[val_df["source_type"] == "mobile_still"].reset_index(drop=True)

    # 1. Real countertop / outer-corner background patches from 10MP validation stills
    corner_crops: list[Image.Image] = []
    for _, row in stills_df.iterrows():
        p = REPO_ROOT / str(row["filepath"])
        with Image.open(p) as im:
            im_up = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im_up.size
            patch_sz = 360
            corners = [
                (0, 0, patch_sz, patch_sz),
                (w - patch_sz, 0, w, patch_sz),
                (0, h - patch_sz, patch_sz, h),
                (w - patch_sz, h - patch_sz, w, h),
            ]
            for box in corners:
                crop = im_up.crop(box).resize((256, 256), Image.Resampling.BILINEAR)
                arr = np.asarray(crop, dtype=np.float32)
                # Keep neutral/dark/metallic background corners (low green/orange chromaticity)
                r_c, g_c, b_c = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
                chroma = max(r_c, g_c, b_c) - min(r_c, g_c, b_c)
                if chroma < 32.0:
                    corner_crops.append(crop)

    for idx in range(per_category):
        if corner_crops:
            base = corner_crops[idx % len(corner_crops)].copy()
            if idx >= len(corner_crops):
                angle = int(rng.choice([0, 90, 180, 270]))
                base = base.rotate(angle)
        else:
            noise = rng.normal(45, 12, size=(256, 256, 3)).clip(0, 255).astype(np.uint8)
            base = Image.fromarray(noise)
        fpath = out_dir / f"ood_countertop_{idx:03d}.jpg"
        base.save(fpath, quality=92)
        generated.append((fpath, "empty_countertop_patch"))

    # 2. Empty retail wooden/plastic shelf crates & grid slats
    for idx in range(per_category):
        bg_tone = int(rng.integers(35, 150))
        wood_tint = np.array(
            [bg_tone + rng.integers(0, 35), bg_tone + rng.integers(-10, 15), bg_tone - rng.integers(5, 25)],
            dtype=np.float32,
        )
        canvas = np.ones((256, 256, 3), dtype=np.float32) * wood_tint
        canvas += rng.normal(0, 9.0, size=(256, 256, 3))
        img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        step = int(rng.integers(24, 48))
        slat_col = tuple(int(c) for c in np.clip(wood_tint * 0.45, 0, 255))
        for pos in range(0, 256, step):
            draw.rectangle([pos, 0, pos + 5, 255], fill=slat_col)
            if idx % 2 == 0:
                draw.rectangle([0, pos, 255, pos + 5], fill=slat_col)
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
        fpath = out_dir / f"ood_empty_crate_{idx:03d}.jpg"
        img.save(fpath, quality=92)
        generated.append((fpath, "empty_shelf_crate"))

    # 3. Retail price tags, promotional yellow/white cards & barcode stickers
    for idx in range(per_category):
        is_promo = idx % 2 == 0
        card_col = (248, 232, 72) if is_promo else (242, 242, 240)
        bg = np.ones((256, 256, 3), dtype=np.uint8) * int(rng.integers(30, 75))
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        pad_x = int(rng.integers(14, 38))
        pad_y = int(rng.integers(24, 52))
        draw.rectangle([pad_x, pad_y, 256 - pad_x, 256 - pad_y], fill=card_col, outline=(30, 30, 30), width=3)
        # Simulated barcode / text bars
        bx0, bx1 = pad_x + 18, 256 - pad_x - 18
        for x in range(bx0, bx1, int(rng.integers(4, 8))):
            w_bar = int(rng.integers(1, 4))
            draw.rectangle([x, 145, x + w_bar, 195], fill=(20, 20, 20))
        draw.rectangle([bx0, pad_y + 15, bx1, pad_y + 34], fill=(200, 35, 35) if is_promo else (35, 35, 35))
        draw.rectangle([bx0 + 20, pad_y + 52, bx1 - 20, pad_y + 85], fill=(25, 25, 25))
        img = img.rotate(float(rng.uniform(-14, 14)), expand=False, fillcolor=(45, 45, 45))
        fpath = out_dir / f"ood_pricetag_{idx:03d}.jpg"
        img.save(fpath, quality=92)
        generated.append((fpath, "price_tag_barcode"))

    # 4. Empty frosted polythene / specular plastic glare sheets
    for idx in range(per_category):
        yy, xx = np.mgrid[0:256, 0:256]
        cx, cy = rng.uniform(60, 196), rng.uniform(60, 196)
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / float(rng.uniform(2500, 9000)))
        wrinkles = np.sin(xx / rng.uniform(5, 11) + np.cos(yy / rng.uniform(6, 13))) * 18.0
        base_val = 95.0 + 145.0 * glare + wrinkles + rng.normal(0, 8.0, size=(256, 256))
        rgb = np.stack([base_val, base_val + 3.0, base_val + 8.0], axis=-1)
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        fpath = out_dir / f"ood_plastic_glare_{idx:03d}.jpg"
        img.save(fpath, quality=92)
        generated.append((fpath, "plastic_glare"))

    # 5. Camera faults: covered dark lens, severe motion streak, blown-out ceiling light
    for idx in range(per_category):
        mode = idx % 3
        if mode == 0:
            # Dark covered / pocket frame
            arr = rng.normal(14, 5, size=(256, 256, 3)).clip(0, 45).astype(np.uint8)
            img = Image.fromarray(arr)
        elif mode == 1:
            # Overexposed fluorescent ceiling light
            arr = rng.normal(244, 7, size=(256, 256, 3)).clip(210, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            # Heavy aisle motion streak
            stripes = rng.integers(25, 225, size=(256, 12, 3), dtype=np.uint8)
            img = Image.fromarray(stripes).resize((256, 256), Image.Resampling.BILINEAR)
            img = img.filter(ImageFilter.GaussianBlur(radius=6.0))
        fpath = out_dir / f"ood_camera_fault_{idx:03d}.jpg"
        img.save(fpath, quality=92)
        generated.append((fpath, "camera_fault"))

    # 6. Out-of-scope packaged goods / non-produce blue-purple-crimson boxes & cans
    palette = [
        (35, 85, 195),   # blue carton
        (135, 45, 165),  # purple box
        (30, 145, 165),  # cyan tin
        (185, 35, 95),   # magenta pack
        (125, 130, 138), # brushed aluminum can
    ]
    for idx in range(per_category):
        bg = rng.normal(50, 10, size=(256, 256, 3)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        col = palette[idx % len(palette)]
        x0, y0 = int(rng.integers(28, 68)), int(rng.integers(25, 55))
        x1, y1 = int(rng.integers(188, 228)), int(rng.integers(195, 232))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=col, outline=(220, 220, 225), width=2)
        draw.rectangle([x0 + 12, y0 + 28, x1 - 12, y0 + 68], fill=(240, 240, 245))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        fpath = out_dir / f"ood_packaged_item_{idx:03d}.jpg"
        img.save(fpath, quality=92)
        generated.append((fpath, "non_produce_package"))

    return generated


@torch.no_grad()
def infer_ood_logits(
    model: torch.nn.Module,
    ood_items: list[tuple[Path, str]],
    device: torch.device,
    img_size: int = 224,
    batch_size: int = 64,
) -> np.ndarray:
    """Preprocess and run forward pass on OOD benchmark images."""
    resize_short = int(round(img_size * 256 / 224))
    tensors: list[np.ndarray] = []
    for fpath, _ in ood_items:
        with Image.open(fpath) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            tensors.append(deterministic_eval_preprocess_pil(rgb, resize_short=resize_short, crop_size=img_size))

    arr_all = np.stack(tensors, axis=0)
    logits_list: list[np.ndarray] = []
    for start in range(0, len(arr_all), batch_size):
        batch = torch.from_numpy(arr_all[start : start + batch_size]).to(device)
        logits_list.append(model(batch).cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def plot_ood_calibration(
    id_confs: np.ndarray,
    ood_confs: np.ndarray,
    thresholds: np.ndarray,
    id_retention: np.ndarray,
    ood_rejection: np.ndarray,
    balanced_scores: np.ndarray,
    best_thresh: float,
    temperature: float,
    auroc: float,
    out_path: Path,
) -> None:
    """Render 2-panel OOD calibration histogram & threshold sweep (`assets/ood_threshold_curve.png`)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=180)

    # Panel A: Confidence distribution
    ax = axes[0]
    bins = np.linspace(0.15, 1.0, 35)
    ax.hist(id_confs, bins=bins, alpha=0.68, density=True, label=f"In-Distribution Val (N={len(id_confs)})", color="#2b6cb0")
    ax.hist(ood_confs, bins=bins, alpha=0.68, density=True, label=f"OOD Store Benchmark (N={len(ood_confs)})", color="#c53030")
    ax.axvline(best_thresh, color="#1a202c", linestyle="--", linewidth=2.0, label=f"Threshold = {best_thresh:.2f}")
    ax.set_xlabel("Temperature-Scaled Max Softmax Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"A. Calibrated Confidence Separation (T={temperature:.3f}, AUROC={auroc:.4f})", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.8)
    ax.grid(alpha=0.3, linestyle="--")

    # Panel B: Threshold sweep
    ax = axes[1]
    ax.plot(thresholds, id_retention * 100.0, label="ID Target Retention (%)", color="#2b6cb0", linewidth=2.2)
    ax.plot(thresholds, ood_rejection * 100.0, label="OOD Rejection as 'unknown' (%)", color="#c53030", linewidth=2.2)
    ax.plot(thresholds, balanced_scores * 100.0, label="Balanced Score (0.5 * [ID + OOD])", color="#2f855a", linewidth=2.4, linestyle="-.")
    ax.axvline(best_thresh, color="#1a202c", linestyle="--", linewidth=1.8, label=f"Selected Threshold = {best_thresh:.2f}")
    ax.set_xlabel("Confidence Threshold (max_prob < threshold -> 'unknown')")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 103)
    ax.set_title("B. OOD Threshold Sweep (ID Retention vs OOD Rejection)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.8, loc="lower left")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate temperature and unknown OOD threshold.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--val-csv", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--config-out", type=Path, default=Path("models/config.json"))
    parser.add_argument("--plot-out", type=Path, default=Path("assets/ood_threshold_curve.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    model, ckpt = load_trained_model(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 224))

    # 1. Clean + store-perturbed validation logits
    val_clean_ds = VegetableDataset(args.val_csv, repo_root=REPO_ROOT, transform=build_eval_transforms(img_size))
    seed_everything(args.seed)
    val_aug_ds = VegetableDataset(args.val_csv, repo_root=REPO_ROOT, transform=build_store_train_transforms(img_size))

    clean_res = run_inference(model, val_clean_ds, device)
    aug_res = run_inference(model, val_aug_ds, device)

    temperature, cal_stats = fit_temperature_scaling(
        clean_res["logits"], clean_res["targets"], aug_res["logits"], aug_res["targets"]
    )
    print(f"[INFO] Calibrated Temperature T = {temperature:.4f} | ECE: {cal_stats['ece_before']:.4f} -> {cal_stats['ece_after']:.4f}")

    # 2. Build 300-image OOD benchmark and compute calibrated confidences
    ood_dir = REPO_ROOT / "artifacts" / "ood_benchmark"
    ood_items = generate_ood_benchmark(args.val_csv, ood_dir, seed=args.seed, per_category=50)
    ood_logits = infer_ood_logits(model, ood_items, device, img_size=img_size)

    # Combine clean + store-augmented ID val probabilities for realistic store threshold selection
    id_logits_combined = np.concatenate([clean_res["logits"], aug_res["logits"]], axis=0)
    id_targets_combined = np.concatenate([clean_res["targets"], aug_res["targets"]], axis=0)

    id_probs = softmax_np(id_logits_combined, temperature=temperature)
    id_confs = id_probs.max(axis=1)
    id_preds = id_probs.argmax(axis=1)

    clean_id_probs = softmax_np(clean_res["logits"], temperature=temperature)
    clean_id_confs = clean_id_probs.max(axis=1)
    clean_id_preds = clean_id_probs.argmax(axis=1)

    ood_probs = softmax_np(ood_logits, temperature=temperature)
    ood_confs = ood_probs.max(axis=1)

    y_binary = np.concatenate([np.ones(len(id_confs)), np.zeros(len(ood_confs))])
    scores_binary = np.concatenate([id_confs, ood_confs])
    auroc = float(roc_auc_score(y_binary, scores_binary))

    # 3. Sweep thresholds in [0.30, 0.96]
    thresholds = np.round(np.linspace(0.30, 0.96, 133), 3)
    id_retention = np.array([float(((id_confs >= t) & (id_preds == id_targets_combined)).mean()) for t in thresholds])
    clean_id_retention = np.array([float(((clean_id_confs >= t) & (clean_id_preds == clean_res["targets"])).mean()) for t in thresholds])
    ood_rejection = np.array([float((ood_confs < t).mean()) for t in thresholds])
    balanced_scores = 0.5 * (id_retention + ood_rejection)

    # Require clean ID retention >= 0.92 and practical upper cap <= 0.90 to avoid brittle rejection
    feasible = np.where((clean_id_retention >= 0.92) & (thresholds <= 0.90))[0]
    best_idx = int(feasible[np.argmax(balanced_scores[feasible])]) if len(feasible) > 0 else int(np.argmax(balanced_scores))
    best_thresh = float(thresholds[best_idx])

    per_cat_rejection: dict[str, float] = {}
    cats = np.array([c for _, c in ood_items])
    for cat_name in sorted(set(cats)):
        m = cats == cat_name
        per_cat_rejection[cat_name] = round(float((ood_confs[m] < best_thresh).mean() * 100.0), 2)

    plot_ood_calibration(
        id_confs=id_confs,
        ood_confs=ood_confs,
        thresholds=thresholds,
        id_retention=id_retention,
        ood_rejection=ood_rejection,
        balanced_scores=balanced_scores,
        best_thresh=best_thresh,
        temperature=temperature,
        auroc=auroc,
        out_path=args.plot_out,
    )

    # Update models/config.json
    cfg: dict[str, Any] = {
        "architecture": "mobilenet_v3_large",
        "classes": CANONICAL_CLASSES,
        "unknown_label": "unknown",
        "img_size": img_size,
        "resize_short": int(round(img_size * 256 / 224)),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "temperature": round(temperature, 4),
        "unknown_threshold": round(best_thresh, 4),
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Update artifacts/metrics.json
    metrics_path = REPO_ROOT / "artifacts" / "metrics.json"
    metrics_data: dict[str, Any] = {}
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_data["ood_calibration"] = {
        "temperature": round(temperature, 4),
        "unknown_threshold": round(best_thresh, 4),
        "ood_benchmark_size": len(ood_items),
        "auroc_id_vs_ood": round(auroc, 4),
        "clean_val_id_retention_pct": round(float(clean_id_retention[best_idx] * 100.0), 2),
        "combined_val_id_retention_pct": round(float(id_retention[best_idx] * 100.0), 2),
        "ood_rejection_pct": round(float(ood_rejection[best_idx] * 100.0), 2),
        "balanced_score_pct": round(float(balanced_scores[best_idx] * 100.0), 2),
        "id_mean_confidence": round(float(id_confs.mean()), 4),
        "ood_mean_confidence": round(float(ood_confs.mean()), 4),
        "calibration_stats": cal_stats,
        "per_category_ood_rejection_pct": per_cat_rejection,
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    print(
        f"[OK] Selected unknown_threshold = {best_thresh:.4f} (T = {temperature:.4f}) | "
        f"Clean ID Retention = {clean_id_retention[best_idx]*100:.2f}% | "
        f"Combined ID Retention = {id_retention[best_idx]*100:.2f}% | "
        f"OOD Rejection = {ood_rejection[best_idx]*100:.2f}% | AUROC = {auroc:.4f}"
    )
    print(f"[OK] Saved {args.config_out} and {args.plot_out}")


if __name__ == "__main__":
    main()
