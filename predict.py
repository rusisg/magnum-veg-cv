#!/usr/bin/env python3
"""Offline CPU-only retail shelf vegetable classifier CLI (`predict.py`).

Usage:
    python predict.py --input путь/к/папке --output preds.csv

Dependencies:
    Strictly CPU-only: `onnxruntime`, `numpy`, `Pillow`, and Python standard library.
    Zero network calls, zero PyTorch/timm/OpenCV runtime dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_LABELS = {
    "Capsicum",
    "Carrot",
    "Cucumber",
    "Potato",
    "Raddish",
    "Squash",
    "unknown",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("predict")


def discover_images(input_path: Path) -> list[tuple[Path, str]]:
    """Discover supported images recursively; return `(abs_path, relative_filename)`."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [(input_path, input_path.name)]
        return []

    found: list[tuple[Path, str]] = []
    for root, dirs, files in os.walk(input_path):
        # Skip hidden directories deterministically
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            fpath = Path(root) / fname
            if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel_name = fpath.relative_to(input_path).as_posix()
                found.append((fpath, rel_name))

    found.sort(key=lambda item: item[1])
    return found


def preprocess_image(
    img_path: Path,
    img_size: int,
    resize_short: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """EXIF-safe RGB load -> shortest-side resize -> center crop -> ImageNet normalize."""
    with Image.open(img_path) as img:
        img_upright = ImageOps.exif_transpose(img)
        rgb = img_upright.convert("RGB")

    w, h = rgb.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    if w <= h:
        new_w = resize_short
        new_h = int(round(h * (resize_short / w)))
    else:
        new_h = resize_short
        new_w = int(round(w * (resize_short / h)))

    if (w, h) != (new_w, new_h):
        rgb = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = int(round((new_w - img_size) / 2.0))
    top = int(round((new_h - img_size) / 2.0))
    rgb = rgb.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute numerically stable temperature-scaled softmax over 1D/2D logits."""
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def load_config_and_session(
    model_path: Path,
    config_path: Path,
) -> tuple[ort.InferenceSession, dict[str, Any]]:
    """Load local `config.json` and initialize offline CPU ONNX Runtime session."""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model file: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Disable any ONNX Runtime background telemetry for strict zero-network operation
    if hasattr(ort, "disable_telemetry_events"):
        ort.disable_telemetry_events()

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 1)

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    active_providers = session.get_providers()
    if active_providers != ["CPUExecutionProvider"]:
        logger.warning("Unexpected providers %s; forcing CPUExecutionProvider.", active_providers)

    return session, config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Offline CPU inference for Magnum retail shelf vegetable classification."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input folder (or single image) to classify.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output CSV file (columns: filename,label,confidence).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=base_dir / "models" / "veg_classifier.onnx",
        help="Path to local ONNX model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "models" / "config.json",
        help="Path to local model config JSON.",
    )
    return parser.parse_args(argv)


def run_predictions(args: argparse.Namespace) -> int:
    session, config = load_config_and_session(args.model, args.config)
    input_name = session.get_inputs()[0].name

    classes: list[str] = list(config["classes"])
    unknown_label: str = str(config.get("unknown_label", "unknown"))
    img_size = int(config.get("img_size", 224))
    resize_short = int(config.get("resize_short", int(round(img_size * 256 / 224))))
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    temperature = float(config.get("temperature", 1.0))
    unknown_threshold = float(config.get("unknown_threshold", 0.5))

    image_items = discover_images(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    rows: list[tuple[str, str, str]] = []

    for img_path, rel_name in image_items:
        try:
            batch = preprocess_image(
                img_path=img_path,
                img_size=img_size,
                resize_short=resize_short,
                mean=mean,
                std=std,
            )
            logits = session.run(None, {input_name: batch})[0]
            probs = calibrated_softmax(logits, temperature=temperature)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label = unknown_label if confidence < unknown_threshold else classes[pred_idx]
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Predicted label '{label}' not in allowed contract {ALLOWED_LABELS}")
            rows.append((rel_name, label, f"{confidence:.4f}"))
            processed += 1
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Skipping corrupted or unreadable image '%s': %s", rel_name, exc)
            skipped += 1
        except Exception as exc:
            logger.warning("Unexpected error processing '%s' (skipping): %s", rel_name, exc)
            skipped += 1

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        writer.writerows(rows)

    logger.info(
        "Wrote %d predictions to %s (skipped %d corrupted files).",
        processed,
        args.output,
        skipped,
    )
    return processed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_predictions(args)


if __name__ == "__main__":
    main()
