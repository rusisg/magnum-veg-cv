"""Phase 1: Dataset acquisition, forensic leakage audit, EDA, and leak-free splitting.

Audits both `Dataset/` and `final_dataset/` from `preethi866/vegetable-dataset`:
1. Verifies dataset availability in `--data-dir` (downloads via Kaggle CLI if absent).
2. Analyzes class counts, resolutions, EXIF orientation tags, aspect ratios, RGB/luminance
   brightness & contrast distributions, and filename conventions (video frames vs burst stills).
3. Computes cryptographic hashes (MD5, SHA256), perceptual hashes (64-bit dHash & pHash),
   and normalized spatial-color patch embeddings to detect exact duplicates, near-duplicates,
   and sequential burst clusters via connected components (Union-Find).
4. Audits `final_dataset/train`, `val`, and `test` for cross-split exact & near-duplicate leakage.
5. Constructs a leak-free 70/15/15 `train/val/test` split using `StratifiedGroupKFold` grouped
   by near-duplicate/sequence clusters, verifying zero cross-split leakage.
6. Exports `artifacts/eda_summary.json`, `splits/{train,val,test}.csv`,
   `assets/class_distribution.png`, and `assets/duplicates_examples.png`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

CANONICAL_CLASSES = [
    "Capsicum",
    "Carrot",
    "Cucumber",
    "Potato",
    "Raddish",
    "Squash",
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}
CLASS_LOOKUP = {cls_name.lower(): cls_name for cls_name in CANONICAL_CLASSES}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

VIDEO_FRAME_RE = re.compile(r"^(\d{8}_\d{6})_frame_(\d+)\.[a-zA-Z0-9]+$")
TIMESTAMP_STILL_RE = re.compile(r"^(\d{8})_(\d{6})(?:_(\d+))?\.[a-zA-Z0-9]+$")


def ensure_dataset_present(data_dir: Path) -> tuple[Path, Path]:
    """Ensure `Dataset/` and `final_dataset/` exist inside `data_dir`, downloading if absent."""
    dataset_root = _find_dataset_subdir(data_dir, "Dataset")
    final_root = _find_dataset_subdir(data_dir, "final_dataset")

    if dataset_root is not None and final_root is not None:
        return dataset_root, final_root

    print(f"[INFO] Dataset not found in {data_dir}. Attempting Kaggle CLI download...")
    data_dir.mkdir(parents=True, exist_ok=True)
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    cmd = [
        str(kaggle_bin if kaggle_bin.exists() else "kaggle"),
        "datasets",
        "download",
        "-d",
        "preethi866/vegetable-dataset",
        "-p",
        str(data_dir),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install `kaggle` or place the dataset under ./data/."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle dataset download failed. Ensure ~/.kaggle/kaggle.json or "
            "KAGGLE_USERNAME / KAGGLE_KEY credentials are configured."
        ) from exc

    dataset_root = _find_dataset_subdir(data_dir, "Dataset")
    final_root = _find_dataset_subdir(data_dir, "final_dataset")
    if dataset_root is None or final_root is None:
        raise RuntimeError(f"Expected `Dataset` and `final_dataset` directories inside {data_dir}.")
    return dataset_root, final_root


def _find_dataset_subdir(data_dir: Path, target_name: str) -> Path | None:
    """Locate canonical root containing class folders or train/val/test folders."""
    if not data_dir.exists():
        return None
    candidates = [p for p in data_dir.rglob(target_name) if p.is_dir()]
    if not candidates:
        return None
    # Prefer the deepest nested matching directory (handles data/Dataset/Dataset layout)
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    return candidates[0]


def _parse_timestamp_seconds(ts_str: str) -> int | None:
    """Convert YYYYMMDD_HHMMSS string to integer seconds within day + date offset."""
    try:
        import datetime as dt

        parsed = dt.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        return int(parsed.timestamp())
    except ValueError:
        return None


def _inspect_single_image(args: tuple[Path, str, str | None, Path]) -> dict[str, Any]:
    """Read image metadata, cryptographic hashes, perceptual hashes, and brightness/contrast."""
    path, cls_name, split_name, repo_root = args
    raw_bytes = path.read_bytes()
    md5_hex = hashlib.md5(raw_bytes).hexdigest()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

    try:
        rel_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel_path = path.as_posix()

    fname = path.name
    vid_match = VIDEO_FRAME_RE.match(fname)
    still_match = TIMESTAMP_STILL_RE.match(fname)

    if vid_match:
        source_type = "video_frame"
        session_id = vid_match.group(1)
        frame_idx = int(vid_match.group(2))
        timestamp_sec = _parse_timestamp_seconds(session_id)
    elif still_match:
        source_type = "mobile_still"
        session_id = f"{still_match.group(1)}_{still_match.group(2)}"
        frame_idx = -1
        timestamp_sec = _parse_timestamp_seconds(session_id)
    else:
        source_type = "other"
        session_id = ""
        frame_idx = -1
        timestamp_sec = None

    try:
        with Image.open(path) as img:
            raw_w, raw_h = img.size
            raw_mode = img.mode
            exif = img.getexif()
            exif_orientation = int(exif.get(274, 1)) if exif else 1
            img_upright = ImageOps.exif_transpose(img)
            width, height = img_upright.size
            upright_mode = img_upright.mode
            img_rgb = img_upright.convert("RGB")

            dhash_obj = imagehash.dhash(img_rgb)
            phash_obj = imagehash.phash(img_rgb)
            dhash_hex = str(dhash_obj)
            phash_hex = str(phash_obj)
            dhash_uint = int(dhash_hex, 16)
            phash_uint = int(phash_hex, 16)

            # Resize to 64x64 for fast, accurate luminance/contrast & 32x32 patch embedding
            arr64 = np.asarray(img_rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = 0.299 * arr64[:, :, 0] + 0.587 * arr64[:, :, 1] + 0.114 * arr64[:, :, 2]
            brightness_mean = float(gray.mean())
            contrast_std = float(gray.std())
            rgb_mean = arr64.mean(axis=(0, 1)).tolist()
            rgb_std = arr64.std(axis=(0, 1)).tolist()

            arr32 = np.asarray(img_rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
            emb = arr32 - arr32.mean()
            norm = float(np.linalg.norm(emb))
            emb = emb / (norm + 1e-8)

            return {
                "filepath": rel_path,
                "abs_path": str(path.resolve()),
                "filename": fname,
                "label": cls_name,
                "label_idx": CLASS_TO_IDX[cls_name],
                "orig_split": split_name,
                "corrupted": False,
                "md5": md5_hex,
                "sha256": sha256_hex,
                "raw_width": raw_w,
                "raw_height": raw_h,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "raw_mode": raw_mode,
                "mode": upright_mode,
                "exif_orientation": exif_orientation,
                "exif_rotated": exif_orientation != 1,
                "dhash_hex": dhash_hex,
                "phash_hex": phash_hex,
                "dhash_uint": dhash_uint,
                "phash_uint": phash_uint,
                "brightness": brightness_mean,
                "contrast": contrast_std,
                "rgb_mean": rgb_mean,
                "rgb_std": rgb_std,
                "source_type": source_type,
                "session_id": session_id,
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "emb": emb,
            }
    except Exception as exc:
        return {
            "filepath": rel_path,
            "abs_path": str(path.resolve()),
            "filename": fname,
            "label": cls_name,
            "label_idx": CLASS_TO_IDX[cls_name],
            "orig_split": split_name,
            "corrupted": True,
            "error": str(exc),
            "md5": md5_hex,
            "sha256": sha256_hex,
        }


def collect_image_tasks(root: Path, has_splits: bool, repo_root: Path) -> list[tuple[Path, str, str | None, Path]]:
    """Enumerate all image files under `Dataset/` or `final_dataset/`."""
    tasks: list[tuple[Path, str, str | None, Path]] = []
    if has_splits:
        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir() or split_dir.name.startswith("."):
                continue
            split_name = split_dir.name.lower()
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                    continue
                canonical_cls = CLASS_LOOKUP.get(cls_dir.name.lower())
                if canonical_cls is None:
                    continue
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        tasks.append((img_path, canonical_cls, split_name, repo_root))
    else:
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            canonical_cls = CLASS_LOOKUP.get(cls_dir.name.lower())
            if canonical_cls is None:
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    tasks.append((img_path, canonical_cls, None, repo_root))
    return tasks


def hamming_vec(anchor: np.uint64, arr: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit Hamming distance between scalar `anchor` and uint64 array `arr`."""
    if arr.size == 0:
        return np.empty(0, dtype=np.int32)
    xor_vals = np.bitwise_xor(anchor, arr)
    return np.unpackbits(xor_vals.view(np.uint8)).reshape(-1, 64).sum(axis=1).astype(np.int32)


class UnionFind:
    """Disjoint-set data structure for connected-component duplicate/burst clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def cluster_dataset(
    valid_records: list[dict[str, Any]],
    hamming_threshold: int = 6,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Cluster exact duplicates, perceptual near-duplicates, and sequential burst neighbors."""
    n = len(valid_records)
    uf = UnionFind(n)

    dh_arr = np.array([r["dhash_uint"] for r in valid_records], dtype=np.uint64)
    ph_arr = np.array([r["phash_uint"] for r in valid_records], dtype=np.uint64)
    embs = np.stack([r["emb"] for r in valid_records], axis=0)
    labels = np.array([r["label"] for r in valid_records])

    # Track statistics
    md5_groups: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(valid_records):
        md5_groups[rec["md5"]].append(idx)

    exact_dup_groups = {k: v for k, v in md5_groups.items() if len(v) > 1}
    for idxs in exact_dup_groups.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    near_dup_pairs_dhash = 0
    near_dup_pairs_phash = 0
    near_dup_pairs_union = 0
    emb_high_sim_pairs = 0
    burst_linked_pairs = 0
    cross_class_both_hashes = 0
    cross_class_single_hash_collisions = 0
    example_near_pairs: list[dict[str, Any]] = []
    near_per_cls: Counter[str] = Counter()

    for i in range(n - 1):
        d_dh = hamming_vec(dh_arr[i], dh_arr[i + 1 :])
        d_ph = hamming_vec(ph_arr[i], ph_arr[i + 1 :])
        cos_sim = embs[i + 1 :] @ embs[i]

        same_cls = labels[i + 1 :] == labels[i]
        diff_cls = ~same_cls

        # Audit cross-class collisions
        cross_both = diff_cls & (d_dh <= hamming_threshold) & (d_ph <= hamming_threshold)
        cross_any = diff_cls & ((d_dh <= hamming_threshold) | (d_ph <= hamming_threshold))
        cross_class_both_hashes += int(cross_both.sum())
        cross_class_single_hash_collisions += int(cross_any.sum())

        # Within-class perceptual near-duplicates (dHash <= threshold OR pHash <= threshold)
        m_dh = same_cls & (d_dh <= hamming_threshold)
        m_ph = same_cls & (d_ph <= hamming_threshold)
        m_emb = same_cls & (cos_sim >= 0.985) & ((d_dh <= hamming_threshold + 2) | (d_ph <= hamming_threshold + 2))
        m_union = m_dh | m_ph | m_emb

        near_dup_pairs_dhash += int(m_dh.sum())
        near_dup_pairs_phash += int(m_ph.sum())
        near_dup_pairs_union += int((m_dh | m_ph).sum())
        emb_high_sim_pairs += int(m_emb.sum())

        matched_offsets = np.where(m_union)[0]
        for off in matched_offsets:
            j = i + 1 + int(off)
            uf.union(i, j)
            cls_i = valid_records[i]["label"]
            if near_per_cls[cls_i] < 10 and (2 <= int(d_dh[off]) <= hamming_threshold):
                near_per_cls[cls_i] += 1
                example_near_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "label": cls_i,
                        "dhash_dist": int(d_dh[off]),
                        "phash_dist": int(d_ph[off]),
                        "cos_sim": float(cos_sim[off]),
                    }
                )

    # Sequential burst / frame continuity within same video clip or mobile burst
    by_session: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, rec in enumerate(valid_records):
        if rec["session_id"]:
            by_session[(rec["label"], rec["source_type"], rec["session_id"])].append(idx)

    for (cls_name, src_type, _), idxs in by_session.items():
        if src_type == "video_frame" and len(idxs) > 1:
            idxs_sorted = sorted(idxs, key=lambda k: valid_records[k]["frame_idx"])
            for pos in range(len(idxs_sorted) - 1):
                a, b = idxs_sorted[pos], idxs_sorted[pos + 1]
                frame_gap = abs(valid_records[b]["frame_idx"] - valid_records[a]["frame_idx"])
                d1 = int(hamming_vec(dh_arr[a], np.array([dh_arr[b]], dtype=np.uint64))[0])
                d2 = int(hamming_vec(ph_arr[a], np.array([ph_arr[b]], dtype=np.uint64))[0])
                cos_ab = float(embs[a] @ embs[b])
                if frame_gap <= 2 and (d1 <= hamming_threshold + 2 or d2 <= hamming_threshold + 2 or cos_ab >= 0.97):
                    if uf.find(a) != uf.find(b):
                        burst_linked_pairs += 1
                    uf.union(a, b)

    # Mobile still timestamp proximity (<= 10 seconds apart within same class & visually close)
    by_cls_stills: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(valid_records):
        if rec["source_type"] == "mobile_still" and rec["timestamp_sec"] is not None:
            by_cls_stills[rec["label"]].append(idx)

    for cls_name, idxs in by_cls_stills.items():
        idxs_sorted = sorted(idxs, key=lambda k: valid_records[k]["timestamp_sec"])
        for pos in range(len(idxs_sorted) - 1):
            a, b = idxs_sorted[pos], idxs_sorted[pos + 1]
            dt_sec = abs(valid_records[b]["timestamp_sec"] - valid_records[a]["timestamp_sec"])
            d1 = int(hamming_vec(dh_arr[a], np.array([dh_arr[b]], dtype=np.uint64))[0])
            d2 = int(hamming_vec(ph_arr[a], np.array([ph_arr[b]], dtype=np.uint64))[0])
            if dt_sec <= 10 and (d1 <= hamming_threshold + 2 or d2 <= hamming_threshold + 2):
                if uf.find(a) != uf.find(b):
                    burst_linked_pairs += 1
                uf.union(a, b)

    raw_roots = [uf.find(i) for i in range(n)]
    unique_roots = sorted(set(raw_roots))
    root_to_cid = {r: cid for cid, r in enumerate(unique_roots)}
    cluster_ids = np.array([root_to_cid[r] for r in raw_roots], dtype=np.int32)

    cluster_sizes = Counter(cluster_ids.tolist())
    multi_clusters = [sz for sz in cluster_sizes.values() if sz > 1]

    cluster_stats = {
        "hamming_threshold": hamming_threshold,
        "exact_duplicate_groups_in_dataset": len(exact_dup_groups),
        "exact_duplicate_extra_files_in_dataset": sum(len(v) - 1 for v in exact_dup_groups.values()),
        "near_duplicate_pairs_dhash_le_threshold": near_dup_pairs_dhash,
        "near_duplicate_pairs_phash_le_threshold": near_dup_pairs_phash,
        "near_duplicate_pairs_dhash_or_phash": near_dup_pairs_union,
        "high_cosine_similarity_pairs": emb_high_sim_pairs,
        "additional_temporal_burst_links": burst_linked_pairs,
        "cross_class_true_near_duplicates_both_hashes": cross_class_both_hashes,
        "cross_class_boundary_single_hash_collisions": cross_class_single_hash_collisions,
        "total_clusters": len(cluster_sizes),
        "multi_image_clusters": len(multi_clusters),
        "singleton_clusters": sum(1 for sz in cluster_sizes.values() if sz == 1),
        "images_in_multi_image_clusters": sum(multi_clusters),
        "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "median_cluster_size": float(np.median(list(cluster_sizes.values()))) if cluster_sizes else 0.0,
        "top_10_cluster_sizes": sorted(cluster_sizes.values(), reverse=True)[:10],
    }
    return cluster_ids, cluster_stats, example_near_pairs


def audit_final_dataset_leakage(
    dataset_records: list[dict[str, Any]],
    cluster_ids: np.ndarray,
    final_records: list[dict[str, Any]],
    hamming_threshold: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Quantify exact, perceptual near-duplicate, and cluster leakage in `final_dataset/`."""
    ds_md5_to_idx = {r["md5"]: idx for idx, r in enumerate(dataset_records)}
    dh_arr = np.array([r["dhash_uint"] for r in dataset_records], dtype=np.uint64)
    ph_arr = np.array([r["phash_uint"] for r in dataset_records], dtype=np.uint64)
    ds_labels = np.array([r["label"] for r in dataset_records])

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in final_records:
        if not rec.get("corrupted", False):
            by_split[rec["orig_split"]].append(rec)

    split_counts = {sp: len(items) for sp, items in sorted(by_split.items())}
    split_class_counts = {
        sp: dict(Counter(r["label"] for r in items))
        for sp, items in sorted(by_split.items())
    }

    all_final_md5 = {r["md5"] for r in final_records if not r.get("corrupted", False)}
    ds_md5_set = set(ds_md5_to_idx.keys())

    train_recs = by_split.get("train", [])
    val_recs = by_split.get("val", [])
    test_recs = by_split.get("test", [])

    train_md5_to_rec = {r["md5"]: r for r in train_recs}
    train_md5_set = set(train_md5_to_rec.keys())
    val_md5_set = {r["md5"] for r in val_recs}
    test_md5_set = {r["md5"] for r in test_recs}

    train_ds_idxs = np.array(
        sorted({ds_md5_to_idx[m] for m in train_md5_set if m in ds_md5_to_idx}),
        dtype=np.int32,
    )
    train_clusters = {int(cluster_ids[i]) for i in train_ds_idxs}
    train_videos = {
        (r["label"], r["session_id"])
        for r in train_recs
        if r["source_type"] == "video_frame"
    }

    exact_leak_examples: list[dict[str, Any]] = []
    exact_per_cls: Counter[str] = Counter()

    def _audit_eval_split(eval_recs: list[dict[str, Any]], split_label: str) -> dict[str, Any]:
        exact_count = 0
        near_only_count = 0
        cluster_leak_count = 0
        same_video_count = 0
        video_total = 0
        per_class_leak: dict[str, dict[str, Any]] = {
            c: {"total": 0, "exact": 0, "near_only": 0, "leaked_total": 0}
            for c in CANONICAL_CLASSES
        }

        for r in eval_recs:
            cls_name = r["label"]
            per_class_leak[cls_name]["total"] += 1
            if r["source_type"] == "video_frame":
                video_total += 1
                if (cls_name, r["session_id"]) in train_videos:
                    same_video_count += 1

            is_exact = r["md5"] in train_md5_set
            is_near = False
            if is_exact:
                exact_count += 1
                per_class_leak[cls_name]["exact"] += 1
                if split_label == "test" and exact_per_cls[cls_name] < 5:
                    exact_per_cls[cls_name] += 1
                    exact_leak_examples.append(
                        {
                            "label": cls_name,
                            "train_path": train_md5_to_rec[r["md5"]]["filepath"],
                            "test_path": r["filepath"],
                            "filename": r["filename"],
                            "md5": r["md5"],
                        }
                    )
            else:
                same_cls_train = train_ds_idxs[ds_labels[train_ds_idxs] == cls_name]
                d1 = hamming_vec(np.uint64(r["dhash_uint"]), dh_arr[same_cls_train])
                d2 = hamming_vec(np.uint64(r["phash_uint"]), ph_arr[same_cls_train])
                if np.any((d1 <= hamming_threshold) | (d2 <= hamming_threshold)):
                    is_near = True
                    near_only_count += 1
                    per_class_leak[cls_name]["near_only"] += 1

            if is_exact or is_near:
                per_class_leak[cls_name]["leaked_total"] += 1

            ds_idx = ds_md5_to_idx.get(r["md5"])
            if ds_idx is not None and int(cluster_ids[ds_idx]) in train_clusters:
                cluster_leak_count += 1

        for c in CANONICAL_CLASSES:
            tot = per_class_leak[c]["total"]
            lk = per_class_leak[c]["leaked_total"]
            per_class_leak[c]["leakage_pct"] = round((lk / tot * 100.0) if tot else 0.0, 2)

        total_n = len(eval_recs)
        leaked_total = exact_count + near_only_count
        return {
            "total_images": total_n,
            "exact_md5_in_train": exact_count,
            "exact_md5_in_train_pct": round(exact_count / total_n * 100.0, 2) if total_n else 0.0,
            "near_duplicate_only_in_train": near_only_count,
            "near_duplicate_only_in_train_pct": round(near_only_count / total_n * 100.0, 2) if total_n else 0.0,
            "total_exact_or_near_leaked": leaked_total,
            "total_exact_or_near_leaked_pct": round(leaked_total / total_n * 100.0, 2) if total_n else 0.0,
            "cluster_leaked_in_train": cluster_leak_count,
            "cluster_leaked_in_train_pct": round(cluster_leak_count / total_n * 100.0, 2) if total_n else 0.0,
            "video_frames_total": video_total,
            "video_frames_from_train_videos": same_video_count,
            "video_frames_from_train_videos_pct": round(same_video_count / video_total * 100.0, 2) if video_total else 0.0,
            "per_class": per_class_leak,
        }

    test_audit = _audit_eval_split(test_recs, "test")
    val_audit = _audit_eval_split(val_recs, "val")

    leakage_summary = {
        "final_dataset_total_files": len(final_records),
        "final_dataset_unique_md5": len(all_final_md5),
        "redundant_duplicate_files_across_splits": len(final_records) - len(all_final_md5),
        "final_dataset_files_present_in_raw_dataset": len(all_final_md5 & ds_md5_set),
        "final_dataset_split_counts": split_counts,
        "final_dataset_split_class_counts": split_class_counts,
        "val_test_exact_md5_overlap": len(val_md5_set & test_md5_set),
        "test_vs_train": test_audit,
        "val_vs_train": val_audit,
    }
    return leakage_summary, exact_leak_examples


def build_leak_free_splits(
    records: list[dict[str, Any]],
    cluster_ids: np.ndarray,
    seed: int = 42,
    hamming_threshold: int = 6,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build 70/15/15 StratifiedGroupKFold split and verify zero leakage."""
    n = len(records)
    labels = np.array([r["label"] for r in records])

    # 20 stratified group folds -> each fold is ~5% of data; 14 train (70%), 3 val (15%), 3 test (15%)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_ids = np.zeros(n, dtype=np.int32)
    for fold_idx, (_, hold_idx) in enumerate(sgkf.split(np.zeros(n), labels, cluster_ids)):
        fold_ids[hold_idx] = fold_idx

    # Deterministically select disjoint 3-fold test (~15%) and 3-fold val (~15%) closest to 0.15/0.15
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_val_folds = (0, 1, 2)
    best_test_folds = (3, 4, 5)
    class_totals = {c: int((labels == c).sum()) for c in CANONICAL_CLASSES}

    all_folds = np.arange(20)
    for _ in range(2000):
        perm = rng.permutation(all_folds)
        vf = tuple(sorted(int(x) for x in perm[:3]))
        tf = tuple(sorted(int(x) for x in perm[3:6]))
        v_mask = np.isin(fold_ids, vf)
        t_mask = np.isin(fold_ids, tf)

        overall_err = abs(v_mask.mean() - 0.15) * 3.0 + abs(t_mask.mean() - 0.15) * 3.0
        cls_err = 0.0
        valid = True
        for c in CANONICAL_CLASSES:
            c_mask = labels == c
            vr = (v_mask & c_mask).sum() / class_totals[c]
            tr = (t_mask & c_mask).sum() / class_totals[c]
            if vr < 0.09 or tr < 0.09:
                valid = False
                break
            cls_err += abs(vr - 0.15) + abs(tr - 0.15)
        if valid and (overall_err + cls_err) < best_score:
            best_score = overall_err + cls_err
            best_val_folds = vf
            best_test_folds = tf

    val_mask = np.isin(fold_ids, best_val_folds)
    test_mask = np.isin(fold_ids, best_test_folds)
    train_mask = ~(val_mask | test_mask)

    # Hard verification gates: zero MD5, zero cluster, zero within-class perceptual leakage!
    split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        m1, m2 = split_masks[s1], split_masks[s2]
        c1 = set(cluster_ids[m1].tolist())
        c2 = set(cluster_ids[m2].tolist())
        assert len(c1 & c2) == 0, f"Cluster leakage detected between {s1} and {s2}!"
        md5_1 = {records[i]["md5"] for i in np.where(m1)[0]}
        md5_2 = {records[i]["md5"] for i in np.where(m2)[0]}
        assert len(md5_1 & md5_2) == 0, f"MD5 leakage detected between {s1} and {s2}!"

    dh_arr = np.array([r["dhash_uint"] for r in records], dtype=np.uint64)
    ph_arr = np.array([r["phash_uint"] for r in records], dtype=np.uint64)
    cross_split_near_dups = 0
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        idx1 = np.where(split_masks[s1])[0]
        idx2 = np.where(split_masks[s2])[0]
        for i in idx1:
            same_c = idx2[labels[idx2] == labels[i]]
            d1 = hamming_vec(dh_arr[i], dh_arr[same_c])
            d2 = hamming_vec(ph_arr[i], ph_arr[same_c])
            cross_split_near_dups += int(np.any((d1 <= hamming_threshold) | (d2 <= hamming_threshold)))
    assert cross_split_near_dups == 0, f"Perceptual near-dup leakage detected: {cross_split_near_dups}!"

    dfs: dict[str, pd.DataFrame] = {}
    split_summary: dict[str, Any] = {
        "method": "StratifiedGroupKFold (20 folds -> 14 train / 3 val / 3 test)",
        "seed": seed,
        "val_folds": list(best_val_folds),
        "test_folds": list(best_test_folds),
        "verified_exact_md5_leakage": 0,
        "verified_cluster_leakage": 0,
        "verified_perceptual_near_dup_leakage": cross_split_near_dups,
        "splits": {},
    }

    cols = [
        "filepath",
        "filename",
        "label",
        "label_idx",
        "cluster_id",
        "md5",
        "width",
        "height",
        "source_type",
    ]
    for sp_name, mask in split_masks.items():
        rows = []
        for idx in np.where(mask)[0]:
            rec = records[idx]
            rows.append(
                {
                    "filepath": rec["filepath"],
                    "filename": rec["filename"],
                    "label": rec["label"],
                    "label_idx": rec["label_idx"],
                    "cluster_id": int(cluster_ids[idx]),
                    "md5": rec["md5"],
                    "width": rec["width"],
                    "height": rec["height"],
                    "source_type": rec["source_type"],
                }
            )
        df = pd.DataFrame(rows, columns=cols).sort_values(["label", "filename"]).reset_index(drop=True)
        dfs[sp_name] = df
        cls_counts = {c: int((df["label"] == c).sum()) for c in CANONICAL_CLASSES}
        split_summary["splits"][sp_name] = {
            "count": len(df),
            "ratio_pct": round(len(df) / n * 100.0, 2),
            "unique_clusters": int(df["cluster_id"].nunique()),
            "class_counts": cls_counts,
        }

    return dfs, split_summary


def plot_class_distribution(
    records: list[dict[str, Any]],
    leakage_summary: dict[str, Any],
    split_summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Render `assets/class_distribution.png` comparing raw data, leaked split, and clean split."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=180)

    x = np.arange(len(CANONICAL_CLASSES))

    # Panel A: Raw Dataset class imbalance & capture mode
    vid_counts = [
        sum(1 for r in records if r["label"] == c and r["source_type"] == "video_frame")
        for c in CANONICAL_CLASSES
    ]
    still_counts = [
        sum(1 for r in records if r["label"] == c and r["source_type"] != "video_frame")
        for c in CANONICAL_CLASSES
    ]
    ax = axes[0]
    b1 = ax.bar(x, vid_counts, width=0.62, label="Video frames (200x200)", color="#2b6cb0")
    b2 = ax.bar(x, still_counts, width=0.62, bottom=vid_counts, label="Mobile stills (10MP/12MP)", color="#ed8936")
    for idx, tot in enumerate(np.array(vid_counts) + np.array(still_counts)):
        ax.text(idx, tot + 22, f"{tot}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_CLASSES, rotation=22, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(vid_counts) + 210)
    ax.set_title("A. Raw Dataset/ Class & Source Distribution (N=3,879)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: Audit of final_dataset/test leakage into final_dataset/train
    ax = axes[1]
    per_cls_leak = leakage_summary["test_vs_train"]["per_class"]
    exact_vals = [per_cls_leak[c]["exact"] for c in CANONICAL_CLASSES]
    near_vals = [per_cls_leak[c]["near_only"] for c in CANONICAL_CLASSES]
    clean_vals = [
        per_cls_leak[c]["total"] - per_cls_leak[c]["leaked_total"] for c in CANONICAL_CLASSES
    ]
    ax.bar(x, exact_vals, width=0.62, label="Exact MD5 in train", color="#c53030")
    ax.bar(x, near_vals, width=0.62, bottom=exact_vals, label="Near-dup (dHash/pHash<=6) in train", color="#dd6b20")
    ax.bar(
        x,
        clean_vals,
        width=0.62,
        bottom=np.array(exact_vals) + np.array(near_vals),
        label="Unmatched in train",
        color="#38a169",
    )
    for idx, c in enumerate(CANONICAL_CLASSES):
        tot = per_cls_leak[c]["total"]
        pct = per_cls_leak[c]["leakage_pct"]
        ax.text(idx, tot + 5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8.5, color="#9b2c2c", fontweight="bold")
    tot_pct = leakage_summary["test_vs_train"]["total_exact_or_near_leaked_pct"]
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_CLASSES, rotation=22, ha="right")
    ax.set_ylabel("Images in final_dataset/test")
    ax.set_ylim(0, max(per_cls_leak[c]["total"] for c in CANONICAL_CLASSES) + 35)
    ax.set_title(f"B. final_dataset/test Leakage Audit ({tot_pct:.1f}% Leaked!)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Leak-free StratifiedGroupKFold split
    ax = axes[2]
    w = 0.26
    tr_c = [split_summary["splits"]["train"]["class_counts"][c] for c in CANONICAL_CLASSES]
    va_c = [split_summary["splits"]["val"]["class_counts"][c] for c in CANONICAL_CLASSES]
    te_c = [split_summary["splits"]["test"]["class_counts"][c] for c in CANONICAL_CLASSES]
    ax.bar(x - w, tr_c, width=w, label=f"Train ({split_summary['splits']['train']['ratio_pct']:.1f}%)", color="#2b6cb0")
    ax.bar(x, va_c, width=w, label=f"Val ({split_summary['splits']['val']['ratio_pct']:.1f}%)", color="#319795")
    ax.bar(x + w, te_c, width=w, label=f"Test ({split_summary['splits']['test']['ratio_pct']:.1f}%)", color="#805ad5")
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_CLASSES, rotation=22, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("C. Leak-Free StratifiedGroupKFold Split (0.0% Leakage)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_duplicates_examples(
    records: list[dict[str, Any]],
    cluster_ids: np.ndarray,
    exact_leak_examples: list[dict[str, Any]],
    near_pair_examples: list[dict[str, Any]],
    repo_root: Path,
    out_path: Path,
) -> None:
    """Render `assets/duplicates_examples.png` showcasing cross-split exact & burst near-duplicates."""
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2), dpi=180)

    def _load_thumb(rel_or_abs: str) -> np.ndarray:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = repo_root / p
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (224, 224), method=Image.Resampling.BILINEAR)
            return np.asarray(im)

    # Pick 2 distinct classes for Row 0 (exact cross-split leaks in final_dataset)
    chosen_exact: list[dict[str, Any]] = []
    seen_cls: set[str] = set()
    for ex in exact_leak_examples:
        if ex["label"] not in seen_cls:
            chosen_exact.append(ex)
            seen_cls.add(ex["label"])
        if len(chosen_exact) == 2:
            break

    for pair_idx, ex in enumerate(chosen_exact):
        c_left = pair_idx * 2
        c_right = c_left + 1
        img_tr = _load_thumb(ex["train_path"])
        img_te = _load_thumb(ex["test_path"])
        md5_short = ex["md5"][:10]

        axes[0, c_left].imshow(img_tr)
        axes[0, c_left].set_title(
            f"final_dataset/train ({ex['label']})\n{ex['filename']}\nMD5: {md5_short}...",
            fontsize=8.5,
            color="#9b2c2c",
            fontweight="bold",
        )
        axes[0, c_left].axis("off")

        axes[0, c_right].imshow(img_te)
        axes[0, c_right].set_title(
            f"final_dataset/test (EXACT LEAK)\n{ex['filename']}\nMD5: {md5_short}... (dHash=0)",
            fontsize=8.5,
            color="#9b2c2c",
            fontweight="bold",
        )
        axes[0, c_right].axis("off")

    # Pick 2 distinct classes for Row 1 (perceptual near-duplicates / video burst frames)
    chosen_near: list[dict[str, Any]] = []
    seen_near_cls: set[str] = set()
    for npair in sorted(near_pair_examples, key=lambda x: (x["dhash_dist"] == 0, abs(x["dhash_dist"] - 3))):
        if npair["label"] not in seen_near_cls and npair["dhash_dist"] >= 2:
            chosen_near.append(npair)
            seen_near_cls.add(npair["label"])
        if len(chosen_near) == 2:
            break

    for pair_idx, npair in enumerate(chosen_near):
        c_left = pair_idx * 2
        c_right = c_left + 1
        rec_i = records[npair["i"]]
        rec_j = records[npair["j"]]
        cid = int(cluster_ids[npair["i"]])

        axes[1, c_left].imshow(_load_thumb(rec_i["abs_path"]))
        axes[1, c_left].set_title(
            f"Burst / Near-Dup A ({rec_i['label']})\n{rec_i['filename']}\nAssigned Cluster #{cid}",
            fontsize=8.5,
            color="#2b6cb0",
            fontweight="bold",
        )
        axes[1, c_left].axis("off")

        axes[1, c_right].imshow(_load_thumb(rec_j["abs_path"]))
        axes[1, c_right].set_title(
            f"Burst / Near-Dup B ({rec_j['label']})\n{rec_j['filename']}\ndHash={npair['dhash_dist']}, pHash={npair['phash_dist']} -> Cluster #{cid}",
            fontsize=8.5,
            color="#2b6cb0",
            fontweight="bold",
        )
        axes[1, c_right].axis("off")

    fig.suptitle(
        "Forensic Audit: Cross-Split Exact Leaks in final_dataset/ (Top) vs Grouped Near-Duplicates in Dataset/ (Bottom)",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_eda_distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize class balance, resolutions, EXIF tags, luminance/contrast, and sequence stats."""
    class_counts = {c: sum(1 for r in records if r["label"] == c) for c in CANONICAL_CLASSES}
    max_cls = max(class_counts, key=class_counts.get)
    min_cls = min(class_counts, key=class_counts.get)

    raw_res_counts = Counter(f"{r['raw_width']}x{r['raw_height']}" for r in records)
    upright_res_counts = Counter(f"{r['width']}x{r['height']}" for r in records)
    aspect_counts = Counter(str(round(r["width"] / r["height"], 2)) for r in records)
    mode_counts = Counter(r["mode"] for r in records)
    source_counts = Counter(r["source_type"] for r in records)

    exif_rotated_total = sum(1 for r in records if r["exif_rotated"])
    stills_total = sum(1 for r in records if r["source_type"] == "mobile_still")
    exif_orientations = Counter(int(r["exif_orientation"]) for r in records)

    brightness_all = np.array([r["brightness"] for r in records], dtype=np.float64)
    contrast_all = np.array([r["contrast"] for r in records], dtype=np.float64)
    rgb_mean_all = np.mean([r["rgb_mean"] for r in records], axis=0) / 255.0
    rgb_std_all = np.mean([r["rgb_std"] for r in records], axis=0) / 255.0

    per_class_visual: dict[str, Any] = {}
    for c in CANONICAL_CLASSES:
        sub = [r for r in records if r["label"] == c]
        b_vals = [r["brightness"] for r in sub]
        c_vals = [r["contrast"] for r in sub]
        vids = {r["session_id"] for r in sub if r["source_type"] == "video_frame"}
        per_class_visual[c] = {
            "count": len(sub),
            "pct": round(len(sub) / len(records) * 100.0, 2),
            "video_frames": sum(1 for r in sub if r["source_type"] == "video_frame"),
            "mobile_stills": sum(1 for r in sub if r["source_type"] == "mobile_still"),
            "unique_video_clips": len(vids),
            "brightness_mean": round(float(np.mean(b_vals)), 2),
            "brightness_std": round(float(np.std(b_vals)), 2),
            "contrast_mean": round(float(np.mean(c_vals)), 2),
            "contrast_std": round(float(np.std(c_vals)), 2),
        }

    # Quantify consecutive frame similarity vs random same-class similarity
    by_vid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["source_type"] == "video_frame":
            by_vid[(r["label"], r["session_id"])].append(r)

    consec_dhash: list[int] = []
    consec_phash: list[int] = []
    for _, items in by_vid.items():
        items_sorted = sorted(items, key=lambda x: x["frame_idx"])
        for k in range(len(items_sorted) - 1):
            d_dh = bin(items_sorted[k]["dhash_uint"] ^ items_sorted[k + 1]["dhash_uint"]).count("1")
            d_ph = bin(items_sorted[k]["phash_uint"] ^ items_sorted[k + 1]["phash_uint"]).count("1")
            consec_dhash.append(d_dh)
            consec_phash.append(d_ph)

    rng = np.random.default_rng(42)
    rand_dhash: list[int] = []
    rand_phash: list[int] = []
    for c in CANONICAL_CLASSES:
        sub = [r for r in records if r["label"] == c]
        for _ in range(300):
            i, j = rng.choice(len(sub), size=2, replace=False)
            rand_dhash.append(bin(sub[i]["dhash_uint"] ^ sub[j]["dhash_uint"]).count("1"))
            rand_phash.append(bin(sub[i]["phash_uint"] ^ sub[j]["phash_uint"]).count("1"))

    return {
        "total_images": len(records),
        "class_counts": class_counts,
        "imbalance_ratio_max_to_min": round(class_counts[max_cls] / class_counts[min_cls], 2),
        "majority_class": {"name": max_cls, "count": class_counts[max_cls]},
        "minority_class": {"name": min_cls, "count": class_counts[min_cls]},
        "raw_resolutions": dict(raw_res_counts.most_common()),
        "exif_transposed_resolutions": dict(upright_res_counts.most_common()),
        "aspect_ratios_upright": dict(aspect_counts.most_common()),
        "color_modes": dict(mode_counts),
        "exif_rotated_images": exif_rotated_total,
        "mobile_stills_total": stills_total,
        "exif_rotated_among_stills_pct": round(exif_rotated_total / stills_total * 100.0, 2) if stills_total else 0.0,
        "exif_orientation_tag_counts": {str(k): v for k, v in sorted(exif_orientations.items())},
        "source_type_counts": dict(source_counts),
        "total_unique_video_clips": len(by_vid),
        "sequential_frame_analysis": {
            "consecutive_frame_dhash_median": float(np.median(consec_dhash)),
            "consecutive_frame_dhash_mean": round(float(np.mean(consec_dhash)), 2),
            "consecutive_frame_phash_median": float(np.median(consec_phash)),
            "random_same_class_dhash_median": float(np.median(rand_dhash)),
            "random_same_class_dhash_mean": round(float(np.mean(rand_dhash)), 2),
            "random_same_class_phash_median": float(np.median(rand_phash)),
        },
        "global_brightness_mean": round(float(brightness_all.mean()), 2),
        "global_brightness_std": round(float(brightness_all.std()), 2),
        "global_contrast_mean": round(float(contrast_all.mean()), 2),
        "global_contrast_std": round(float(contrast_all.std()), 2),
        "global_rgb_mean_0_1": [round(float(x), 4) for x in rgb_mean_all],
        "global_rgb_std_0_1": [round(float(x), 4) for x in rgb_std_all],
        "per_class_details": per_class_visual,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 EDA, leakage audit, and leak-free split builder.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("splits"), help="Output directory for split CSVs.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"), help="Output directory for EDA figures.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Output directory for JSON summaries.")
    parser.add_argument("--hamming-threshold", type=int, default=6, help="Max dHash/pHash Hamming distance for near-duplicates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for StratifiedGroupKFold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()

    dataset_root, final_root = ensure_dataset_present(args.data_dir)
    print(f"[INFO] Raw Dataset root:   {dataset_root}")
    print(f"[INFO] final_dataset root: {final_root}")

    ds_tasks = collect_image_tasks(dataset_root, has_splits=False, repo_root=repo_root)
    final_tasks = collect_image_tasks(final_root, has_splits=True, repo_root=repo_root)
    print(f"[INFO] Scanning {len(ds_tasks)} images in Dataset/ and {len(final_tasks)} files in final_dataset/...")

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        ds_records_all = list(pool.map(_inspect_single_image, ds_tasks))
        final_records_all = list(pool.map(_inspect_single_image, final_tasks))

    corrupted_ds = [r for r in ds_records_all if r.get("corrupted", False)]
    corrupted_final = [r for r in final_records_all if r.get("corrupted", False)]
    valid_ds = [r for r in ds_records_all if not r.get("corrupted", False)]

    # Deduplicate exact MD5 within Dataset/ if any exist
    seen_md5: set[str] = set()
    dedup_ds: list[dict[str, Any]] = []
    for r in valid_ds:
        if r["md5"] not in seen_md5:
            seen_md5.add(r["md5"])
            dedup_ds.append(r)

    eda_dist = compute_eda_distributions(valid_ds)
    cluster_ids, cluster_stats, near_examples = cluster_dataset(
        dedup_ds, hamming_threshold=args.hamming_threshold
    )
    leakage_summary, exact_leak_examples = audit_final_dataset_leakage(
        dedup_ds, cluster_ids, final_records_all, hamming_threshold=args.hamming_threshold
    )
    split_dfs, split_summary = build_leak_free_splits(
        dedup_ds, cluster_ids, seed=args.seed, hamming_threshold=args.hamming_threshold
    )

    # Save CSV splits
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sp_name, df in split_dfs.items():
        csv_path = args.out_dir / f"{sp_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Wrote {csv_path} ({len(df)} rows, {df['cluster_id'].nunique()} clusters)")

    # Save figures
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    dist_plot = args.assets_dir / "class_distribution.png"
    dup_plot = args.assets_dir / "duplicates_examples.png"
    plot_class_distribution(valid_ds, leakage_summary, split_summary, dist_plot)
    plot_duplicates_examples(
        dedup_ds, cluster_ids, exact_leak_examples, near_examples, repo_root, dup_plot
    )
    print(f"[OK] Saved plots: {dist_plot}, {dup_plot}")

    # Save JSON summary
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "canonical_classes": CANONICAL_CLASSES,
        "corrupted_files": {
            "dataset": len(corrupted_ds),
            "final_dataset": len(corrupted_final),
        },
        "dataset_eda": eda_dist,
        "clustering": cluster_stats,
        "final_dataset_leakage_audit": leakage_summary,
        "clean_split": split_summary,
    }
    summary_path = args.artifacts_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved EDA & leakage summary: {summary_path}")

    # Print concise audit summary to stdout
    tvt = leakage_summary["test_vs_train"]
    print("\n=== PHASE 1 GATE VERIFICATION SUMMARY ===")
    print(f"Raw Dataset images:           {eda_dist['total_images']} (imbalance ratio {eda_dist['imbalance_ratio_max_to_min']}x)")
    print(f"final_dataset total files:    {leakage_summary['final_dataset_total_files']} ({leakage_summary['final_dataset_unique_md5']} unique MD5)")
    print(
        f"final_dataset/test leakage:   {tvt['total_exact_or_near_leaked']}/{tvt['total_images']} "
        f"({tvt['total_exact_or_near_leaked_pct']}%) -> {tvt['exact_md5_in_train']} exact MD5 + "
        f"{tvt['near_duplicate_only_in_train']} perceptual near-dups"
    )
    print(f"Connected clusters built:     {cluster_stats['total_clusters']} ({cluster_stats['multi_image_clusters']} multi-image)")
    for sp in ("train", "val", "test"):
        sp_info = split_summary["splits"][sp]
        print(f"Clean {sp:5s} split:            {sp_info['count']:4d} images ({sp_info['ratio_pct']:5.2f}%), {sp_info['unique_clusters']:4d} clusters")
    print("Verified cross-split leakage: 0 exact MD5, 0 clusters, 0 near-duplicates (dHash/pHash <= 6)")


if __name__ == "__main__":
    main()
