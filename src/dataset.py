"""Dataset and store-realistic augmentation pipeline for Magnum shelf classification.

Implements defensive EXIF-aware image loading, deterministic evaluation preprocessing
identical to `predict.py` (shortest-edge resize to 256 -> 224x224 center crop -> ImageNet
normalization), and supermarket-domain training augmentations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CANONICAL_CLASSES = [
    "Capsicum",
    "Carrot",
    "Cucumber",
    "Potato",
    "Raddish",
    "Squash",
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_exif_safe(path: Path | str) -> Image.Image:
    """Load an image, apply EXIF orientation transpose, and convert to 3-channel RGB.

    Mobile store cameras frequently record portrait shots with landscape sensor buffers
    and EXIF Orientation tags (175/185 mobile stills in this dataset carry non-default
    EXIF orientation). Transposing before RGB conversion prevents sideways inputs.
    """
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        return upright.convert("RGB")


def deterministic_eval_preprocess_pil(
    img: Image.Image,
    resize_short: int = 256,
    crop_size: int = 224,
) -> np.ndarray:
    """Deterministic PIL + NumPy preprocessing shared identically between eval and `predict.py`.

    1. Resizes shortest side to `resize_short` (256) preserving aspect ratio (BILINEAR).
    2. Extracts centered `crop_size x crop_size` (224x224) crop.
    3. Scales to [0, 1] float32 and normalizes with ImageNet mean/std -> shape (3, H, W).
    """
    w, h = img.size
    if w <= h:
        new_w = resize_short
        new_h = int(round(h * (resize_short / w)))
    else:
        new_h = resize_short
        new_w = int(round(w * (resize_short / h)))

    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = int(round((new_w - crop_size) / 2.0))
    top = int(round((new_h - crop_size) / 2.0))
    img = img.crop((left, top, left + crop_size, top + crop_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def build_store_train_transforms(img_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Build store-domain training augmentations simulating Magnum retail shelf conditions.

    Domain-shift rationale for each augmentation:
    - `RandomResizedCrop(224, scale=(0.65, 1.0))`: Simulates varying merchandiser camera
      distance to shelf crates and partial framing where bin edges crop produce.
    - `RandomHorizontalFlip(p=0.5)` & `RandomVerticalFlip(p=0.5)`: Produce in supermarket
      bins is piled in arbitrary 360-degree orientations with no canonical "up" direction.
    - `RandomAffine(degrees=25, translate=(0.08, 0.08), shear=10)`: Captures off-axis
      handheld smartphone angles and perspective skew along aisle shelving.
    - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05)`: Bridges the
      lighting gap between home flash photos and harsh overhead fluorescent vs warm retail LEDs,
      as well as shadow gradients inside deep shelf crates.
    - `GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))`: Simulates handheld motion blur and
      slight autofocus miss during rapid aisle audits.
    - `RandomErasing(p=0.25, scale=(0.02, 0.15))`: Simulates partial occlusions from Magnum
      price tags, barcode stickers, polythene bag glare, crate slats, and neighboring produce.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(0.65, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=25,
                translate=(0.08, 0.08),
                shear=10,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.25,
                hue=0.05,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.35,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_eval_transforms(img_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic evaluation transform matching `predict.py` byte-for-byte."""
    resize_short = int(round(img_size * 256 / 224))

    def _transform(img: Image.Image) -> torch.Tensor:
        chw = deterministic_eval_preprocess_pil(img, resize_short=resize_short, crop_size=img_size)
        return torch.from_numpy(chw)

    return _transform


class VegetableDataset(Dataset):
    """CSV-backed dataset reading leak-free `splits/{train,val,test}.csv`."""

    def __init__(
        self,
        csv_path: Path | str,
        repo_root: Path | str = Path("."),
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        cache_images: bool = False,
        max_cache_short_side: int = 512,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.repo_root = Path(repo_root)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform if transform is not None else build_eval_transforms(224)
        self.cache_images = cache_images
        self._cache: dict[int, Image.Image] = {}

        if self.cache_images:
            for idx in range(len(self.df)):
                img = self._read_disk_image(idx)
                w, h = img.size
                short = min(w, h)
                if short > max_cache_short_side:
                    scale = max_cache_short_side / short
                    img = img.resize(
                        (int(round(w * scale)), int(round(h * scale))),
                        Image.Resampling.BILINEAR,
                    )
                self._cache[idx] = img

    def _read_disk_image(self, idx: int) -> Image.Image:
        row = self.df.iloc[idx]
        rel_path = str(row["filepath"])
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = self.repo_root / img_path
        return load_rgb_exif_safe(img_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        rel_path = str(row["filepath"])
        img = self._cache[idx] if idx in self._cache else self._read_disk_image(idx)
        tensor = self.transform(img)
        label_idx = int(row["label_idx"])
        return tensor, label_idx, rel_path
