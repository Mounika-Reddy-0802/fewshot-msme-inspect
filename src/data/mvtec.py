"""MVTec AD on-disk layout.

Every category is laid out the same way:

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type>/*.png      defect_type "good" == normal
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

Only anomalous test images have masks, which is what makes pixel-level metrics
possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)

NORMAL = "good"


@dataclass(frozen=True)
class Sample:
    """One test image, with its mask when it has one."""

    image: Path
    label: int          # 0 normal, 1 anomalous
    defect_type: str    # "good", or the defect folder name
    mask: Path | None   # None for normal images


def category_dir(root: str | Path, category: str) -> Path:
    path = Path(root) / category
    if not path.is_dir():
        raise FileNotFoundError(
            f"no category {category!r} under {root} -- "
            f"run scripts/download_mvtec_category.py --category {category}"
        )
    return path


def list_train_normals(root: str | Path, category: str) -> list[Path]:
    """The normal-only training images. Sorted, so ordering never depends on the
    filesystem."""
    train_dir = category_dir(root, category) / "train" / NORMAL
    return sorted(train_dir.glob("*.png"))


def list_test_samples(root: str | Path, category: str) -> list[Sample]:
    """The full test split: normals and every defect type, with masks attached."""
    base = category_dir(root, category)
    test_dir = base / "test"
    ground_truth = base / "ground_truth"

    samples: list[Sample] = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        is_normal = defect_type == NORMAL
        for image in sorted(defect_dir.glob("*.png")):
            mask = None
            if not is_normal:
                candidate = ground_truth / defect_type / f"{image.stem}_mask.png"
                mask = candidate if candidate.exists() else None
            samples.append(
                Sample(
                    image=image,
                    label=0 if is_normal else 1,
                    defect_type=defect_type,
                    mask=mask,
                )
            )
    return samples


def available_categories(root: str | Path) -> list[str]:
    """Which categories are actually downloaded, in canonical order."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [c for c in CATEGORIES if (root / c / "train" / NORMAL).is_dir()]
