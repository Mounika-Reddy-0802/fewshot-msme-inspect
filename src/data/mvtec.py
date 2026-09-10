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

# Published per-category counts: (train normals, test images). Used to tell a
# finished download from one that stopped halfway -- a directory existing says
# nothing about whether it is complete.
OFFICIAL_COUNTS: dict[str, tuple[int, int]] = {
    "bottle": (209, 83), "cable": (224, 150), "capsule": (219, 132),
    "carpet": (280, 117), "grid": (264, 78), "hazelnut": (391, 110),
    "leather": (245, 124), "metal_nut": (220, 115), "pill": (267, 167),
    "screw": (320, 160), "tile": (230, 117), "toothbrush": (60, 42),
    "transistor": (213, 100), "wood": (247, 79), "zipper": (240, 151),
}


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


def category_counts(root: str | Path, category: str) -> tuple[int, int]:
    """Actual (train normals, test images) on disk. (0, 0) if absent."""
    base = Path(root) / category
    if not base.is_dir():
        return (0, 0)
    train = len(list((base / "train" / NORMAL).glob("*.png"))) if (base / "train" / NORMAL).is_dir() else 0
    test = len(list((base / "test").rglob("*.png"))) if (base / "test").is_dir() else 0
    return (train, test)


def verify_category(root: str | Path, category: str) -> tuple[bool, str]:
    """Check a downloaded category against the published counts.

    A half-finished download leaves a perfectly valid-looking directory tree, so
    presence is not evidence of completeness -- only the counts are.
    """
    if category not in OFFICIAL_COUNTS:
        return False, f"{category}: not an MVTec AD category"
    expected = OFFICIAL_COUNTS[category]
    actual = category_counts(root, category)
    if actual == (0, 0):
        return False, f"{category}: missing"
    if actual != expected:
        return False, (
            f"{category}: incomplete -- have {actual[0]}/{actual[1]} "
            f"train/test, expected {expected[0]}/{expected[1]}"
        )
    return True, f"{category}: complete ({actual[0]} train, {actual[1]} test)"


def complete_categories(root: str | Path) -> list[str]:
    """Only the categories that fully match the published counts."""
    return [c for c in CATEGORIES if verify_category(root, c)[0]]
