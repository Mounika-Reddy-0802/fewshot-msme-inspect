"""Reading Member B's split manifests.

She owns split generation (`scripts/generate_splits.py`); this only reads what it
writes, so there is one definition of a split in the project rather than two.

Her layout is one JSON per dataset:

    {"dataset": "mvtec_ad", "seed": 42, "shot_counts": [0, 1, 5, 10],
     "categories": {"bottle": {"test": [...],
                               "shots": {"0": [], "1": [...], ...},
                               "num_train_good_available": 209}}}

Paths are relative to the data root and include the dataset directory, e.g.
``mvtec_ad/bottle/test/broken_large/000.png``.

Her shot sets are **nested** -- the 1-shot support is the first of the 5-shot,
which is the first of the 10-shot -- so a gain from adding shots is attributable
to the extra images rather than to a different draw.

Two things her manifest does not record, because a split does not need them:
per-image labels and mask paths. Both follow from the MVTec directory
convention and are derived here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_KEYS = ("dataset", "seed", "shot_counts", "categories")
NORMAL = "good"

# Datasets disagree on how deep the category directories sit. MVTec puts them
# directly under the dataset root; VisA nests them one level further. Mirrors
# CATEGORY_SUBDIR in scripts/generate_splits.py, which writes the manifests.
CATEGORY_SUBDIR = {
    "mvtec_ad": "",
    "visa": "visa_pytorch",
    "neu_det": "",
}


def category_root(dataset: str, data_root: str | Path) -> Path:
    """Directory that directly contains the category folders."""
    subdir = CATEGORY_SUBDIR.get(dataset, "")
    base = Path(data_root) / dataset
    return (base / subdir) if subdir else base


@dataclass(frozen=True)
class Sample:
    """One test image, with its mask when it has one."""

    image: Path
    label: int          # 0 normal, 1 anomalous
    defect_type: str
    mask: Path | None


def manifest_path(manifests_dir: str | Path, dataset: str) -> Path:
    return Path(manifests_dir) / f"{dataset}.json"


def manifest_paths(manifests_dir: str | Path, dataset: str) -> list[Path]:
    """Every seed's manifest for a dataset, default seed first.

    The default seed keeps the plain ``<dataset>.json`` name; additional seeds
    are ``<dataset>_seed<N>.json``. Averaging over several is what turns a
    single lucky or unlucky support draw into an error bar.
    """
    directory = Path(manifests_dir)
    found = []
    primary = directory / f"{dataset}.json"
    if primary.exists():
        found.append(primary)
    found.extend(sorted(directory.glob(f"{dataset}_seed*.json")))
    return found


def load_manifest(path: str | Path) -> dict:
    """Load and sanity-check one of her dataset manifests."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in SCHEMA_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"{path} is missing {missing}; expected keys {list(SCHEMA_KEYS)}")
    return manifest


def categories(manifest: dict) -> list[str]:
    return sorted(manifest["categories"])


def shot_counts(manifest: dict) -> list[int]:
    return sorted(int(k) for k in manifest["shot_counts"])


def _defect_of(relative_path: str) -> str:
    """`.../test/<defect>/000.png` -> `<defect>`."""
    parts = Path(relative_path).parts
    return parts[-2] if len(parts) >= 2 else NORMAL


def _mask_for(relative_path: str, root: Path) -> Path | None:
    """Find the ground-truth mask for a test image.

    Datasets disagree on the naming, and guessing wrong is silent: the image
    simply gets an all-zero mask and its pixel metrics quietly become
    meaningless. Both conventions in use here are tried explicitly.

        MVTec AD   test/broken_large/000.png -> ground_truth/broken_large/000_mask.png
        VisA       test/bad/000.JPG          -> ground_truth/bad/000.png
    """
    path = Path(relative_path)
    parts = list(path.parts)
    if "test" not in parts:
        return None
    parts[parts.index("test")] = "ground_truth"
    directory = root / Path(*parts[:-1])

    for name in (
        f"{path.stem}_mask{path.suffix}",   # MVTec, same extension
        f"{path.stem}_mask.png",            # MVTec, png masks
        f"{path.stem}.png",                 # VisA, png masks
        f"{path.stem}{path.suffix}",        # same name, same extension
    ):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def split_for(
    manifest: dict,
    category: str,
    shots: int,
    root: str | Path,
) -> dict:
    """One (category, shots) split, in the shape the evaluation pipeline uses.

    ``root`` is the data root her paths are relative to. Returned paths are
    relative to the *category* directory, matching how the pipeline resolves
    them.
    """
    if category not in manifest["categories"]:
        raise KeyError(f"no category {category!r}; have {categories(manifest)}")
    block = manifest["categories"][category]

    key = str(shots)
    if key not in block["shots"]:
        raise KeyError(
            f"{category} has no {shots}-shot split; have {sorted(block['shots'])}"
        )

    root = Path(root)
    marker = f"/{category}/"

    def strip(rel: str) -> str:
        """Path relative to the category directory.

        Derived from the path itself rather than assumed to be
        ``<dataset>/<category>/``: VisA nests categories under
        ``visa/visa_pytorch/``, so a hardcoded prefix silently fails to strip
        and every later lookup misses.
        """
        rel = rel.replace("\\", "/")
        index = rel.find(marker)
        return rel[index + len(marker):] if index >= 0 else rel

    test = []
    for rel in block["test"]:
        rel = rel.replace("\\", "/")
        defect = _defect_of(rel)
        mask = _mask_for(rel, root) if defect != NORMAL else None
        test.append({
            "image": strip(rel),
            "label": 0 if defect == NORMAL else 1,
            "defect_type": defect,
            "mask": mask.relative_to(
                category_root(manifest["dataset"], root) / category
            ).as_posix() if mask else None,
        })

    return {
        "dataset": manifest["dataset"],
        "category": category,
        "shots": shots,
        "seed": manifest["seed"],
        "support": [strip(p) for p in block["shots"][key]],
        "test": test,
        "counts": {
            "support": len(block["shots"][key]),
            "train_normals_available": block.get("num_train_good_available"),
            "test_total": len(test),
            "test_normal": sum(1 for t in test if t["label"] == 0),
            "test_anomalous": sum(1 for t in test if t["label"] == 1),
            "test_with_mask": sum(1 for t in test if t["mask"]),
        },
    }


def resolve_split(split: dict, root: str | Path) -> tuple[list[Path], list[Sample]]:
    """Turn a split's relative paths into absolute ones for this machine.

    ``root`` is the data root; the dataset's own layout is applied here.
    """
    base = category_root(split["dataset"], root) / split["category"]
    support = [base / p for p in split["support"]]
    test = [
        Sample(
            image=base / t["image"],
            label=t["label"],
            defect_type=t["defect_type"],
            mask=base / t["mask"] if t["mask"] else None,
        )
        for t in split["test"]
    ]
    return support, test
