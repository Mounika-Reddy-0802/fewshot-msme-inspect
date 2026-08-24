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
    """MVTec keeps masks at ground_truth/<defect>/<stem>_mask.png."""
    path = Path(relative_path)
    parts = list(path.parts)
    if "test" not in parts:
        return None
    parts[parts.index("test")] = "ground_truth"
    candidate = root / Path(*parts[:-1]) / f"{path.stem}_mask{path.suffix}"
    if candidate.exists():
        return candidate
    # some datasets store masks as .png regardless of the image suffix
    alternative = candidate.with_suffix(".png")
    return alternative if alternative.exists() else None


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
    prefix = f"{manifest['dataset']}/{category}/"

    def strip(rel: str) -> str:
        rel = rel.replace("\\", "/")
        return rel[len(prefix):] if rel.startswith(prefix) else rel

    test = []
    for rel in block["test"]:
        rel = rel.replace("\\", "/")
        defect = _defect_of(rel)
        mask = _mask_for(rel, root) if defect != NORMAL else None
        test.append({
            "image": strip(rel),
            "label": 0 if defect == NORMAL else 1,
            "defect_type": defect,
            "mask": mask.relative_to(root / manifest["dataset"] / category).as_posix()
            if mask else None,
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
    """Turn a split's relative paths into absolute ones for this machine."""
    base = Path(root) / split["dataset"] / split["category"]
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
