"""Seeded few-shot splits.

Two properties this generator guarantees, both of which matter for the claims we
want to make:

**Nested shots.** The 1-shot support set is a subset of the 5-shot set, which is
a subset of the 10-shot set. Support images are drawn by shuffling the training
normals once per (category, seed) and taking the first k. So a jump from 1 to 5
shots is attributable to the four added images, not to having drawn a different,
luckier set.

**Fixed test split.** k never changes the test images. Accuracy at 1 and 10 shots
is measured on exactly the same queries, so the numbers are directly comparable.

Seeds are derived with sha256 rather than :func:`hash`, which is salted per
process and would silently produce different splits on different runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.data.mvtec import Sample, category_dir, list_test_samples, list_train_normals

SCHEMA_VERSION = 1
SHOT_COUNTS = (0, 1, 5, 10)


def _derive_seed(base_seed: int, *parts: object) -> int:
    """Stable per-(category, seed) seed. sha256, not hash() -- hash() is salted."""
    key = "|".join([str(base_seed), *(str(p) for p in parts)]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def make_split(
    root: str | Path,
    category: str,
    shots: int,
    seed: int,
    dataset: str = "mvtec_ad",
) -> dict:
    """Build one split manifest. Paths are relative to the category directory so
    the manifest is portable between machines (local, Kaggle, a teammate's box)."""
    if shots < 0:
        raise ValueError(f"shots must be >= 0, got {shots}")

    base = category_dir(root, category)
    normals = list_train_normals(root, category)
    if shots > len(normals):
        raise ValueError(
            f"asked for {shots} shots but {category} only has {len(normals)} training normals"
        )

    # shuffle once per (category, seed), then take the first k -> nested supports
    order = list(normals)
    random.Random(_derive_seed(seed, dataset, category)).shuffle(order)
    support = sorted(p.relative_to(base).as_posix() for p in order[:shots])

    test = [
        {
            "image": s.image.relative_to(base).as_posix(),
            "label": s.label,
            "defect_type": s.defect_type,
            "mask": s.mask.relative_to(base).as_posix() if s.mask else None,
        }
        for s in list_test_samples(root, category)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "category": category,
        "shots": shots,
        "seed": seed,
        "support": support,
        "test": test,
        "counts": {
            "support": len(support),
            "train_normals_available": len(normals),
            "test_total": len(test),
            "test_normal": sum(1 for t in test if t["label"] == 0),
            "test_anomalous": sum(1 for t in test if t["label"] == 1),
            "test_with_mask": sum(1 for t in test if t["mask"]),
        },
    }


def split_filename(manifest: dict) -> str:
    return (
        f"{manifest['dataset']}_{manifest['category']}"
        f"_{manifest['shots']}shot_seed{manifest['seed']}.json"
    )


def write_split(manifest: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / split_filename(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema v{manifest.get('schema_version')}, expected v{SCHEMA_VERSION}"
        )
    return manifest


def resolve_split(manifest: dict, root: str | Path) -> tuple[list[Path], list[Sample]]:
    """Turn a manifest's relative paths back into absolute ones for this machine."""
    base = category_dir(root, manifest["category"])
    support = [base / p for p in manifest["support"]]
    test = [
        Sample(
            image=base / t["image"],
            label=t["label"],
            defect_type=t["defect_type"],
            mask=base / t["mask"] if t["mask"] else None,
        )
        for t in manifest["test"]
    ]
    return support, test
