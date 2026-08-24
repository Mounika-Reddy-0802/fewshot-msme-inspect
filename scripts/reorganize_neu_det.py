"""Reorganize the raw NEU-DET Kaggle download into data/neu_det/ (T2.1).

NEU-DET has no defect-free class -- all 1800 images belong to one of six
defect classes (crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches), pre-split into train/ and validation/ by the
Kaggle uploader. This script only copies files into the project's usual
data/ location and pairs each image with its annotation; it does NOT
invent a train/good folder, since no defect-free images exist here. How
this dataset feeds the 0/1/5/10-shot protocol (or whether it's used
differently, e.g. as a cross-domain zero-shot test) is an open question
for the team -- not decided by this script.

Expects the raw Kaggle download already extracted to
data/neu_det_raw/NEU-DET/ (images nested by class, annotations flat,
matched by filename stem):

    python scripts/reorganize_neu_det.py

Writes to data/neu_det/<class>/<split>/images/*.jpg and
data/neu_det/<class>/<split>/annotations/*.xml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import dataset_dir  # noqa: E402

CLASSES = (
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
)
SPLITS = ("train", "validation")


def main() -> int:
    raw_root = dataset_dir("neu_det_raw") / "NEU-DET"
    out_root = dataset_dir("neu_det")

    if not raw_root.is_dir():
        print(f"raw download not found at {raw_root} -- extract the Kaggle zip there first")
        return 1

    total = 0
    for split in SPLITS:
        images_root = raw_root / split / "images"
        annotations_root = raw_root / split / "annotations"

        for cls in CLASSES:
            src_images = images_root / cls
            if not src_images.is_dir():
                print(f"  {split}/{cls}: no images folder found, skipping")
                continue

            dst_images = out_root / cls / split / "images"
            dst_annotations = out_root / cls / split / "annotations"
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_annotations.mkdir(parents=True, exist_ok=True)

            count = 0
            missing_ann = 0
            for img_path in sorted(src_images.glob("*.jpg")):
                shutil.copy2(img_path, dst_images / img_path.name)

                ann_path = annotations_root / f"{img_path.stem}.xml"
                if ann_path.exists():
                    shutil.copy2(ann_path, dst_annotations / ann_path.name)
                else:
                    missing_ann += 1
                count += 1

            total += count
            note = f", {missing_ann} missing annotations" if missing_ann else ""
            print(f"  {split}/{cls:<16} {count} images{note}")

    print(f"\ntotal images copied: {total}")
    print(f"written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
