"""Build the one-class-as-normal folder layout for NEU-DET (T2.1).

NEU-DET has no defect-free class, so each of the six defect classes
becomes its own "category" under the one-vs-rest protocol: for category
c, c's own images are "normal" and every other class is "anomalous".
This produces the exact MVTec folder shape so the existing split/label
code (generate_splits.py, src/data/splits.py) needs zero special-casing
beyond adding "neu_det" to CATEGORY_SUBDIR.

Reads from data/neu_det/<class>/{train,validation}/images/ (written by
reorganize_neu_det.py) and writes, per category c, into the SAME
data/neu_det/ root:

    data/neu_det/<c>/train/good/*.jpg        <- c's own train images
    data/neu_det/<c>/test/good/*.jpg         <- c's own validation images
    data/neu_det/<c>/test/<other>/*.jpg      <- a fixed sample of each
                                                 other class's validation
                                                 images, split evenly

The anomalous test sample is drawn with a FIXED, hardcoded seed
(independent of whatever --seeds generate_splits.py is later run with),
so the test set is identical across every seed's manifest -- only the
*shot* selection from train/good varies by seed. This is what makes the
test split "fixed across shot counts and seeds", per the protocol.

    python scripts/build_neu_det_oneclass.py

Safe to re-run: existing destination files are left alone, not
duplicated or overwritten with a different draw.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import dataset_dir  # noqa: E402

CLASSES = (
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
)

# Fixed and independent of generate_splits.py's --seeds, so the test set
# never changes no matter which shot-seeds are generated later.
FIXED_TEST_SAMPLE_SEED = 20260824


def _list_images(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.jpg"))


def _copy_all(src_files: list[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_files:
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1
    return count


def main() -> int:
    root = dataset_dir("neu_det")
    rng = random.Random(FIXED_TEST_SAMPLE_SEED)

    if not root.is_dir():
        print(f"{root} not found -- run reorganize_neu_det.py first")
        return 1

    print(f"building one-vs-rest layout under {root}\n")

    for c in CLASSES:
        train_src = root / c / "train" / "images"
        val_src = root / c / "validation" / "images"
        if not train_src.is_dir() or not val_src.is_dir():
            print(f"  {c}: missing train/validation images, skipping")
            continue

        # 1. train/good <- c's own train images
        n_train = _copy_all(_list_images(train_src), root / c / "train" / "good")

        # 2. test/good <- c's own validation images
        own_val = _list_images(val_src)
        n_good_test = _copy_all(own_val, root / c / "test" / "good")

        # 3. test/<other> <- fixed sample from the other five classes'
        #    validation images, split as evenly as possible, totalling
        #    n_good_test (so the test set is 50/50 normal vs anomalous)
        other_classes = [x for x in CLASSES if x != c]
        base_count, remainder = divmod(n_good_test, len(other_classes))

        n_anomalous_total = 0
        for i, other in enumerate(other_classes):
            take = base_count + (1 if i < remainder else 0)
            other_val = _list_images(root / other / "validation" / "images")
            candidates = sorted(other_val)
            rng.shuffle(candidates)
            chosen = candidates[:take]
            n_anomalous_total += _copy_all(chosen, root / c / "test" / other)

        print(
            f"  {c:<16} train/good={n_train:<4} test/good={n_good_test:<4} "
            f"test/anomalous={n_anomalous_total:<4} "
            f"(balanced: {n_good_test == n_anomalous_total})"
        )

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
