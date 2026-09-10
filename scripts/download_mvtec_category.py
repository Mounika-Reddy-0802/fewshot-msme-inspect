"""Fetch MVTec AD categories, and prove they actually arrived.

Week 1 needed one category; the benchmark needs all fifteen. The official archive
is a single ~5 GB tarball, so this pulls per-category from a mirror that keeps the
original folder layout (train/good, test/<defect>, ground_truth/<defect>).

    python scripts/download_mvtec_category.py --category bottle
    python scripts/download_mvtec_category.py --all

Two things this guards against, both of which bit us:

* **Rate limiting.** Hugging Face returns HTTP 429 after a burst of category
  downloads. Retries use exponential backoff rather than hammering.
* **Silent truncation.** A download that dies partway leaves a directory tree
  that looks perfectly valid. Every category is checked against the published
  file counts, and the script exits non-zero if any is short -- so a caller can
  trust the exit code instead of scraping stdout.

Full-dataset handling and the split generators are Member B's; this exists so the
benchmark and the validation in docs/ can be reproduced.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mvtec import CATEGORIES, verify_category  # noqa: E402
from src.utils.paths import dataset_dir  # noqa: E402

MIRROR = "foersben/mvtec-ad"


def download_category(
    category: str,
    root: Path,
    retries: int = 5,
    max_workers: int = 4,
) -> bool:
    """Download one category, retrying with backoff. Returns True if complete."""
    from huggingface_hub import snapshot_download

    ok, message = verify_category(root, category)
    if ok:
        print(f"  {message} -- skipping")
        return True

    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=MIRROR,
                repo_type="dataset",
                allow_patterns=[f"{category}/**", "license.txt", "readme.txt"],
                local_dir=str(root),
                max_workers=max_workers,
            )
        except Exception as exc:  # network, 429, partial writes
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"  attempt {attempt}/{retries} failed ({type(exc).__name__}), "
                  f"retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)
            continue

        ok, message = verify_category(root, category)
        if ok:
            print(f"  {message}")
            return True
        # A clean return with wrong counts means files were silently skipped;
        # another pass fills the gaps rather than declaring success.
        print(f"  {message} -- retrying")
        if attempt < retries:
            time.sleep(5)

    print(f"  {category}: FAILED after {retries} attempts")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=None, choices=CATEGORIES)
    parser.add_argument("--all", action="store_true", help="every category")
    parser.add_argument("--root", default=None, help="defaults to <data root>/mvtec_ad")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=4,
                        help="lower this if the mirror rate-limits (HTTP 429)")
    parser.add_argument("--pause", type=float, default=10.0,
                        help="seconds between categories, to stay under rate limits")
    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("pass --category <name> or --all")

    root = Path(args.root) if args.root else dataset_dir("mvtec_ad")
    root.mkdir(parents=True, exist_ok=True)

    targets = list(CATEGORIES) if args.all else [args.category]
    print(f"target: {root}\ncategories: {len(targets)}\n")

    failed = []
    for i, category in enumerate(targets):
        print(f"[{i + 1}/{len(targets)}] {category}")
        if not download_category(category, root, args.retries, args.max_workers):
            failed.append(category)
        if args.pause and i < len(targets) - 1:
            time.sleep(args.pause)

    print("\n=== summary ===")
    complete = [c for c in targets if verify_category(root, c)[0]]
    print(f"complete: {len(complete)}/{len(targets)}")
    if failed:
        print(f"FAILED:   {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
