"""Validate every committed split manifest (T2.4).

Loads every manifest under results/manifests/splits/ (every dataset,
every seed) and fails with a non-zero exit code if any of the following
hold for any (dataset, seed, category):

1. A support/shot image also appears in that category's test set.
2. Shot sets are not nested (0 subset 1 subset 5 subset 10).
3. Two different seeds of the same dataset draw an identical shot set
   for the same category and shot count (ignoring the trivially-empty
   0-shot case).

    python scripts/check_splits.py

Exits 0 and prints a summary if everything is clean.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import load_manifest  # noqa: E402
from src.utils.paths import results_root  # noqa: E402

MANIFEST_NAME = re.compile(r"^(?P<dataset>.+?)(?:_seed\d+)?\.json$")


def discover_datasets(splits_dir: Path) -> list[str]:
    names = set()
    for path in splits_dir.glob("*.json"):
        match = MANIFEST_NAME.match(path.name)
        if match:
            names.add(match.group("dataset"))
    return sorted(names)


def check_manifest(path: Path, errors: list[str], draws: dict) -> None:
    manifest = load_manifest(path)
    dataset = manifest["dataset"]
    seed = manifest["seed"]

    for category, block in manifest["categories"].items():
        test_set = set(block["test"])
        shots = block["shots"]
        shot_levels = sorted((int(k) for k in shots), key=int)

        prev: set = set()
        for level in shot_levels:
            key = str(level)
            current = set(shots[key])

            # 1. nesting
            if not prev <= current:
                errors.append(
                    f"{path.name}: {category} shot {level} is not a superset of the previous level"
                )

            # 2. disjointness from test
            overlap = current & test_set
            if overlap:
                errors.append(
                    f"{path.name}: {category} shot {level} overlaps test set "
                    f"({len(overlap)} shared image(s), e.g. {next(iter(overlap))})"
                )

            prev = current

            # record draw for cross-seed comparison, skip trivial 0-shot
            if level != 0:
                draws[(dataset, category, level)].append((seed, tuple(sorted(current))))


def check_cross_seed_duplicates(draws: dict, errors: list[str]) -> None:
    for (dataset, category, level), entries in draws.items():
        seen: dict[tuple, int] = {}
        for seed, draw in entries:
            if draw in seen:
                errors.append(
                    f"{dataset}: {category} shot {level} -- seed {seen[draw]} and "
                    f"seed {seed} drew an identical shot set"
                )
            else:
                seen[draw] = seed


def main() -> int:
    splits_dir = results_root() / "manifests" / "splits"
    if not splits_dir.is_dir():
        print(f"no manifests found at {splits_dir}")
        return 1

    datasets = discover_datasets(splits_dir)
    print(f"checking {len(datasets)} dataset(s): {datasets}\n")

    errors: list[str] = []
    draws: dict = defaultdict(list)
    manifest_count = 0

    for path in sorted(splits_dir.glob("*.json")):
        check_manifest(path, errors, draws)
        manifest_count += 1

    check_cross_seed_duplicates(draws, errors)

    if errors:
        print(f"FAILED -- {len(errors)} issue(s) across {manifest_count} manifest(s):\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK -- {manifest_count} manifest(s) across {len(datasets)} dataset(s), no issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
