"""Dataset loading and split generation.

Member B owns this area. These are minimal stand-ins written so the modelling
and profiling work is not blocked; the on-disk formats (split manifests, feature
banks) are the shared interface and are documented in docs/data_formats.md.
Their real loaders should be able to replace these without changing either format.
"""

from src.data.mvtec import Sample, list_test_samples, list_train_normals
from src.data.splits import load_split, make_split, resolve_split, write_split

__all__ = [
    "Sample",
    "list_test_samples",
    "list_train_normals",
    "load_split",
    "make_split",
    "resolve_split",
    "write_split",
]
