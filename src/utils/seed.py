"""Seeding.

Every run is seeded from its config so numbers can be reproduced, and every
seed we actually used gets recorded next to the results.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so workers stay reproducible."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass
