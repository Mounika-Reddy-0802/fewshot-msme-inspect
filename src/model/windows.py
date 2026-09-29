"""Multi-scale window scoring on patch tokens, and image-level text scores.

**Windows.** WinCLIP's core idea is that a defect is judged better over a small
neighbourhood than a single patch. It gets window features by re-running the
encoder with attention masked to each window -- one pass per window, far too
expensive on a CPU. This keeps the idea at single-pass cost: average-pool the
already-computed patch tokens over k x k windows, score each window against a
memory of equally pooled support windows, spread each window's score back over
the patches it covers, and average across scales. Scale 1 is exactly the
single-pass prototype memory, so the ablation isolates what windows add.

**Text.** Patch tokens here are 768-dim and cannot meet text, but the pooled
image embedding is always in the 512-dim joint space. So a typed description
can score a whole image even though it cannot localise: the softmax weight on
the "defective" prompt ensemble, fused with the memory's image score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.encoder import l2_normalise
from src.model.memory import AnomalyScores, PrototypeMemory


def window_pool(patches: torch.Tensor, grid: tuple[int, int], k: int) -> torch.Tensor:
    """(N, P, D) patch tokens -> (N, (H-k+1)*(W-k+1), D) window tokens, stride 1."""
    if k == 1:
        return patches
    n, _, d = patches.shape
    h, w = grid
    x = patches.float().reshape(n, h, w, d).permute(0, 3, 1, 2)     # (N, D, H, W)
    pooled = F.avg_pool2d(x, kernel_size=k, stride=1)               # (N, D, H', W')
    return l2_normalise(pooled.flatten(2).transpose(1, 2))


def _spread(window_map: torch.Tensor, k: int) -> torch.Tensor:
    """(N, H-k+1, W-k+1) window scores -> (N, H, W): each patch gets the mean of
    the windows covering it."""
    if k == 1:
        return window_map
    x = window_map.unsqueeze(1)
    summed = F.conv_transpose2d(x, torch.ones(1, 1, k, k))
    count = F.conv_transpose2d(torch.ones_like(x), torch.ones(1, 1, k, k))
    return (summed / count).squeeze(1)


def windowed_scores(
    support: torch.Tensor,
    query: torch.Tensor,
    grid: tuple[int, int],
    scales: tuple[int, ...] = (1, 2, 3),
    top_k: int = 1,
) -> AnomalyScores:
    """Score (B, P, D) queries against (N, P, D) support at several window sizes."""
    h, w = grid
    maps = []
    for k in scales:
        memory = PrototypeMemory(support.shape[-1], (h - k + 1, w - k + 1), top_k=top_k)
        memory.add_class("c", window_pool(support, grid, k))
        maps.append(_spread(memory.score(window_pool(query, grid, k), "c").patches, k))
    combined = torch.stack(maps).mean(dim=0)                          # (B, H, W)
    return AnomalyScores(image=combined.flatten(1).max(dim=1).values, patches=combined)


def text_image_scores(
    images: torch.Tensor,
    text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """(B, D) pooled image embeddings, (2, D) [normal, anomalous] text -> (B,)
    probability of the anomalous prompt ensemble."""
    logits = l2_normalise(images.float()) @ text.T / temperature
    return logits.softmax(dim=-1)[:, 1]
