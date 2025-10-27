from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class Box:
    """Simple bounding box container used during collation."""

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int


def assign_segments_for_fixations(xy: torch.Tensor, seg_masks: torch.Tensor) -> torch.Tensor:
    """Return segmentation channel index for each fixation."""

    xy_long = xy.long().clamp_min(0)
    xs = xy_long[..., 0]
    ys = xy_long[..., 1]

    batch, time, _ = xy.shape
    _, seg_channels, height, width = seg_masks.shape
    xs = xs.clamp(max=width - 1)
    ys = ys.clamp(max=height - 1)

    seg_ids = torch.full((batch, time), fill_value=seg_channels - 1, dtype=torch.long, device=xy.device)
    for idx in range(batch):
        vals = seg_masks[idx, :, ys[idx], xs[idx]]
        hit = vals > 0.5
        any_hit = hit.any(dim=0)
        first = torch.argmax(hit.int(), dim=0)
        seg_ids[idx, any_hit] = first[any_hit]
    return seg_ids


def assign_boxes_for_fixations(
    xy: torch.Tensor,
    boxes: List[List[Box]],
    default_cls: Optional[int],
) -> torch.Tensor:
    """Assign the smallest-area enclosing box class to each fixation."""

    batch, time, _ = xy.shape
    device = xy.device
    xy_long = xy.long()

    if default_cls is None:
        max_cls = 0
        for case in boxes:
            if case:
                max_cls = max(max_cls, max(b.cls_id for b in case))
        default_cls = max_cls

    out = torch.full((batch, time), fill_value=default_cls, dtype=torch.long, device=device)
    for idx in range(batch):
        case_boxes = boxes[idx]
        if not case_boxes:
            continue
        areas = torch.tensor(
            [(b.x2 - b.x1) * (b.y2 - b.y1) for b in case_boxes],
            dtype=torch.long,
            device=device,
        ).clamp(min=1)
        for t in range(time):
            x = int(xy_long[idx, t, 0].item())
            y = int(xy_long[idx, t, 1].item())
            indices = [i for i, b in enumerate(case_boxes) if b.x1 <= x < b.x2 and b.y1 <= y < b.y2]
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                pick = idx_tensor[torch.argmin(areas[idx_tensor])]
                out[idx, t] = case_boxes[int(pick.item())].cls_id
    return out


__all__ = ["Box", "assign_segments_for_fixations", "assign_boxes_for_fixations"]

