"""Dataset implementations for EGD-CXR."""

from .egd_cxr import (
    BoxRow,
    EGDCXRDataset,
    collate_fn,
    create_dataloader,
)

__all__ = [
    "BoxRow",
    "EGDCXRDataset",
    "collate_fn",
    "create_dataloader",
]

