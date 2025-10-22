"""
EGD-CXR dataset utilities.

Provides the `MimicDataset` PyTorch dataset along with helper processors
for fixations and labels.
"""

from .dataset import MimicDataset, CropConfig, ConfigLoader, LabelProcessor, FixationProcessor
from .datasets import BoxRow, EGDCXRDataset, collate_fn as egd_cxr_collate, create_dataloader
from .models import SilenceThoughtModel, compute_losses as silence_losses
from .split import SplitConfig, load_case_ids, split_ids, write_split_files
from .utils import (
    BOS,
    EOS,
    PAD,
    UNK,
    Box,
    Vocab,
    build_vocab,
)

__all__ = [
    "MimicDataset",
    "CropConfig",
    "ConfigLoader",
    "LabelProcessor",
    "FixationProcessor",
    "EGDCXRDataset",
    "BoxRow",
    "egd_cxr_collate",
    "create_dataloader",
    "SilenceThoughtModel",
    "silence_losses",
    "Vocab",
    "build_vocab",
    "Box",
    "PAD",
    "BOS",
    "EOS",
    "UNK",
    "SplitConfig",
    "load_case_ids",
    "split_ids",
    "write_split_files",
]
