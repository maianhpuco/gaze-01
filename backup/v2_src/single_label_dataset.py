#!/usr/bin/env python3
"""
Single-label variant of the EGD-CXR multimodal dataset.

This wrapper converts the original multi-label (binary vector) annotation
into a single categorical target suitable for standard multi-class
classification experiments. By default it derives one of three classes:
    • CHF
    • Pneumonia
    • Normal

The transformation follows a simple priority scheme: for each case we inspect
the binary label columns (taken from `master_sheet.csv`) in the order supplied
by the user (default: CHF → Pneumonia → Normal). The first positive column
encountered defines the class. Cases without any positive indicator for the
requested classes are excluded from the dataset to avoid noisy supervision.

The rest of the multimodal content (gaze fixations, segmentation masks,
transcripts, image tensors, etc.) is forwarded unchanged from the base
`EGDCXRDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Re-use the existing multimodal dataset and utilities
from egd_cxr_dataset.datasets.egd_cxr import (
    EGDCXRDataset,
    collate_fn as multimodal_collate_fn,
)


@dataclass(frozen=True)
class SingleLabelInfo:
    """Metadata describing the single label mapping."""

    class_names: Tuple[str, ...]
    class_to_index: Dict[str, int]
    label_column_indices: Tuple[int, ...]


class EGDCXRSingleLabelDataset(Dataset):
    """
    Wraps :class:`EGDCXRDataset` to expose a single categorical target.

    Args:
        root: Path to gaze dataset root (same as ``EGDCXRDataset``).
        seg_path: Path to segmentation data.
        transcripts_path: Optional transcripts directory/CSV.
        dicom_root: Optional directory of DICOM images.
        max_fixations: Optional maximum # of fixations per case.
        case_ids: Optional subset of case identifiers to keep.
        classes: Ordered sequence of class names to derive from the
            multi-label columns. The first positive label encountered
            becomes the class assignment.
        drop_unlabelled: If True, cases with no positive indicator for the
            requested classes are removed (default: True).
        priority_first: Optional explicit priority order. When omitted the
            order of ``classes`` is used.
    """

    def __init__(
        self,
        root: Path,
        seg_path: Path,
        transcripts_path: Optional[Path] = None,
        *,
        dicom_root: Optional[Path] = None,
        max_fixations: Optional[int] = None,
        case_ids: Optional[Sequence[str]] = None,
        classes: Sequence[str] = ("CHF", "Pneumonia", "Normal"),
        drop_unlabelled: bool = True,
        priority_first: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.base = EGDCXRDataset(
            root=root,
            seg_path=seg_path,
            transcripts_path=transcripts_path,
            dicom_root=dicom_root,
            max_fixations=max_fixations,
            case_ids=case_ids,
        )

        if not classes:
            raise ValueError("At least one class must be supplied for single-label conversion.")
        self.classes: Tuple[str, ...] = tuple(classes)
        priority_sequence = tuple(priority_first) if priority_first else self.classes

        # Build mapping from desired class -> column index in the base dataset
        base_label_names = list(self.base.label_proc.schema.class_columns)
        class_to_index: Dict[str, int] = {}
        column_indices: List[int] = []
        for cls in self.classes:
            if cls not in base_label_names:
                raise ValueError(
                    f"Requested class '{cls}' not found among label columns "
                    f"{base_label_names}"
                )
            class_to_index[cls] = len(class_to_index)
            column_indices.append(base_label_names.index(cls))

        self.label_info = SingleLabelInfo(
            class_names=self.classes,
            class_to_index=class_to_index,
            label_column_indices=tuple(column_indices),
        )
        self.priority = tuple(priority_sequence)

        # Pre-compute which cases satisfy the single-label requirement
        self._kept_indices: List[int] = []
        self._targets: List[int] = []
        base_case_ids = list(self.base.case_ids)

        for idx, case_id in enumerate(base_case_ids):
            labels_vec, _, _ = self.base.label_proc.vector(case_id)
            label_index = self._resolve_single_label(labels_vec)
            if label_index is None:
                if drop_unlabelled:
                    continue
                else:
                    # Skip but keep dataset consistent by assigning last class
                    label_index = len(self.classes) - 1
            self._kept_indices.append(idx)
            self._targets.append(label_index)

        if not self._kept_indices:
            raise ValueError(
                "No cases matched the requested label specification. "
                "Adjust the class list or priority ordering."
            )

        counts = torch.bincount(torch.tensor(self._targets, dtype=torch.long), minlength=len(self.classes))
        self._class_counts = counts

    # ------------------------------------------------------------------ #
    # Helper API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.label_info.class_names

    @property
    def class_counts(self) -> torch.Tensor:
        return self._class_counts.clone()

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_weights(self) -> torch.Tensor:
        """
        Return inverse-frequency weights (useful for CrossEntropy).
        """
        counts = self.class_counts
        total = counts.sum().item()
        weights = torch.zeros_like(counts, dtype=torch.float32)
        for idx, count in enumerate(counts):
            weights[idx] = total / max(float(count.item()), 1.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights

    def sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights (inverse class frequency) for weighted sampling.
        """
        weights = self.class_weights()
        targets = torch.tensor(self._targets, dtype=torch.long)
        return weights[targets]

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                   #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._kept_indices)

    def __getitem__(self, idx: int):
        base_idx = self._kept_indices[idx]
        sample = self.base[base_idx]
        target = self._targets[idx]

        # Attach single-label metadata while preserving the original structure
        labels_payload = sample.setdefault("labels", {})
        labels_payload["single_index"] = torch.tensor(target, dtype=torch.long)
        labels_payload["single_name"] = self.classes[target]
        labels_payload["single_class_names"] = self.classes

        return sample

    # ------------------------------------------------------------------ #
    # Internal utilities                                                 #
    # ------------------------------------------------------------------ #
    def _resolve_single_label(self, labels_vec: torch.Tensor) -> Optional[int]:
        """
        Convert a binary label vector into a single class index by priority.
        """
        positives: Dict[str, bool] = {}
        for cls_name, col_idx in zip(self.classes, self.label_info.label_column_indices):
            positives[cls_name] = bool(int(labels_vec[col_idx].item()) == 1)

        for cls_name in self.priority:
            if positives.get(cls_name, False):
                return self.label_info.class_to_index[cls_name]

        return None


def create_single_label_dataloader(
    dataset: EGDCXRSingleLabelDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience dataloader factory mirroring ``egd_cxr_dataset.create_dataloader``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=False,
        pin_memory=True,
    )

