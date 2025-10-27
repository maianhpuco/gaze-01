#!/usr/bin/env python3
"""
Single-label classification wrapper around the multimodal EGD-CXR dataset.

This module reuses the canonical `EGDCXRDataset` implementation (which reads
labels from `master_sheet.csv`) and converts its multi-label binary targets
into a single categorical label suitable for standard classification tasks
such as CHF vs Pneumonia vs Normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from egd_cxr_dataset.datasets.egd_cxr import (  # type: ignore[import]
    EGDCXRDataset,
    collate_fn as multimodal_collate_fn,
)


@dataclass(frozen=True)
class SingleLabelSpec:
    """Metadata describing the mapping from multi-label columns to a single label."""

    class_names: Tuple[str, ...]
    class_to_index: Dict[str, int]
    column_indices: Tuple[int, ...]
    priority: Tuple[str, ...]


class EGDCXRSingleLabelDataset(Dataset):
    """
    Dataset wrapper that yields a single categorical label per case.

    Args:
        classes: Sequence of column names within `master_sheet.csv` to supervise.
        priority: Optional override defining the order in which positives are picked.
        drop_unlabelled: Drop cases that do not have any of the requested labels.
    """

    def __init__(
        self,
        *,
        root: Path,
        seg_path: Path,
        transcripts_path: Optional[Path] = None,
        dicom_root: Optional[Path] = None,
        max_fixations: Optional[int] = None,
        case_ids: Optional[Sequence[str]] = None,
        classes: Sequence[str] = ("CHF", "Pneumonia", "Normal"),
        priority: Optional[Sequence[str]] = None,
        drop_unlabelled: bool = True,
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
            raise ValueError("`classes` must contain at least one class name.")
        class_names = tuple(cls.strip() for cls in classes if cls.strip())
        if not class_names:
            raise ValueError("All provided class names were empty.")

        base_columns = list(self.base.label_proc.schema.class_columns)
        class_to_index: Dict[str, int] = {}
        column_indices: List[int] = []
        for cls in class_names:
            if cls not in base_columns:
                raise ValueError(
                    f"Requested class '{cls}' not present in master_sheet columns {base_columns}"
                )
            class_to_index[cls] = len(class_to_index)
            column_indices.append(base_columns.index(cls))

        priority_order = tuple(priority) if priority else class_names
        for cls in priority_order:
            if cls not in class_to_index:
                raise ValueError(
                    f"Priority class '{cls}' is not part of the requested class set {class_names}"
                )

        self.spec = SingleLabelSpec(
            class_names=class_names,
            class_to_index=class_to_index,
            column_indices=tuple(column_indices),
            priority=priority_order,
        )
        self.drop_unlabelled = drop_unlabelled

        self._indices: List[int] = []
        self._targets: List[int] = []
        for idx, case_id in enumerate(self.base.case_ids):
            labels_vector, _, _ = self.base.label_proc.vector(case_id)
            class_idx = self._resolve_label(labels_vector)
            if class_idx is None:
                if self.drop_unlabelled:
                    continue
                class_idx = len(class_names) - 1
            self._indices.append(idx)
            self._targets.append(class_idx)

        if not self._indices:
            raise ValueError("No cases matched the single-label mapping.")

        self._class_counts = torch.bincount(
            torch.tensor(self._targets, dtype=torch.long),
            minlength=len(class_names),
        )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        base_idx = self._indices[index]
        sample = self.base[base_idx]
        target = self._targets[index]

        labels_payload = sample.setdefault("labels", {})
        labels_payload["single_index"] = torch.tensor(target, dtype=torch.long)
        labels_payload["single_name"] = self.spec.class_names[target]
        labels_payload["single_class_names"] = self.spec.class_names
        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.spec.class_names

    @property
    def class_counts(self) -> torch.Tensor:
        return self._class_counts.clone()

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts.to(torch.float32)
        total = counts.sum().item()
        weights = torch.zeros_like(counts)
        for idx, count in enumerate(counts):
            weights[idx] = total / max(float(count.item()), 1.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights

    def sample_weights(self) -> torch.Tensor:
        weights = self.class_weights()
        targets = torch.tensor(self._targets, dtype=torch.long)
        return weights[targets]

    def _resolve_label(self, label_vec: torch.Tensor) -> Optional[int]:
        for cls in self.spec.priority:
            column_idx = self.spec.column_indices[self.spec.class_names.index(cls)]
            if int(label_vec[column_idx].item()) == 1:
                return self.spec.class_to_index[cls]
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
    Convenience dataloader factory using the canonical collate function.
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

