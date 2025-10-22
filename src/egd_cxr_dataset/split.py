#!/usr/bin/env python3
"""
Utilities for generating deterministic train/val/test splits of EGD-CXR case IDs.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# Avoid thread-heavy defaults when pandas or numpy are loaded inside the cluster sandbox.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SplitMapping = Dict[str, List[str]]


@dataclass(frozen=True)
class SplitConfig:
    ratios: Tuple[float, float, float]  # train, val, test
    seed: int = 17
    column: str = "dicom_id"

    def normalised(self) -> "SplitConfig":
        total = sum(self.ratios)
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value.")
        ratios = tuple(r / total for r in self.ratios)
        return SplitConfig(ratios=ratios, seed=self.seed, column=self.column)

    @property
    def names(self) -> Tuple[str, str, str]:
        return ("train", "val", "test")


def load_case_ids(master_sheet: Path, *, column: str = "dicom_id") -> List[str]:
    """Extract the unique case IDs from master_sheet.csv."""
    master_sheet = Path(master_sheet)
    if not master_sheet.exists():
        raise FileNotFoundError(f"master_sheet.csv not found at {master_sheet}")

    df = pd.read_csv(master_sheet, usecols=[column], engine="python")
    ids = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    if not ids:
        raise ValueError(f"No IDs found in column '{column}' of {master_sheet}")
    return ids


def _compute_split_counts(total: int, ratio_triplet: Sequence[float]) -> Tuple[int, int, int]:
    """Compute integer counts for each split while preserving totals."""
    if total <= 0:
        raise ValueError("Total number of IDs must be positive.")
    if any(r < 0 for r in ratio_triplet):
        raise ValueError("Split ratios must be non-negative.")

    expected = [r * total for r in ratio_triplet]
    counts = [int(math.floor(x)) for x in expected]
    remainder = total - sum(counts)

    if remainder > 0:
        fractional = sorted(
            enumerate(expected),
            key=lambda item: (item[1] - math.floor(item[1])),
            reverse=True,
        )
        for idx, _ in fractional:
            if remainder == 0:
                break
            counts[idx] += 1
            remainder -= 1

    train_count, val_count, test_count = counts
    if train_count + val_count + test_count != total:
        raise AssertionError("Split counts do not sum to total number of IDs.")
    return train_count, val_count, test_count


def split_ids(
    ids: Iterable[str],
    config: SplitConfig,
) -> SplitMapping:
    """
    Shuffle IDs deterministically and divide into train/val/test splits.
    """
    ids_list = list(ids)
    if not ids_list:
        raise ValueError("No IDs provided for splitting.")

    config = config.normalised()
    rng = random.Random(config.seed)
    rng.shuffle(ids_list)

    train_count, val_count, test_count = _compute_split_counts(
        len(ids_list), config.ratios
    )
    train_ids = ids_list[:train_count]
    val_ids = ids_list[train_count : train_count + val_count]
    test_ids = ids_list[train_count + val_count :]

    return dict(zip(config.names, (train_ids, val_ids, test_ids)))


def write_split_files(splits: SplitMapping, output_dir: Path) -> Dict[str, Path]:
    """Persist split IDs into text files under the specified directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for split_name, split_ids in splits.items():
        path = output_dir / f"{split_name}_ids.txt"
        payload = "\n".join(split_ids)
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")
        written[split_name] = path
    return written

