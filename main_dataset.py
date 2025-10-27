#!/usr/bin/env python3
"""Inspect/train/test split helper for single-label EGD-CXR dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader  # type: ignore[import]
from src.datasets.egd_cxr_rewritten import (  # type: ignore[import]
    EGDCXRRewrittenDataset,
    create_dataloader,
)


def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    ids: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _slice_batch(value: Any, idx: int, batch_size: int):
    if isinstance(value, torch.Tensor):
        return value[idx]
    if isinstance(value, dict):
        return {key: _slice_batch(val, idx, batch_size) for key, val in value.items()}
    if isinstance(value, list):
        return value[idx] if len(value) == batch_size else value
    return value


def inspect_sample(sample: dict, idx: int) -> None:
    labels = sample["labels"]
    print(f"\nSample {idx}")
    print(f"  dicom_id: {sample['dicom_id']}")
    
    # Print class information more clearly
    single_name = labels.get('single_name', 'Unknown')
    single_index = labels.get('single_index', torch.tensor(-1))
    if isinstance(single_index, torch.Tensor):
        single_index = single_index.item()
    
    print(f"  class: {single_name} (index {single_index})")
    
    # Print classification details if available
    classification = labels.get("classification")
    if isinstance(classification, dict):
        cls_name = classification.get("name", "Unknown")
        cls_idx = classification.get("index", -1)
        ambiguous = classification.get("ambiguous", False)
        tag = " (ambiguous)" if ambiguous else ""
        print(f"  classification: {cls_name} (index {cls_idx}){tag}")
        
        # Print all available class names
        class_names = classification.get("classes", [])
        if class_names:
            print(f"  available classes: {list(class_names)}")
    
    print(f"  image: {tuple(sample['image'].shape)}")
    fixation = sample["fixations"]
    num_fix = fixation["xy"].shape[0]
    durations = fixation.get("duration")
    if isinstance(durations, torch.Tensor) and durations.numel() > 0:
        mean_dur = durations.mean().item()
        print(f"  fixations: {num_fix} points | mean duration {mean_dur:.3f}s")
    else:
        print(f"  fixations: {num_fix} points")
    segments = sample.get("segments")
    if isinstance(segments, torch.Tensor):
        print(f"  segments: {tuple(segments.shape)} (channels, H, W) - onehot encoding")
    box_masks = sample.get("box_masks")
    if isinstance(box_masks, torch.Tensor):
        print(f"  box masks: {tuple(box_masks.shape)} - onehot encoding")
    boxes = sample.get("boxes", [])
    print(f"  boxes: {len(boxes)} entries")
    transcript = sample["transcript"]
    if isinstance(transcript, dict):
        print(f"  transcript segments: {len(transcript.get('segments', []))}")
        print(f"  transcript text length: {len(transcript.get('text', ''))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect single-label EGD-CXR datasets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["CHF", "pneumonia", "Normal"])
    parser.add_argument("--max-fixations", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    return parser.parse_args()


def build_dataset(config: ConfigLoader, split: str, *, classes, max_fixations, case_ids):
    if not case_ids:
        raise ValueError(
            f"Split '{split}' has no case identifiers. Populate configs/splits/{split}_ids.txt with case IDs."
        )
    root = Path(config.get("input_path", "gaze_raw"))
    seg = Path(config.get("input_path", "segmentation_dir"))
    transcripts = Path(config.get("input_path", "transcripts_dir", default=seg))
    dicom_root = Path(config.get("input_path", "dicom_raw"))

    return EGDCXRRewrittenDataset(
        root=root,
        seg_path=seg,
        transcripts_path=transcripts,
        dicom_root=dicom_root,
        max_fixations=max_fixations,
        case_ids=case_ids,
        classes=classes,
    )


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader(args.config)

    split_dir = Path(cfg.get("split_files", "dir", default=ROOT / "configs" / "splits"))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")

    datasets = {
        "train": build_dataset(cfg, "train", classes=args.classes, max_fixations=args.max_fixations, case_ids=train_ids),
        "val": build_dataset(cfg, "val", classes=args.classes, max_fixations=args.max_fixations, case_ids=val_ids),
        "test": build_dataset(cfg, "test", classes=args.classes, max_fixations=args.max_fixations, case_ids=test_ids),
    }

    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} samples | class counts {dataset.class_counts.tolist()}")
        print(f"  Class names: {list(dataset.class_names)}")
        print(f"  Class mapping: {dict(enumerate(dataset.class_names))}")

        sampler = None
        if split == "train" and not args.disable_weighted_sampler:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.sample_weights().double(),
                num_samples=len(dataset),
                replacement=True,
            )

        loader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            num_workers=args.num_workers,
        )

        if split == "train":
            inspected = 0
            for batch in loader:
                batch_size = batch["labels"]["single_index"].shape[0]
                for i in range(batch_size):
                    sample = {key: _slice_batch(value, i, batch_size) for key, value in batch.items()}
                    if "transcripts" in sample:
                        sample["transcript"] = sample.pop("transcripts")
                    inspect_sample(sample, inspected)
                    inspected += 1
                    if inspected >= args.max_samples:
                        break
                if inspected >= args.max_samples:
                    break


if __name__ == "__main__":
    main()
