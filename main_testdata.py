#!/usr/bin/env python3
"""
Smoke-test script for the EGDCXRDataset and DataLoader.

Example:
    python main_testdata.py --split train --batch-size 2 --num-batches 1
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader
from egd_cxr_dataset.datasets import EGDCXRDataset, create_dataloader

DEFAULT_CONFIG = ROOT / "config" / "data_egd-cxr.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test EGDCXRDataset train/val/test splits.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Which split to load (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the DataLoader (default: 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="torch DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to iterate for inspection (default: 1).",
    )
    parser.add_argument(
        "--max-fixations",
        type=int,
        help="Optional cap on the number of fixations per sample.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Force shuffling regardless of split.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        help="Optional limit on the number of cases to load from the split file.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Emit batch summary as JSON.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print the raw batch payload returned by the DataLoader (can be large).",
    )
    return parser.parse_args()


def read_split_ids(split_file: Path, limit: int | None = None) -> List[str]:
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return ids[:limit]
    return ids


def serialisable_batch_summary(batch: Dict) -> Dict:
    fix = batch["fixations"]
    first_idx = 0
    transcript_entry = batch["transcripts"][first_idx] if batch["transcripts"] else {}
    transcript_text = ""
    transcript_segments = []
    if isinstance(transcript_entry, dict):
        transcript_text = transcript_entry.get("text", "")
        segments = transcript_entry.get("segments", [])
        transcript_segments = segments[: min(3, len(segments))]
    return {
        "batch_size": int(fix["xy"].shape[0]),
        "seq_len": int(fix["xy"].shape[1]),
        "xy_shape": list(fix["xy"].shape),
        "time_shape": list(fix["time"].shape),
        "dwell_shape": list(fix["dwell"].shape),
        "seg_hits_shape": list(fix["seg_hits"].shape),
        "box_hits_shape": list(fix["box_hits"].shape),
        "images_shape": list(batch["images"].shape),
        "labels_binary_shape": list(batch["labels"]["binary"].shape),
        "transcripts_segments": [len((t or {}).get("segments", [])) for t in batch["transcripts"]],
        "lengths": fix["lengths"].tolist(),
        "dicom_ids": batch["dicom_ids"],
        "transcript_sample": {
            "text": transcript_text,
            "segments_head": transcript_segments,
        },
    }


def main() -> None:
    args = parse_args()
    config_loader = ConfigLoader(args.config_path)

    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    split_dir = Path(
        config_loader.get("split_files", "dir", default=ROOT / "config" / "splits")
    )
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_file = split_dir / f"{args.split}_ids.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    case_ids = read_split_ids(split_file, limit=args.limit_cases)

    dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=case_ids,
    )

    shuffle = args.shuffle or args.split == "train"
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )

    print(
        f"Split '{args.split}': {len(case_ids)} ids -> dataset {len(dataset)} samples | "
        f"batch_size={args.batch_size} | shuffle={shuffle}"
    )

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        summary = serialisable_batch_summary(batch)
        if args.print_raw:
            fix = batch["fixations"]
            lengths = fix["lengths"].tolist()
            sample_len = lengths[0] if lengths else 0
            head = min(sample_len, 5)
            raw_payload = {
                "dicom_id": batch["dicom_ids"][0] if batch["dicom_ids"] else None,
                "length": sample_len,
                "xy_head": fix["xy"][0, :head].tolist() if head else [],
                "dwell_head": fix["dwell"][0, :head].tolist() if head else [],
                "time_head": fix["time"][0, :head].tolist() if head else [],
                "seg_hits_head": fix["seg_hits"][0, :head].tolist() if head else [],
                "box_hits_head": fix["box_hits"][0, :head].tolist() if head else [],
                "transcript": batch["transcripts"][0] if batch["transcripts"] else {},
            }
            print(json.dumps({"batch_idx": batch_idx, "raw_sample": raw_payload}, indent=2))
        if args.show_json:
            print(json.dumps({"batch_idx": batch_idx, **summary}, indent=2))
        else:
            print(
                f"[batch {batch_idx}] "
                f"ids={summary['dicom_ids']} "
                f"xy={summary['xy_shape']} seg_hits={summary['seg_hits_shape']} "
                f"box_hits={summary['box_hits_shape']} labels={summary['labels_binary_shape']}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
