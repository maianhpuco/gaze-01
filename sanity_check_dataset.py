#!/usr/bin/env python3
"""
Dump EGD-CXR dataset samples to JSON for quick sanity checking.

Example:
    python sanity_check_dataset.py --config-path config/data_egd-cxr.yaml \
        --split train --batch-size 1 --num-batches 1 --max-fixations 5 \
        --output-dir sanity_check/dataset --show-json --print-raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader
from egd_cxr_dataset.datasets import EGDCXRDataset, create_dataloader

DEFAULT_CONFIG = ROOT / "config" / "data_egd-cxr.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit dataset samples as JSON for sanity checking.")
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
        help="Emit batch summary as JSON to stdout.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print a small raw sample (first case) to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sanity_check" / "dataset",
        help="Directory to store per-case JSON dumps (default: sanity_check/dataset).",
    )
    return parser.parse_args()


def read_split_ids(split_file: Path, limit: int | None = None) -> List[str]:
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return ids[:limit]
    return ids


def case_payload(batch: Dict, idx: int) -> Dict:
    fix = batch["fixations"]
    length = int(fix["lengths"][idx].item())
    xy = fix["xy"][idx, :length]
    dwell = fix["dwell"][idx, :length]
    time_s = fix["time"][idx, :length]
    seg_hits = fix["seg_hits"][idx, :length]
    box_hits = fix["box_hits"][idx, :length]
    image = batch["images"][idx]
    head = min(length, 5)

    payload = {
        "dicom_id": batch["dicom_ids"][idx],
        "length": length,
        "fixations": {
            "xy_head": xy[:head].tolist(),
            "dwell_head": dwell[:head].tolist(),
            "time_head": time_s[:head].tolist(),
            "seg_hits_head": seg_hits[:head].tolist(),
            "box_hits_head": box_hits[:head].tolist(),
        },
        "image": {
            "shape": list(image.shape),
            "min": float(image.min().item()),
            "max": float(image.max().item()),
            "mean": float(image.mean().item()),
        },
        "labels": {
            "binary": batch["labels"]["binary"][idx].tolist(),
            "binary_names": batch["labels"]["binary_names"],
            "final_diagnosis": batch["labels"]["final_diagnosis"][idx],
            "diagnoses": batch["labels"]["diagnoses"][idx],
        },
        "transcript": batch["transcripts"][idx],
        "meta": batch["meta"],
    }
    return payload


def serialisable_batch_summary(batch: Dict) -> Dict:
    fix = batch["fixations"]
    lengths = fix["lengths"]
    return {
        "batch_size": int(fix["xy"].shape[0]),
        "seq_len_max": int(fix["xy"].shape[1]),
        "lengths": lengths.tolist(),
        "xy_shape": list(fix["xy"].shape),
        "seg_hits_shape": list(fix["seg_hits"].shape),
        "box_hits_shape": list(fix["box_hits"].shape),
        "images_shape": list(batch["images"].shape),
        "labels_binary_shape": list(batch["labels"]["binary"].shape),
        "dicom_ids": batch["dicom_ids"],
    }


def maybe_print_raw(batch: Dict) -> None:
    fix = batch["fixations"]
    lengths = fix["lengths"]
    if lengths.numel() == 0:
        return
    idx = 0
    length = int(lengths[idx].item())
    head = min(length, 5)
    payload = {
        "dicom_id": batch["dicom_ids"][idx],
        "length": length,
        "xy_head": fix["xy"][idx, :head].tolist(),
        "dwell_head": fix["dwell"][idx, :head].tolist(),
        "time_head": fix["time"][idx, :head].tolist(),
        "seg_hits_head": fix["seg_hits"][idx, :head].tolist(),
        "box_hits_head": fix["box_hits"][idx, :head].tolist(),
        "transcript": batch["transcripts"][idx],
    }
    print(json.dumps({"raw_sample": payload}, indent=2))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_cases(batch: Dict, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for idx, dicom_id in enumerate(batch["dicom_ids"]):
        payload = case_payload(batch, idx)
        path = output_dir / f"{dicom_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    config_loader = ConfigLoader(args.config_path)

    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    split_dir = Path(config_loader.get("split_files", "dir", default=ROOT / "config" / "splits"))
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

    ensure_dir(args.output_dir)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        summary = serialisable_batch_summary(batch)
        if args.print_raw:
            maybe_print_raw(batch)
        if args.show_json:
            print(json.dumps({"batch_idx": batch_idx, **summary}, indent=2))
        else:
            print(
                f"[batch {batch_idx}] ids={summary['dicom_ids']} "
                f"xy_shape={summary['xy_shape']} seg_hits_shape={summary['seg_hits_shape']} "
                f"box_hits_shape={summary['box_hits_shape']} labels_shape={summary['labels_binary_shape']}"
            )
        dump_cases(batch, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

