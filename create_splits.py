#!/usr/bin/env python3
"""
Generate train/val/test splits for EGD-CXR case IDs using the rewritten dataset.

This script creates deterministic splits based on the classification logic from the notebook,
ensuring proper distribution of CHF, pneumonia, and Normal cases across splits.

Example:
    python create_splits.py --config-path configs/data_egd_cxr_single_label.yaml \
        --output-dir configs/splits --train 0.7 --val 0.1 --test 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader  # noqa: E402
from egd_cxr_dataset.split import SplitConfig  # noqa: E402
from src.datasets.egd_cxr_rewritten import EGDCXRRewrittenDataset  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "data_egd_cxr_single_label.yaml"
DEFAULT_OUTPUT = ROOT / "configs" / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits for EGD-CXR IDs using rewritten dataset."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to dataset configuration YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory where split files will be written (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Proportion of IDs assigned to the training split (default: 0.7).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Proportion of IDs assigned to the validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.2,
        help="Proportion of IDs assigned to the test split (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used for shuffling IDs (default: 17).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print JSON summary of counts and file paths after writing splits.",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify splits by class to ensure balanced distribution.",
    )
    return parser.parse_args()


def ratios_from_args(args: argparse.Namespace) -> Tuple[float, float, float]:
    ratios = (args.train, args.val, args.test)
    if any(r < 0 for r in ratios):
        raise ValueError("Split proportions must be non-negative.")
    if sum(ratios) == 0:
        raise ValueError("At least one split proportion must be positive.")
    return ratios


def load_dataset_with_classification(config_path: Path) -> Tuple[EGDCXRRewrittenDataset, List[str]]:
    """Load the rewritten dataset and get all valid case IDs."""
    config_loader = ConfigLoader(config_path)
    
    # Get paths from config
    root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_path = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_path = Path(config_loader.get("input_path", "transcripts_dir", default=seg_path))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))
    
    # Create dataset to get valid case IDs
    dataset = EGDCXRRewrittenDataset(
        root=root,
        seg_path=seg_path,
        transcripts_path=transcripts_path,
        dicom_root=dicom_root,
        classes=["CHF", "pneumonia", "Normal"],
        drop_unlabelled=True,
    )
    
    # Get all case IDs from the dataset
    case_ids = []
    for idx in range(len(dataset)):
        master_idx = dataset._indices[idx]
        row = dataset.master_df.iloc[master_idx]
        case_ids.append(row["dicom_id"])
    
    return dataset, case_ids


def create_stratified_splits(
    dataset: EGDCXRRewrittenDataset, 
    case_ids: List[str], 
    ratios: Tuple[float, float, float], 
    seed: int
) -> Dict[str, List[str]]:
    """Create stratified splits ensuring balanced class distribution."""
    np.random.seed(seed)
    
    # Group case IDs by class
    class_to_ids = {"CHF": [], "pneumonia": [], "Normal": []}
    
    for idx in range(len(dataset)):
        master_idx = dataset._indices[idx]
        row = dataset.master_df.iloc[master_idx]
        dicom_id = row["dicom_id"]
        target = dataset._targets[idx]
        class_name = dataset.classes[target]
        class_to_ids[class_name].append(dicom_id)
    
    # Create splits for each class
    splits = {"train": [], "val": [], "test": []}
    
    for class_name, ids in class_to_ids.items():
        if not ids:
            continue
            
        # Shuffle IDs for this class
        np.random.shuffle(ids)
        
        # Calculate split sizes for this class
        total = len(ids)
        train_size = int(total * ratios[0])
        val_size = int(total * ratios[1])
        test_size = total - train_size - val_size
        
        # Split the IDs
        train_ids = ids[:train_size]
        val_ids = ids[train_size:train_size + val_size]
        test_ids = ids[train_size + val_size:]
        
        # Add to overall splits
        splits["train"].extend(train_ids)
        splits["val"].extend(val_ids)
        splits["test"].extend(test_ids)
    
    # Shuffle each split to avoid class clustering
    for split_name in splits:
        np.random.shuffle(splits[split_name])
    
    return splits


def create_random_splits(
    case_ids: List[str], 
    ratios: Tuple[float, float, float], 
    seed: int
) -> Dict[str, List[str]]:
    """Create random splits without stratification."""
    np.random.seed(seed)
    
    # Shuffle all IDs
    shuffled_ids = case_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    # Calculate split sizes
    total = len(shuffled_ids)
    train_size = int(total * ratios[0])
    val_size = int(total * ratios[1])
    test_size = total - train_size - val_size
    
    # Split the IDs
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:]
    
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }


def write_split_files(splits: Dict[str, List[str]], output_dir: Path) -> Dict[str, Path]:
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


def print_split_statistics(dataset: EGDCXRRewrittenDataset, splits: Dict[str, List[str]]) -> None:
    """Print detailed statistics about the splits."""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    # Calculate class distribution for each split
    for split_name, split_ids in splits.items():
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"  Total cases: {len(split_ids)}")
        
        # Count classes in this split
        class_counts = {"CHF": 0, "pneumonia": 0, "Normal": 0}
        
        for dicom_id in split_ids:
            # Find the case in the dataset
            for idx in range(len(dataset)):
                master_idx = dataset._indices[idx]
                row = dataset.master_df.iloc[master_idx]
                if row["dicom_id"] == dicom_id:
                    target = dataset._targets[idx]
                    class_name = dataset.classes[target]
                    class_counts[class_name] += 1
                    break
        
        # Print class distribution
        total_cases = len(split_ids)
        for class_name, count in class_counts.items():
            percentage = (count / total_cases * 100) if total_cases > 0 else 0
            print(f"    {class_name:10s}: {count:4d} cases ({percentage:5.1f}%)")
    
    # Print overall totals
    print(f"\nOVERALL TOTALS:")
    total_cases = sum(len(split_ids) for split_ids in splits.values())
    print(f"  Total cases: {total_cases}")
    
    for split_name, split_ids in splits.items():
        percentage = (len(split_ids) / total_cases * 100) if total_cases > 0 else 0
        print(f"    {split_name:5s}: {len(split_ids):4d} cases ({percentage:5.1f}%)")


def main() -> None:
    args = parse_args()
    ratios = ratios_from_args(args)

    print("Loading dataset and extracting valid case IDs...")
    dataset, case_ids = load_dataset_with_classification(args.config_path)
    
    print(f"Found {len(case_ids)} valid cases with complete data")
    print(f"Dataset class distribution: {dataset.class_counts.tolist()}")
    
    # Create splits
    if args.stratify:
        print("Creating stratified splits...")
        splits = create_stratified_splits(dataset, case_ids, ratios, args.seed)
    else:
        print("Creating random splits...")
        splits = create_random_splits(case_ids, ratios, args.seed)

    # Write split files
    output_dir = args.output_dir
    if args.output_dir == DEFAULT_OUTPUT:
        configured_dir = ConfigLoader(args.config_path).get("split_files", "dir", default=None)
        if configured_dir is not None:
            output_dir = Path(configured_dir)
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
    output_dir = Path(output_dir)

    written_paths = write_split_files(splits, output_dir)

    print("\nCreated splits:")
    for name, path in written_paths.items():
        print(f"  {name:5s}: {len(splits[name]):4d} ids -> {path}")

    # Print detailed statistics
    print_split_statistics(dataset, splits)

    if args.print_summary:
        summary: Dict[str, Dict[str, object]] = {}
        for name in ["train", "val", "test"]:
            summary[name] = {
                "count": len(splits[name]),
                "file": str(written_paths[name]),
            }
        print("\nJSON Summary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()