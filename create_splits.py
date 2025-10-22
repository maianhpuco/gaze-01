#!/usr/bin/env python3
"""
Generate train/val/test splits for EGD-CXR case IDs.

Example:
    python create_splits.py --config-path config/data_egd-cxr.yaml \
        --output-dir config/splits --train 0.7 --val 0.1 --test 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import (  # noqa: E402
    ConfigLoader,
    SplitConfig,
)
from egd_cxr_dataset.split import load_case_ids, split_ids, write_split_files  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "data_egd-cxr.yaml"
DEFAULT_OUTPUT = ROOT / "config" / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits for EGD-CXR IDs."
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
        "--column",
        type=str,
        default="dicom_id",
        help="Column name inside master_sheet.csv containing unique IDs (default: dicom_id).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print JSON summary of counts and file paths after writing splits.",
    )
    return parser.parse_args()


def ratios_from_args(args: argparse.Namespace) -> Tuple[float, float, float]:
    ratios = (args.train, args.val, args.test)
    if any(r < 0 for r in ratios):
        raise ValueError("Split proportions must be non-negative.")
    if sum(ratios) == 0:
        raise ValueError("At least one split proportion must be positive.")
    return ratios


def main() -> None:
    args = parse_args()
    ratios = ratios_from_args(args)

    config_loader = ConfigLoader(args.config_path)
    gaze_root = config_loader.get("input_path", "gaze_raw")
    if gaze_root is None:
        raise ValueError("Configuration missing 'input_path.gaze_raw'.")
    master_sheet = Path(gaze_root).expanduser() / "master_sheet.csv"

    ids = load_case_ids(master_sheet, column=args.column)
    split_config = SplitConfig(ratios=ratios, seed=args.seed, column=args.column)
    splits = split_ids(ids, split_config)

    output_dir = args.output_dir
    if args.output_dir == DEFAULT_OUTPUT:
        configured_dir = config_loader.get("split_files", "dir", default=None)
        if configured_dir is not None:
            output_dir = Path(configured_dir)
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
    output_dir = Path(output_dir)

    written_paths = write_split_files(splits, output_dir)

    print(
        "Created splits:\n"
        + "\n".join(
            f"  {name:5s}: {len(splits[name]):4d} ids -> {path}"
            for name, path in written_paths.items()
        )
    )

    if args.print_summary:
        summary: Dict[str, Dict[str, object]] = {}
        for name in split_config.names:
            summary[name] = {
                "count": len(splits[name]),
                "file": str(written_paths[name]),
            }
        print("\nSummary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
