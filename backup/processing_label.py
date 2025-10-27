#!/usr/bin/env python3
"""
Lookup clinical labels for an EGD-CXR case from the dataset master sheet.

By default the script targets the sample case used throughout this repository,
so running:

    python processing_label.py

will print the structured labels (textual diagnoses and binary findings) for
the exemplar study, provided the dataset is mounted at the expected path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0")
DEFAULT_MASTER_SHEET = DEFAULT_DATA_ROOT / "master_sheet.csv"
DEFAULT_CASE_ID = "24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4"
DEFAULT_OUTPUT_ROOT = ROOT / "plots" / "labels"


@dataclass
class CaseLabels:
    case_id: str
    final_diagnosis: Optional[str]
    diagnoses: List[str]
    binary_labels: Dict[str, int]
    cxr_exam_indication: Optional[str]
    source_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clinical labels for an EGD-CXR case from master_sheet.csv."
    )
    parser.add_argument(
        "--case-id",
        default=DEFAULT_CASE_ID,
        help=f"Case identifier to process (default: {DEFAULT_CASE_ID}).",
    )
    parser.add_argument(
        "--master-sheet",
        type=Path,
        default=DEFAULT_MASTER_SHEET,
        help=f"Path to master_sheet.csv (default: {DEFAULT_MASTER_SHEET}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for saving extracted labels as JSON (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Persist the extracted labels to <output-root>/<case-id>/labels.json.",
    )
    return parser.parse_args()


def load_case_row(master_sheet: Path, case_id: str) -> pd.Series:
    if not master_sheet.exists():
        raise FileNotFoundError(f"master_sheet.csv not found at {master_sheet}")

    df = pd.read_csv(master_sheet)
    mask = df["dicom_id"] == case_id
    if not mask.any():
        raise ValueError(f"Case {case_id} not found in {master_sheet}")
    return df.loc[mask].iloc[0]


def extract_diagnoses(case_row: pd.Series) -> List[str]:
    dx_columns = [col for col in case_row.index if col.startswith("dx") and "_icd" not in col]
    diagnoses: List[str] = []
    for col in sorted(dx_columns, key=lambda name: int(name[2:]) if name[2:].isdigit() else 0):
        value = case_row[col]
        if isinstance(value, str) and value.strip():
            diagnoses.append(value.strip())
    return diagnoses


def extract_binary_labels(case_row: pd.Series) -> Dict[str, int]:
    columns = list(case_row.index)
    if "Normal" not in columns or "support_devices__chx" not in columns:
        return {}
    start_idx = columns.index("Normal")
    end_idx = columns.index("support_devices__chx")
    binary_section = columns[start_idx : end_idx + 1]

    labels: Dict[str, int] = {}
    for col in binary_section:
        value = case_row[col]
        if pd.isna(value):
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        labels[col] = numeric
    return labels


def assemble_case_labels(case_id: str, master_sheet: Path) -> CaseLabels:
    row = load_case_row(master_sheet, case_id)
    diagnoses = extract_diagnoses(row)
    final_diagnosis = diagnoses[0] if diagnoses else None
    binary_labels = extract_binary_labels(row)

    return CaseLabels(
        case_id=case_id,
        final_diagnosis=final_diagnosis,
        diagnoses=diagnoses,
        binary_labels=binary_labels,
        cxr_exam_indication=row.get("cxr_exam_indication"),
        source_csv=master_sheet,
    )


def write_labels(output_root: Path, case_labels: CaseLabels) -> Path:
    output_dir = output_root / case_labels.case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "labels.json"
    payload = asdict(case_labels)
    payload["source_csv"] = str(case_labels.source_csv)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()

    case_labels = assemble_case_labels(args.case_id, args.master_sheet)

    print(f"Case ID: {case_labels.case_id}")
    print(f"Final diagnosis: {case_labels.final_diagnosis or 'N/A'}")
    if case_labels.diagnoses:
        print("All diagnoses:")
        for idx, diagnosis in enumerate(case_labels.diagnoses, start=1):
            print(f"  {idx}. {diagnosis}")
    if case_labels.binary_labels:
        positives = [k for k, v in case_labels.binary_labels.items() if v == 1]
        print(f"Positive binary labels ({len(positives)}): {', '.join(positives) if positives else 'None'}")
    if case_labels.cxr_exam_indication:
        print(f"CXR exam indication: {case_labels.cxr_exam_indication}")

    if args.write_json:
        output_path = write_labels(args.output_root, case_labels)
        print(f"Wrote labels JSON to {output_path}")


if __name__ == "__main__":
    main()
