#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class CaseLabels:
    case_id: str
    final_diagnosis: Optional[str]
    diagnoses: List[str]
    binary_labels: Dict[str, int]
    cxr_exam_indication: Optional[str]
    source_csv: Path


@dataclass
class LabelSchema:
    """Schema describing how to build label vectors.
    - class_columns: explicit ordered list of binary columns to use
    """
    class_columns: List[str]

class LabelProcessor:
    """Process labels from master_sheet.csv, with utilities for dicts, vectors, and export.

    Provides:
      - discovery of binary label columns (Normal..support_devices__chx inclusive)
      - per-ID label dict and vector encoding
      - CaseLabels assembly (diagnoses, final diagnosis, etc.)
      - CSV export for all samples
    """

    def __init__(self, master_sheet_csv: Path, schema: Optional[LabelSchema] = None):
        self.master_sheet_csv = Path(master_sheet_csv)
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        self.df = pd.read_csv(self.master_sheet_csv)
        self.schema = schema or LabelSchema(class_columns=self.__discover_binary_columns(self.df))

    def get_labels(self, case_id: str) -> CaseLabels:
        row = self.__get_row(case_id)
        diagnoses = self.__extract_diagnoses(row)
        final_diagnosis = diagnoses[0] if diagnoses else None
        binary_labels = self.__labels_dict(case_id)
        return CaseLabels(
            case_id=case_id,
            final_diagnosis=final_diagnosis,
            diagnoses=diagnoses,
            binary_labels=binary_labels,
            cxr_exam_indication=row.get("cxr_exam_indication"),
            source_csv=self.master_sheet_csv,
        )

    @staticmethod
    def __discover_binary_columns(df: pd.DataFrame) -> List[str]:
        cols = list(df.columns)
        if "Normal" in cols and "support_devices__chx" in cols:
            si = cols.index("Normal")
            ei = cols.index("support_devices__chx")
            if si <= ei:
                return cols[si : ei + 1]
        return []

    def __get_row(self, case_id: str) -> pd.Series:
        mask = self.df["dicom_id"] == case_id
        if not mask.any():
            raise ValueError(f"Case {case_id} not found in {self.master_sheet_csv}")
        return self.df.loc[mask].iloc[0]

    @staticmethod
    def __extract_diagnoses(case_row: pd.Series) -> List[str]:
        dx_columns = [col for col in case_row.index if col.startswith("dx") and "_icd" not in col]
        diagnoses: List[str] = []
        for col in sorted(dx_columns, key=lambda name: int(name[2:]) if name[2:].isdigit() else 0):
            value = case_row[col]
            if isinstance(value, str) and value.strip():
                diagnoses.append(value.strip())
        return diagnoses

    def __labels_dict(self, case_id: str) -> Dict[str, int]:
        row = self.__get_row(case_id)
        cols = self.schema.class_columns
        labels: Dict[str, int] = {}
        for col in cols:
            if col not in row.index:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            try:
                labels[col] = int(val)
            except (TypeError, ValueError):
                labels[col] = 1 if str(val).strip() not in ("", "0", "nan", "None") else 0
        return labels

    