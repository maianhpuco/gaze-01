#!/usr/bin/env python3
"""
Dataset and DataLoader helpers for the EGD-CXR multimodal dataset.

This module implements a comprehensive multimodal dataset for eye-gaze tracking research
on chest X-ray (CXR) images. It combines multiple data modalities:

1. Eye gaze tracking data (fixations, coordinates, timing)
2. Chest X-ray images (DICOM format)
3. Anatomical segmentation masks
4. Bounding box annotations for abnormalities
5. Radiologist text transcripts/reports
6. Diagnostic labels and metadata

The dataset enables research on radiologist attention patterns, multimodal diagnosis,
and attention-guided AI systems for medical imaging.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore[assignment]
    HAS_PYDICOM = False

# Configure threading to avoid conflicts in multiprocessing environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Column names for eye tracking data in the fixations CSV
TIME_COLUMN = "Time (in secs)"      # Timestamp of fixation
X_COLUMN = "FPOGX"                  # Normalized X coordinate (0-1)
Y_COLUMN = "FPOGY"                  # Normalized Y coordinate (0-1)
DURATION_COLUMN = "FPOGD"           # Fixation duration in seconds


class Logger:
    """
    Minimal stdout logger for dataset operations.
    
    Provides simple info and error logging without external dependencies.
    Used throughout the dataset for status updates and error reporting.
    """

    @staticmethod
    def info(message: str) -> None:
        """Print informational message."""
        print(message)

    @staticmethod
    def error(message: str) -> None:
        """Print error message with warning symbol."""
        print(f"⚠ {message}")


@dataclass(frozen=True)
class LabelSchema:
    """
    Schema defining which columns in master_sheet.csv contain binary classification labels.
    
    Attributes:
        class_columns: List of column names that contain binary (0/1) disease labels
    """
    class_columns: List[str]


class LabelProcessor:
    """
    Processes diagnostic labels from master_sheet.csv into machine learning format.
    
    This class handles the conversion of medical diagnostic data into:
    - Binary classification tensors for multi-label disease prediction
    - Structured diagnosis metadata (primary diagnosis, all diagnoses)
    - Raw row data for additional analysis
    
    The master_sheet.csv typically contains columns for different diseases/conditions
    with binary values (0=absent, 1=present) for each case.
    """

    def __init__(self, master_sheet_csv: Path, schema: Optional[LabelSchema] = None):
        """
        Initialize the label processor.
        
        Args:
            master_sheet_csv: Path to the master sheet CSV file containing diagnostic labels
            schema: Optional schema defining which columns are binary labels.
                   If None, will auto-discover binary columns.
        """
        self.master_sheet_csv = Path(master_sheet_csv).expanduser()
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        
        # Load the master sheet data
        self.df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.df.empty:
            raise ValueError(f"master_sheet.csv has no rows: {self.master_sheet_csv}")
        
        # Auto-discover binary columns if no schema provided
        self.schema = schema or LabelSchema(class_columns=self._discover_binary_columns(self.df))

    @staticmethod
    def _discover_binary_columns(df: pd.DataFrame) -> List[str]:
        """
        Automatically discover which columns contain binary classification labels.
        
        Uses heuristics to identify columns with binary (0/1) values:
        1. If 'Normal' and 'support_devices__chx' columns exist, assumes all columns
           between them are binary labels (common pattern in medical datasets)
        2. Otherwise, finds all columns where all non-null values are 0 or 1
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that appear to contain binary labels
        """
        cols = list(df.columns)
        
        # Check for common medical dataset pattern: Normal to support_devices__chx
        if "Normal" in cols and "support_devices__chx" in cols:
            si = cols.index("Normal")
            ei = cols.index("support_devices__chx")
            if si <= ei:
                return cols[si : ei + 1]
        
        # Fallback: find columns with only 0/1 values
        fallback: List[str] = []
        for col in cols:
            series = df[col].dropna()
            if not series.empty and series.isin([0, 1]).all():
                fallback.append(col)
        return fallback

    def _get_row(self, case_id: str) -> pd.Series:
        """
        Retrieve the data row for a specific case ID.
        
        Args:
            case_id: DICOM ID of the case to retrieve
            
        Returns:
            pandas Series containing all data for the case
            
        Raises:
            ValueError: If case_id is not found in the dataset
        """
        mask = self.df["dicom_id"] == case_id
        if not mask.any():
            raise ValueError(f"dicom_id {case_id} not found in {self.master_sheet_csv}")
        return self.df.loc[mask].iloc[0]

    def diagnoses(self, case_id: str) -> Tuple[Optional[str], List[str]]:
        """
        Extract diagnosis information for a case.
        
        Looks for columns starting with 'dx' (diagnosis) and extracts text diagnoses.
        Returns the primary diagnosis (first one) and all diagnoses.
        
        Args:
            case_id: DICOM ID of the case
            
        Returns:
            Tuple of (primary_diagnosis, all_diagnoses_list)
            - primary_diagnosis: First diagnosis found (or None if none)
            - all_diagnoses_list: List of all non-empty diagnoses
        """
        row = self._get_row(case_id)
        
        # Find all diagnosis columns (dx1, dx2, dx3, etc.)
        dx_cols = [
            col for col in row.index if col.startswith("dx") and not col.endswith("_icd")
        ]
        
        diagnoses: List[str] = []
        # Sort by column number (dx1, dx2, dx3...) to maintain order
        for col in sorted(dx_cols, key=lambda name: int(name[2:]) if name[2:].isdigit() else 0):
            value = row[col]
            if isinstance(value, str) and value.strip():
                diagnoses.append(value.strip())
        
        # Primary diagnosis is the first one
        final_dx = diagnoses[0] if diagnoses else None
        return final_dx, diagnoses

    def vector(self, case_id: str) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        """
        Convert case labels to a binary classification tensor.
        
        Creates a multi-label binary vector where each element corresponds to
        a disease/condition (1=present, 0=absent). Handles missing values and
        various data types gracefully.
        
        Args:
            case_id: DICOM ID of the case
            
        Returns:
            Tuple of:
            - binary_tensor: PyTorch tensor of binary labels (float32)
            - column_names: List of column names corresponding to tensor elements
            - raw_data: Dictionary of all raw row data for additional analysis
        """
        row = self._get_row(case_id)
        values: List[int] = []
        
        # Process each binary classification column
        for col in self.schema.class_columns:
            value = row.get(col, np.nan)
            
            # Handle missing values as negative (0)
            if pd.isna(value):
                values.append(0)
                continue
            
            # Convert to integer, handling various data types
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                # For non-numeric values, treat non-empty strings as positive (1)
                values.append(1 if str(value).strip() not in ("", "0", "nan", "None") else 0)
        
        # Create tensor and return with metadata
        tensor = torch.tensor(values, dtype=torch.float32)
        return tensor, self.schema.class_columns, row.to_dict()


@dataclass(frozen=True)
class BoxRow:
    """
    Represents a bounding box annotation for an abnormality or region of interest.
    
    Attributes:
        x1: Left edge coordinate (inclusive)
        y1: Top edge coordinate (inclusive)  
        x2: Right edge coordinate (exclusive)
        y2: Bottom edge coordinate (exclusive)
        cls_id: Integer class ID for the box type/abnormality
        cls_name: String name of the box class (e.g., "pneumonia", "nodule")
    """
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    cls_name: str


class EGDCXRDataset(Dataset):
    """
    Multimodal dataset for EGD-CXR combining gaze, segmentations, boxes, transcripts, and labels.
    
    This is the main dataset class that integrates multiple data modalities for eye-gaze
    tracking research on chest X-ray images. It provides a unified interface to access:
    
    Data Modalities:
    - Eye gaze tracking data (fixation coordinates, timing, duration)
    - Chest X-ray images (DICOM format with proper windowing)
    - Anatomical segmentation masks (organ/region boundaries)
    - Bounding box annotations (abnormalities and regions of interest)
    - Radiologist text transcripts/reports
    - Diagnostic labels (binary disease classifications)
    
    Key Features:
    - Automatic gaze-to-segment mapping (which anatomical regions were looked at)
    - Gaze-to-box mapping (which abnormalities were fixated on)
    - Variable-length sequence handling with proper padding
    - Support for multiple data storage formats (CSV, JSON, PNG, NPY/NPZ)
    - Comprehensive error handling and validation
    
    The dataset enables research on radiologist attention patterns, multimodal diagnosis,
    and attention-guided AI systems for medical imaging.
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
    ):
        """
        Initialize the EGD-CXR multimodal dataset.
        
        Args:
            root: Path to the main dataset directory containing:
                 - fixations.csv (eye tracking data)
                 - master_sheet.csv (diagnostic labels)
                 - bounding_boxes.csv (abnormality annotations)
            seg_path: Path to segmentation data (PNG files or NPY/NPZ arrays)
            transcripts_path: Path to transcript data (CSV file or JSON directory).
                            If None, uses seg_path.
            dicom_root: Optional path to DICOM image files (.dcm)
            max_fixations: Optional limit on number of fixations per case
            case_ids: Optional list of specific case IDs to include.
                     If None, includes all cases with complete data.
        
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If no valid cases are found or data is invalid
        """
        # Set up main dataset paths
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Gaze dataset root not found: {self.root}")

        # Define paths to required CSV files
        self.fixations_csv = self.root / "fixations.csv"           # Eye tracking data
        self.master_sheet_csv = self.root / "master_sheet.csv"     # Diagnostic labels
        self.bounding_boxes_csv = self.root / "bounding_boxes.csv" # Abnormality annotations

        # Set up segmentation and transcript paths
        self.seg_path = Path(seg_path).expanduser()
        if transcripts_path is None:
            transcripts_path = seg_path
        self.transcripts_path = Path(transcripts_path).expanduser()
        self.dicom_root = Path(dicom_root).expanduser() if dicom_root is not None else None

        # Validate that all required paths exist
        for path in [
            self.fixations_csv,
            self.master_sheet_csv,
            self.bounding_boxes_csv,
            self.seg_path,
            self.transcripts_path,
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing path: {path}")
        if self.dicom_root is not None and not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM root not found: {self.dicom_root}")

        # Store configuration parameters
        self.max_fix = max_fixations
        
        # Load and validate CSV data
        self.ms_df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.ms_df.empty:
            raise ValueError("master_sheet.csv empty")

        self.fx_df = pd.read_csv(self.fixations_csv, engine="python")
        # Bounding boxes CSV is optional
        self.bb_df = pd.read_csv(self.bounding_boxes_csv, engine="python") if self.bounding_boxes_csv.exists() else pd.DataFrame()

        # Determine transcript storage format and load data
        self.transcripts_mode = "csv" if self.transcripts_path.is_file() else "directory"
        if self.transcripts_mode == "csv":
            self.tr_df = pd.read_csv(self.transcripts_path, engine="python")
        else:
            self.tr_df = None

        # Determine segmentation storage format
        self.seg_mode = "directory" if self.seg_path.is_dir() else "arrays"
        if self.seg_mode not in {"directory", "arrays"}:
            raise ValueError("Unsupported segmentation storage format.")

        # Discover available regions and build label mappings
        self.region_names = self._discover_region_names() if self.seg_mode == "directory" else []
        self.box_label_to_idx = self._build_box_label_mapping()
        self.box_class_names = [name for name, _ in sorted(self.box_label_to_idx.items(), key=lambda kv: kv[1])]
        self.num_box_classes = len(self.box_class_names)
        self.num_segments: Optional[int] = len(self.region_names) if self.region_names else None

        # Find intersection of cases across all modalities
        fx_cases = set(self.fx_df["DICOM_ID"].dropna().astype(str))      # Eye tracking cases
        ms_cases = set(self.ms_df["dicom_id"].dropna().astype(str))      # Label cases
        transcript_cases = self._discover_transcript_case_ids()          # Transcript cases
        seg_cases = self._discover_segmentation_case_ids()               # Segmentation cases

        # Only include cases that have all required modalities
        base_cases = sorted(ms_cases & fx_cases & transcript_cases & seg_cases)
        
        # Filter to requested case IDs if specified
        if case_ids is not None:
            requested = set(case_ids)
            missing = requested - set(base_cases)
            if missing:
                Logger.error(f"{len(missing)} requested IDs missing required modalities; they will be skipped.")
            base_cases = [cid for cid in base_cases if cid in requested]

        if not base_cases:
            raise ValueError("No cases found with all required modalities.")

        # Store final case list and initialize label processor
        self.case_ids = base_cases
        self.label_proc = LabelProcessor(self.master_sheet_csv)

        Logger.info(
            f"EGDCXRDataset ready: {len(self.case_ids)} cases | "
            f"regions={len(self.region_names) if self.region_names else 'n/a'} | "
            f"bbox_classes={len(self.box_label_to_idx)}"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def _discover_region_names(self) -> List[str]:
        names: set[str] = set()
        if not self.seg_path.is_dir():
            return []
        for case_dir in self.seg_path.iterdir():
            if not case_dir.is_dir():
                continue
            for png in case_dir.glob("*.png"):
                names.add(png.stem)
        return sorted(names)

    def _discover_transcript_case_ids(self) -> set[str]:
        if self.transcripts_mode == "csv":
            return set(self.tr_df["dicom_id"].dropna().astype(str))
        cases: set[str] = set()
        for path in self.transcripts_path.iterdir():
            if path.is_dir() and (path / "transcript.json").exists():
                cases.add(path.name)
        return cases

    def _discover_segmentation_case_ids(self) -> set[str]:
        cases: set[str] = set()
        if self.seg_mode == "directory":
            for path in self.seg_path.iterdir():
                if path.is_dir() and any(path.glob("*.png")):
                    cases.add(path.name)
        else:
            for file in self.seg_path.glob("*_segs.npz"):
                cases.add(file.stem.replace("_segs", ""))
            for file in self.seg_path.glob("*_segs.npy"):
                cases.add(file.stem.replace("_segs", ""))
        return cases

    def _build_box_label_mapping(self) -> Dict[str, int]:
        if self.bb_df.empty:
            return {}
        names = sorted(self.bb_df["bbox_name"].dropna().astype(str).unique())
        return {name: idx for idx, name in enumerate(names)}

    def _load_seg_masks(self, dicom_id: str) -> np.ndarray:
        if self.seg_mode == "arrays":
            npz = self.seg_path / f"{dicom_id}_segs.npz"
            npy = self.seg_path / f"{dicom_id}_segs.npy"
            if npz.exists():
                arr = np.load(npz)["masks"]
            elif npy.exists():
                arr = np.load(npy)
            else:
                raise FileNotFoundError(f"Segmentation array not found for {dicom_id}")
            return (arr > 0.5).astype(np.uint8)

        case_dir = self.seg_path / dicom_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Segmentation directory not found for {dicom_id}")

        region_names = self.region_names or sorted([png.stem for png in case_dir.glob("*.png")])
        if not region_names:
            raise ValueError(f"No segmentation PNGs found for {dicom_id}")

        reference_mask = None
        for png_name in region_names:
            png_path = case_dir / f"{png_name}.png"
            if png_path.exists():
                reference_mask = imageio.imread(png_path)
                break
        if reference_mask is None:
            raise ValueError(f"No segmentation PNGs found for {dicom_id}")
        height, width = reference_mask.shape[:2]
        masks = np.zeros((len(region_names), height, width), dtype=np.uint8)

        for idx, png_name in enumerate(region_names):
            png_path = case_dir / f"{png_name}.png"
            if not png_path.exists():
                continue
            img = imageio.imread(png_path)
            if img.ndim == 3:
                mask = img.max(axis=2) > 0
            else:
                mask = img > 0
            masks[idx] = mask.astype(np.uint8)

        background = (masks.sum(axis=0, keepdims=True) == 0).astype(np.uint8)
        stacked = np.concatenate([masks, background], axis=0)
        return stacked

    def _load_dicom_image(self, dicom_id: str) -> Optional[np.ndarray]:
        if self.dicom_root is None:
            return None
        dicom_path = self.dicom_root / f"{dicom_id}.dcm"
        if not dicom_path.exists():
            return None
        try:
            if HAS_PYDICOM:
                ds = pydicom.dcmread(str(dicom_path))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                def _window_value(tag: str) -> Optional[float]:
                    val = getattr(ds, tag, None)
                    if val is None:
                        return None
                    try:
                        if isinstance(val, (list, tuple)):
                            val = val[0]
                        return float(val)
                    except Exception:
                        return None

                center = _window_value("WindowCenter")
                width = _window_value("WindowWidth")
                if center is not None and width is not None and width > 0:
                    min_val = center - width / 2.0
                    max_val = center + width / 2.0
                    arr = np.clip(arr, min_val, max_val)
                    arr = (arr - min_val) / max((max_val - min_val), 1e-6)
                else:
                    arr -= arr.min()
                    max_val = arr.max()
                    if max_val > 0:
                        arr /= max_val
            else:
                arr = imageio.imread(dicom_path)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                arr = arr.astype(np.float32)
                arr -= arr.min()
                max_val = arr.max()
                if max_val > 0:
                    arr /= max_val
        except Exception:
            return None

        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)

    def _load_boxes(self, dicom_id: str) -> List[BoxRow]:
        if self.bb_df.empty:
            return []
        rows = self.bb_df[self.bb_df["dicom_id"] == dicom_id]
        processed: List[BoxRow] = []
        for _, row in rows.iterrows():
            name = str(row["bbox_name"])
            cls_id = self.box_label_to_idx.get(name, -1)
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            processed.append(
                BoxRow(
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    cls_id=cls_id,
                    cls_name=name,
                )
            )
        return processed

    def _load_transcript(self, dicom_id: str) -> Dict[str, Any]:
        if self.transcripts_mode == "csv":
            rows = self.tr_df[self.tr_df["dicom_id"] == dicom_id]
            if rows.empty:
                return {"text": "", "segments": []}
            text = str(rows.iloc[0].get("transcript", "") or "")
            return {"text": text, "segments": []}
        case_dir = self.transcripts_path / dicom_id
        transcript_file = case_dir / "transcript.json"
        if transcript_file.exists():
            data = json.loads(transcript_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                text = str(data.get("transcript") or data.get("full_text") or "").strip()
                segments = data.get("segments") or []
                return {"text": text, "segments": segments}
        return {"text": "", "segments": []}

    def _load_fixations(self, dicom_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and process eye tracking fixation data for a case.
        
        Filters and validates fixation data, ensuring coordinates are within bounds
        and durations are positive. Sorts by timestamp and optionally limits the
        number of fixations.
        
        Args:
            dicom_id: Case identifier
            
        Returns:
            Tuple of (times, xy_norm, dwell):
            - times: Timestamps in seconds (T,)
            - xy_norm: Normalized coordinates [0,1] (T, 2)
            - dwell: Fixation durations in milliseconds (T,)
            
        Raises:
            ValueError: If no valid fixations are found
        """
        df = self.fx_df[self.fx_df["DICOM_ID"] == dicom_id].copy()
        if df.empty:
            raise ValueError(f"No fixations for {dicom_id}")
        
        # Filter valid fixations (coordinates in bounds, positive duration)
        df = df[
            df[X_COLUMN].between(0.0, 1.0)      # X coordinate in [0,1]
            & df[Y_COLUMN].between(0.0, 1.0)    # Y coordinate in [0,1]
            & df[DURATION_COLUMN].notna()       # Duration not null
            & (df[DURATION_COLUMN] > 0)         # Duration positive
        ].copy()
        if df.empty:
            raise ValueError(f"Invalid fixations for {dicom_id}")
        
        # Sort by timestamp and counter for consistent ordering
        df.sort_values(by=[TIME_COLUMN, "CNT"], inplace=True, kind="mergesort")
        
        # Limit number of fixations if specified
        if self.max_fix is not None:
            df = df.iloc[: self.max_fix]
        
        # Extract arrays
        xy_norm = df[[X_COLUMN, Y_COLUMN]].to_numpy(dtype=np.float32)
        dwell = df[DURATION_COLUMN].to_numpy(dtype=np.float32) * 1000.0  # Convert to ms
        times = df[TIME_COLUMN].to_numpy(dtype=np.float32)
        return times, xy_norm, dwell

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a complete multimodal sample for the given index.
        
        This is the main method that loads and processes all data modalities for a single case.
        It performs several key operations:
        1. Loads segmentation masks and determines anatomical regions
        2. Loads bounding box annotations for abnormalities
        3. Loads radiologist transcript/report
        4. Processes diagnostic labels into binary tensors
        5. Loads and processes eye tracking data
        6. Maps gaze coordinates to anatomical regions and abnormalities
        7. Loads and preprocesses the chest X-ray image
        8. Returns a unified dictionary with all modalities
        
        Args:
            idx: Index of the case to retrieve
            
        Returns:
            Dictionary containing all data modalities:
            - dicom_id: Case identifier
            - image: Chest X-ray image tensor (1, 224, 224)
            - fixations: Dictionary with gaze data and mappings
            - transcript: Radiologist report text and segments
            - labels: Diagnostic labels and metadata
            - meta: Region names and class information
        """
        dicom_id = self.case_ids[idx]

        # Load segmentation masks and determine number of anatomical regions
        seg_masks_np = self._load_seg_masks(dicom_id)
        if self.num_segments is None:
            # Auto-discover number of segments (exclude background if present)
            if seg_masks_np.shape[0] > 1:
                self.num_segments = seg_masks_np.shape[0] - 1  # Exclude background
            else:
                self.num_segments = seg_masks_np.shape[0]
            if not self.region_names:
                self.region_names = [f"segment_{i}" for i in range(self.num_segments)]
        
        # Extract anatomical segments (exclude background)
        num_segments = self.num_segments or 0
        segments_np = seg_masks_np[:num_segments] if num_segments > 0 else np.zeros((0, *seg_masks_np.shape[1:]), dtype=np.uint8)

        # Load all other data modalities
        boxes = self._load_boxes(dicom_id)                                    # Abnormality bounding boxes
        transcript_payload = self._load_transcript(dicom_id)                  # Radiologist report
        labels_vec, label_names, labels_row = self.label_proc.vector(dicom_id)  # Binary disease labels
        final_dx, diagnoses = self.label_proc.diagnoses(dicom_id)             # Text diagnoses
        times_sec, xy_norm, dwell = self._load_fixations(dicom_id)            # Eye tracking data

        # Convert normalized gaze coordinates to pixel coordinates
        height, width = seg_masks_np.shape[1:]
        if xy_norm.size == 0:
            xy_px = np.zeros((0, 2), dtype=np.float32)
        else:
            # Convert from normalized [0,1] to pixel coordinates
            xy_px = np.stack(
                [
                    xy_norm[:, 0] * (width - 1),   # X coordinate
                    xy_norm[:, 1] * (height - 1),  # Y coordinate
                ],
                axis=1,
            ).astype(np.float32)

        # Create gaze-to-segment mapping: which anatomical regions were looked at?
        T = xy_px.shape[0]  # Number of fixations
        seg_hits = np.zeros((T, num_segments), dtype=np.float32)
        if num_segments > 0 and T > 0:
            # Round coordinates and clip to image bounds
            xs = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height - 1)
            # Check which segments contain each fixation point
            hits = segments_np[:, ys, xs] > 0
            seg_hits = hits.T.astype(np.float32)

        # Create gaze-to-box mapping: which abnormalities were looked at?
        num_box_classes = max(0, self.num_box_classes)
        box_hits = np.zeros((T, num_box_classes), dtype=np.float32)
        if num_box_classes > 0 and T > 0:
            xs_int = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width - 1)
            ys_int = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height - 1)
            for t in range(T):
                x = int(xs_int[t])
                y = int(ys_int[t])
                # Check if fixation point is inside any bounding box
                for box in boxes:
                    cls_id = box.cls_id
                    if 0 <= cls_id < num_box_classes and box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
                        box_hits[t, cls_id] = 1.0

        # Load and preprocess chest X-ray image
        image_arr = self._load_dicom_image(dicom_id)
        if image_arr is None:
            # Create dummy image if DICOM not available
            image_tensor = torch.zeros(1, 224, 224, dtype=torch.float32)
        else:
            # Convert to tensor and resize to standard size
            img_tensor = torch.from_numpy(image_arr).unsqueeze(0).float()
            image_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)

        # Prepare metadata for region and class names
        segment_names = self.region_names if self.region_names else [f"segment_{i}" for i in range(num_segments)]
        box_class_names = self.box_class_names if self.box_class_names else [f"class_{i}" for i in range(self.num_box_classes)]

        # Assemble the complete multimodal sample
        sample = {
            "dicom_id": dicom_id,                    # Case identifier
            "image": image_tensor,                   # Chest X-ray image (1, 224, 224)
            "fixations": {                          # Eye tracking data and mappings
                "xy": torch.from_numpy(xy_px.astype(np.float32)),           # Gaze coordinates (T, 2)
                "time": torch.from_numpy(times_sec.astype(np.float32)),     # Timestamps (T,)
                "dwell": torch.from_numpy(dwell.astype(np.float32)),        # Fixation durations (T,)
                "seg_hits": torch.from_numpy(seg_hits).float(),             # Segment hits (T, num_segments)
                "box_hits": torch.from_numpy(box_hits).float(),             # Box hits (T, num_box_classes)
            },
            "transcript": transcript_payload,        # Radiologist report
            "labels": {                             # Diagnostic labels
                "binary": labels_vec.float(),       # Binary disease labels
                "binary_names": label_names,        # Names of label columns
                "final_diagnosis": final_dx,        # Primary diagnosis text
                "diagnoses": diagnoses,             # All diagnoses list
                "raw_row": labels_row,              # Raw CSV row data
            },
            "meta": {                               # Metadata
                "segment_names": segment_names,     # Anatomical region names
                "box_class_names": box_class_names, # Abnormality class names
            },
        }
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom batch collation function for EGDCXRDataset.
    
    This function handles the batching of variable-length sequences (eye tracking data)
    by padding them to the same length. It also stacks fixed-size tensors and
    preserves metadata across the batch.
    
    Key operations:
    1. Pads variable-length fixation sequences to the same length
    2. Stacks fixed-size tensors (images, labels)
    3. Preserves sequence lengths for proper masking
    4. Maintains metadata consistency across the batch
    
    Args:
        batch: List of individual samples from the dataset
        
    Returns:
        Dictionary containing batched data with consistent tensor shapes
    """
    # Extract case IDs and sequence lengths
    dicom_ids = [item["dicom_id"] for item in batch]
    lengths = torch.tensor([item["fixations"]["xy"].shape[0] for item in batch], dtype=torch.long)

    # Pad variable-length sequences to the same length
    xy = pad_sequence([item["fixations"]["xy"] for item in batch], batch_first=True)
    dwell = pad_sequence([item["fixations"]["dwell"] for item in batch], batch_first=True)
    times = pad_sequence([item["fixations"]["time"] for item in batch], batch_first=True)
    seg_hits = pad_sequence([item["fixations"]["seg_hits"] for item in batch], batch_first=True)
    box_hits = pad_sequence([item["fixations"]["box_hits"] for item in batch], batch_first=True)

    # Stack fixed-size tensors
    images = torch.stack([item["image"] for item in batch], dim=0)
    transcripts = [item["transcript"] for item in batch]  # Keep as list (variable content)

    # Stack binary labels and preserve metadata
    labels_binary = torch.stack([item["labels"]["binary"] for item in batch], dim=0)
    labels_dict = {
        "binary": labels_binary,
        "binary_names": batch[0]["labels"]["binary_names"],  # Same across batch
        "final_diagnosis": [item["labels"]["final_diagnosis"] for item in batch],
        "diagnoses": [item["labels"]["diagnoses"] for item in batch],
        "raw_row": [item["labels"]["raw_row"] for item in batch],
    }

    # Preserve metadata (same across batch)
    meta = {
        "segment_names": batch[0]["meta"]["segment_names"],
        "box_class_names": batch[0]["meta"]["box_class_names"],
    }

    return {
        "dicom_ids": dicom_ids,           # List of case IDs
        "images": images,                 # Stacked images (B, 1, 224, 224)
        "fixations": {                    # Padded fixation data
            "xy": xy,                     # Padded coordinates (B, max_T, 2)
            "dwell": dwell,               # Padded durations (B, max_T)
            "time": times,                # Padded timestamps (B, max_T)
            "seg_hits": seg_hits,         # Padded segment hits (B, max_T, num_segments)
            "box_hits": box_hits,         # Padded box hits (B, max_T, num_box_classes)
            "lengths": lengths,           # Original sequence lengths (B,)
        },
        "transcripts": transcripts,       # List of transcript dictionaries
        "labels": labels_dict,            # Batched labels and metadata
        "meta": meta,                     # Metadata (same across batch)
    }


def create_dataloader(
    dataset: EGDCXRDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Factory function to create a DataLoader with the custom collate function.
    
    This is a convenience function that creates a PyTorch DataLoader configured
    specifically for the EGDCXRDataset. It automatically uses the custom collate_fn
    to handle variable-length sequences properly.
    
    Args:
        dataset: The EGDCXRDataset instance to create a loader for
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle the data (default: False)
        num_workers: Number of worker processes for data loading (default: 0)
        
    Returns:
        PyTorch DataLoader configured for the multimodal dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # Use custom collate function for variable-length sequences
        drop_last=False,        # Keep all samples, even incomplete batches
    )
