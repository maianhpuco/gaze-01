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
DURATION_COLUMN = "FPOGD"           # Fixation duration in seconds   # <<< NOTE: paper uses seconds


class Logger:
    """Minimal stdout logger for dataset operations."""
    @staticmethod
    def info(message: str) -> None:      print(message)
    @staticmethod
    def error(message: str) -> None:     print(f"Warning: {message}")


@dataclass(frozen=True)
class LabelSchema:
    """Schema for binary classification columns in master_sheet.csv."""
    class_columns: List[str]


@dataclass(frozen=True)
class ClassificationLabel:
    """Single-label classification payload (CHF / pneumonia / Normal)."""
    index: int
    name: str
    classes: Tuple[str, ...]
    per_class: Dict[str, int]
    positives: Tuple[str, ...]
    ambiguous: bool

    def one_hot(self) -> torch.Tensor:
        vec = torch.zeros(len(self.classes), dtype=torch.float32)
        if 0 <= self.index < len(self.classes):
            vec[self.index] = 1.0
        return vec


class LabelProcessor:
    """Convert master_sheet.csv labels into ML-ready format."""
    def __init__(self, master_sheet_csv: Path, schema: Optional[LabelSchema] = None):
        self.master_sheet_csv = Path(master_sheet_csv).expanduser()
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        self.df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.df.empty:
            raise ValueError(f"master_sheet.csv has no rows: {self.master_sheet_csv}")
        self.schema = schema or LabelSchema(class_columns=self._discover_binary_columns(self.df))

    @staticmethod
    def _discover_binary_columns(df: pd.DataFrame) -> List[str]:
        cols = list(df.columns)
        if "Normal" in cols and "support_devices__chx" in cols:
            si = cols.index("Normal")
            ei = cols.index("support_devices__chx")
            if si <= ei:
                return cols[si : ei + 1]
        fallback: List[str] = []
        for col in cols:
            series = df[col].dropna()
            if not series.empty and series.isin([0, 1]).all():
                fallback.append(col)
        return fallback

    def _get_row(self, case_id: str) -> pd.Series:
        mask = self.df["dicom_id"] == case_id
        if not mask.any():
            raise ValueError(f"dicom_id {case_id} not found in {self.master_sheet_csv}")
        return self.df.loc[mask].iloc[0]

    def diagnoses(self, case_id: str) -> Tuple[Optional[str], List[str]]:
        row = self._get_row(case_id)
        dx_cols = [c for c in row.index if c.startswith("dx") and not c.endswith("_icd")]
        diagnoses: List[str] = []
        for col in sorted(dx_cols, key=lambda n: int(n[2:]) if n[2:].isdigit() else 0):
            v = row[col]
            if isinstance(v, str) and v.strip():
                diagnoses.append(v.strip())
        return (diagnoses[0] if diagnoses else None), diagnoses

    def vector(self, case_id: str) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        row = self._get_row(case_id)
        values: List[int] = []
        for col in self.schema.class_columns:
            v = row.get(col, np.nan)
            if pd.isna(v):
                values.append(0); continue
            try:
                values.append(int(v))
            except (TypeError, ValueError):
                values.append(1 if str(v).strip() not in ("", "0", "nan", "None") else 0)
        tensor = torch.tensor(values, dtype=torch.float32)
        return tensor, self.schema.class_columns, row.to_dict()


@dataclass(frozen=True)
class BoxRow:
    """Bounding-box annotation."""
    x1: int; y1: int; x2: int; y2: int
    cls_id: int; cls_name: str


class EGDCXRDataset(Dataset):
    """Multimodal dataset (gaze + image + seg + boxes + transcript + labels)."""

    def __init__(
        self,
        root: Path,
        seg_path: Path,
        transcripts_path: Optional[Path] = None,
        *,
        dicom_root: Optional[Path] = None,
        max_fixations: Optional[int] = None,
        case_ids: Optional[Sequence[str]] = None,
        classes: Sequence[str] = ("CHF", "pneumonia", "Normal"),
        drop_unlabelled: bool = True,
    ):
        # ------------------------------------------------------------------ #
        #   Path setup
        # ------------------------------------------------------------------ #
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Gaze dataset root not found: {self.root}")

        self.fixations_csv = self.root / "fixations.csv"
        self.master_sheet_csv = self.root / "master_sheet.csv"
        self.bounding_boxes_csv = self.root / "bounding_boxes.csv"

        self.seg_path = Path(seg_path).expanduser()
        self.transcripts_path = Path(transcripts_path or seg_path).expanduser()
        self.dicom_root = Path(dicom_root).expanduser() if dicom_root else None

        for p in [self.fixations_csv, self.master_sheet_csv,
                  self.bounding_boxes_csv, self.seg_path, self.transcripts_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing path: {p}")
        if self.dicom_root and not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM root not found: {self.dicom_root}")

        # ------------------------------------------------------------------ #
        #   Config
        # ------------------------------------------------------------------ #
        self.max_fix = max_fixations
        self.single_classes = tuple(c.strip() for c in classes if c and str(c).strip())
        if not self.single_classes:
            raise ValueError("`classes` must contain at least one non-empty name.")
        self.drop_unlabelled = drop_unlabelled

        # ------------------------------------------------------------------ #
        #   Load CSVs
        # ------------------------------------------------------------------ #
        self.ms_df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.ms_df.empty:
            raise ValueError("master_sheet.csv empty")
        self.fx_df = pd.read_csv(self.fixations_csv, engine="python")
        self.bb_df = pd.read_csv(self.bounding_boxes_csv, engine="python") \
            if self.bounding_boxes_csv.exists() else pd.DataFrame()

        # ------------------------------------------------------------------ #
        #   Transcript / segmentation mode
        # ------------------------------------------------------------------ #
        self.transcripts_mode = "csv" if self.transcripts_path.is_file() else "directory"
        self.tr_df = pd.read_csv(self.transcripts_path, engine="python") \
            if self.transcripts_mode == "csv" else None

        self.seg_mode = "directory" if self.seg_path.is_dir() else "arrays"
        if self.seg_mode not in {"directory", "arrays"}:
            raise ValueError("Unsupported segmentation storage format.")

        # ------------------------------------------------------------------ #
        #   Region / box meta
        # ------------------------------------------------------------------ #
        self.region_names = self._discover_region_names() if self.seg_mode == "directory" else []
        self.box_label_to_idx = self._build_box_label_mapping()
        self.box_class_names = [n for n, _ in sorted(self.box_label_to_idx.items(),
                                                   key=lambda kv: kv[1])]
        self.num_box_classes = len(self.box_class_names)
        self.num_segments: Optional[int] = len(self.region_names) if self.region_names else None

        # ------------------------------------------------------------------ #
        #   Case intersection
        # ------------------------------------------------------------------ #
        fx_cases = set(self.fx_df["DICOM_ID"].dropna().astype(str))
        ms_cases = set(self.ms_df["dicom_id"].dropna().astype(str))
        transcript_cases = self._discover_transcript_case_ids()
        seg_cases = self._discover_segmentation_case_ids()
        base_cases = sorted(ms_cases & fx_cases & transcript_cases & seg_cases)

        if case_ids is not None:
            requested = set(case_ids)
            missing = requested - set(base_cases)
            if missing:
                Logger.error(f"{len(missing)} requested IDs missing modalities; skipping.")
            base_cases = [cid for cid in base_cases if cid in requested]

        if not base_cases:
            raise ValueError("No cases with all required modalities.")

        # ------------------------------------------------------------------ #
        #   Label processing & final case list
        # ------------------------------------------------------------------ #
        self.label_proc = LabelProcessor(self.master_sheet_csv)

        classification_map: Dict[str, ClassificationLabel] = {}
        filtered_cases: List[str] = []
        targets: List[int] = []

        for case_id in base_cases:
            row = self.ms_df[self.ms_df["dicom_id"] == case_id]
            if row.empty:
                continue
            cls = self._classify_row(row.iloc[0])
            if cls.index < 0 and self.drop_unlabelled:
                continue
            filtered_cases.append(case_id)
            classification_map[case_id] = cls
            targets.append(cls.index)

        if not filtered_cases:
            raise ValueError("No cases with valid single-label classification.")

        self.case_ids = filtered_cases
        self._classification_map = classification_map
        self._single_targets = targets
        self._class_counts = torch.bincount(
            torch.tensor(targets, dtype=torch.long),
            minlength=len(self.single_classes),
        )

        Logger.info(
            f"EGDCXRDataset ready: {len(self.case_ids)} cases | "
            f"regions={len(self.region_names) if self.region_names else 'n/a'} | "
            f"bbox_classes={len(self.box_label_to_idx)}"
        )

    # ------------------------------------------------------------------ #
    #   Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:                     return len(self.case_ids)
    @property
    def class_names(self) -> Tuple[str, ...]:     return self.single_classes
    @property
    def class_counts(self) -> torch.Tensor:       return self._class_counts.clone()

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts.to(torch.float32)
        total = counts.sum().item()
        w = torch.zeros_like(counts)
        for i, c in enumerate(counts):
            w[i] = total / max(float(c.item()), 1.0)
        return w / w.mean().clamp_min(1e-6)

    def sample_weights(self) -> torch.Tensor:
        w = self.class_weights()
        return w[torch.tensor(self._single_targets, dtype=torch.long)]

    # ------------------------------------------------------------------ #
    #   Helper methods
    # ------------------------------------------------------------------ #
    def _classify_row(self, row: pd.Series) -> ClassificationLabel:
        per_class: Dict[str, int] = {}
        positives: List[str] = []
        for cls in self.single_classes:
            raw = row.get(cls, np.nan)
            if pd.isna(raw):
                val = 0
            else:
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    val = 1 if str(raw).strip() not in ("", "0", "nan", "None") else 0
            val = 1 if val == 1 else 0
            per_class[cls] = val
            if val == 1:
                positives.append(cls)

        idx = -1; name = "Unknown"
        for cls in self.single_classes:
            if per_class.get(cls, 0) == 1:
                idx = self.single_classes.index(cls)
                name = cls
                break
        return ClassificationLabel(
            index=idx, name=name, classes=self.single_classes,
            per_class=per_class, positives=tuple(positives),
            ambiguous=len(positives) > 1,
        )

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
        for p in self.transcripts_path.iterdir():
            if p.is_dir() and (p / "transcript.json").exists():
                cases.add(p.name)
        return cases

    def _discover_segmentation_case_ids(self) -> set[str]:
        cases: set[str] = set()
        if self.seg_mode == "directory":
            for p in self.seg_path.iterdir():
                if p.is_dir() and any(p.glob("*.png")):
                    cases.add(p.name)
        else:
            for f in self.seg_path.glob("*_segs.npz"):
                cases.add(f.stem.replace("_segs", ""))
            for f in self.seg_path.glob("*_segs.npy"):
                cases.add(f.stem.replace("_segs", ""))
        return cases

    def _build_box_label_mapping(self) -> Dict[str, int]:
        if self.bb_df.empty:
            return {}
        names = sorted(self.bb_df["bbox_name"].dropna().astype(str).unique())
        return {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------------ #
    #   Data loading helpers
    # ------------------------------------------------------------------ #
    def _load_seg_masks(self, dicom_id: str) -> np.ndarray:
        """Return (R+1, H, W) uint8 masks (last channel = background)."""
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

        region_names = self.region_names or sorted([p.stem for p in case_dir.glob("*.png")])
        if not region_names:
            raise ValueError(f"No segmentation PNGs for {dicom_id}")

        ref = None
        for n in region_names:
            p = case_dir / f"{n}.png"
            if p.exists():
                ref = imageio.imread(p)
                break
        if ref is None:
            raise ValueError(f"No segmentation PNGs for {dicom_id}")
        h, w = ref.shape[:2]
        masks = np.zeros((len(region_names), h, w), dtype=np.uint8)

        for i, n in enumerate(region_names):
            p = case_dir / f"{n}.png"
            if not p.exists():
                continue
            img = imageio.imread(p)
            mask = img.max(axis=2) > 0 if img.ndim == 3 else img > 0
            masks[i] = mask.astype(np.uint8)

        bg = (masks.sum(axis=0, keepdims=True) == 0).astype(np.uint8)
        return np.concatenate([masks, bg], axis=0)

    def _load_dicom_image(self, dicom_id: str) -> Optional[np.ndarray]:
        """Return float32 image (H, W) in [0,1] with proper windowing."""
        if self.dicom_root is None:
            return None
        path = self.dicom_root / f"{dicom_id}.dcm"
        if not path.exists():
            return None
        try:
            if HAS_PYDICOM:
                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                def _win(tag: str) -> Optional[float]:
                    v = getattr(ds, tag, None)
                    if v is None: return None
                    try:
                        return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
                    except Exception:
                        return None

                center = _win("WindowCenter")
                width  = _win("WindowWidth")
                if center is not None and width is not None and width > 0:
                    lo = center - width / 2.0
                    hi = center + width / 2.0
                    arr = np.clip(arr, lo, hi)
                    arr = (arr - lo) / max((hi - lo), 1e-6)                 # <<< FIX: epsilon
                else:
                    arr -= arr.min()
                    mx = arr.max()
                    if mx > 0:
                        arr = arr / mx
            else:
                arr = imageio.imread(path)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                arr = arr.astype(np.float32)
                arr -= arr.min()
                mx = arr.max()
                if mx > 0:
                    arr = arr / mx
        except Exception:
            return None

        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)

    def _load_boxes(self, dicom_id: str) -> List[BoxRow]:
        if self.bb_df.empty:
            return []
        rows = self.bb_df[self.bb_df["dicom_id"] == dicom_id]
        out: List[BoxRow] = []
        for _, r in rows.iterrows():
            name = str(r["bbox_name"])
            cid = self.box_label_to_idx.get(name, -1)
            out.append(BoxRow(
                x1=int(round(float(r["x1"]))),
                y1=int(round(float(r["y1"]))),
                x2=int(round(float(r["x2"]))),
                y2=int(round(float(r["y2"]))),
                cls_id=cid,
                cls_name=name,
            ))
        return out

    def _load_transcript(self, dicom_id: str) -> Dict[str, Any]:
        if self.transcripts_mode == "csv":
            rows = self.tr_df[self.tr_df["dicom_id"] == dicom_id]
            if rows.empty:
                return {"text": "", "segments": []}
            txt = str(rows.iloc[0].get("transcript", "") or "")
            return {"text": txt, "segments": []}
        case_dir = self.transcripts_path / dicom_id
        tf = case_dir / "transcript.json"
        if tf.exists():
            data = json.loads(tf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                txt = str(data.get("transcript") or data.get("full_text") or "").strip()
                segs = data.get("segments") or []
                return {"text": txt, "segments": segs}
        return {"text": "", "segments": []}

    def _load_fixations(self, dicom_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (times_sec, xy_norm, dwell_sec).
        dwell_sec is **seconds** (paper uses Δt_t in seconds).
        """
        df = self.fx_df[self.fx_df["DICOM_ID"] == dicom_id].copy()
        if df.empty:
            raise ValueError(f"No fixations for {dicom_id}")

        # Keep only valid fixations
        df = df[
            df[X_COLUMN].between(0.0, 1.0) &
            df[Y_COLUMN].between(0.0, 1.0) &
            df[DURATION_COLUMN].notna() &
            (df[DURATION_COLUMN] > 0)
        ].copy()
        if df.empty:
            raise ValueError(f"Invalid fixations for {dicom_id}")

        # Stable sort by time; include CNT only if present
        if "CNT" in df.columns:
            df.sort_values(by=[TIME_COLUMN, "CNT"], inplace=True, kind="mergesort")
        else:
            df.sort_values(by=[TIME_COLUMN], inplace=True, kind="mergesort")
        if self.max_fix is not None:
            df = df.iloc[: self.max_fix]

        xy_norm = df[[X_COLUMN, Y_COLUMN]].to_numpy(dtype=np.float32)
        # <<< FIX: keep dwell in **seconds** (no *1000)
        dwell = df[DURATION_COLUMN].to_numpy(dtype=np.float32)          # seconds
        times = df[TIME_COLUMN].to_numpy(dtype=np.float32)

        # Basic sanity: drop any rows that became NaN (should be none after filters)
        if not np.isfinite(xy_norm).all() or not np.isfinite(dwell).all() or not np.isfinite(times).all():
            mask = np.isfinite(xy_norm).all(axis=1) & np.isfinite(dwell) & np.isfinite(times)
            xy_norm = xy_norm[mask]
            dwell = dwell[mask]
            times = times[mask]
        return times, xy_norm, dwell

    # ------------------------------------------------------------------ #
    #   __getitem__
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dicom_id = self.case_ids[idx]

        # ---- segmentation -------------------------------------------------
        seg_masks_np = self._load_seg_masks(dicom_id)
        if self.num_segments is None:
            self.num_segments = seg_masks_np.shape[0] - 1 if seg_masks_np.shape[0] > 1 else seg_masks_np.shape[0]
            if not self.region_names:
                self.region_names = [f"segment_{i}" for i in range(self.num_segments)]
        num_segments = self.num_segments or 0
        segments_np = seg_masks_np[:num_segments] if num_segments > 0 else np.zeros((0, *seg_masks_np.shape[1:]), dtype=np.uint8)

        # ---- other modalities --------------------------------------------
        boxes = self._load_boxes(dicom_id)
        transcript_payload = self._load_transcript(dicom_id)
        final_dx, diagnoses = self.label_proc.diagnoses(dicom_id)
        times_sec, xy_norm, dwell = self._load_fixations(dicom_id)

        # ---- gaze → pixel ------------------------------------------------
        height, width = seg_masks_np.shape[1:]
        if xy_norm.size == 0:
            xy_px = np.zeros((0, 2), dtype=np.float32)
        else:
            xy_px = np.stack([
                xy_norm[:, 0] * (width - 1),
                xy_norm[:, 1] * (height - 1),
            ], axis=1).astype(np.float32)

        # ---- gaze → ROI hits ---------------------------------------------
        T = xy_px.shape[0]
        seg_hits = np.zeros((T, num_segments), dtype=np.float32)
        if num_segments > 0 and T > 0:
            xs = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height - 1)
            hits = segments_np[:, ys, xs] > 0
            seg_hits = hits.T.astype(np.float32)

        num_box_classes = max(0, self.num_box_classes)
        box_hits = np.zeros((T, num_box_classes), dtype=np.float32)
        if num_box_classes > 0 and T > 0:
            xs_int = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width - 1)
            ys_int = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height - 1)
            for t in range(T):
                x, y = int(xs_int[t]), int(ys_int[t])
                for box in boxes:
                    cid = box.cls_id
                    if 0 <= cid < num_box_classes and box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
                        box_hits[t, cid] = 1.0

        # ---- image -------------------------------------------------------
        img_arr = self._load_dicom_image(dicom_id)
        if img_arr is None:
            image_tensor = torch.zeros(1, 224, 224, dtype=torch.float32)
        else:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()
            image_tensor = F.interpolate(
                img_tensor.unsqueeze(0), size=(224, 224),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # ---- presence vectors --------------------------------------------
        segment_names = self.region_names or [f"segment_{i}" for i in range(num_segments)]
        box_class_names = self.box_class_names or [f"class_{i}" for i in range(self.num_box_classes)]

        if num_segments > 0:
            seg_presence_np = (segments_np.reshape(num_segments, -1).sum(axis=1) > 0).astype(np.float32)
            segment_presence = torch.from_numpy(seg_presence_np)
        else:
            segment_presence = torch.zeros(0, dtype=torch.float32)

        box_presence = torch.zeros(self.num_box_classes, dtype=torch.float32)
        for b in boxes:
            if 0 <= b.cls_id < self.num_box_classes:
                box_presence[b.cls_id] = 1.0

        # ---- assemble sample ---------------------------------------------
        classification = self._classification_map[dicom_id]
        per_class = classification.per_class

        sample = {
            "dicom_id": dicom_id,
            "image": image_tensor,
            "fixations": {
                "xy": torch.from_numpy(xy_px.astype(np.float32)),
                "xy_norm": torch.from_numpy(xy_norm.astype(np.float32)),
                "time": torch.from_numpy(times_sec.astype(np.float32)),
                "dwell": torch.from_numpy(dwell.astype(np.float32)),   # seconds
                "seg_hits": torch.from_numpy(seg_hits).float(),
                "box_hits": torch.from_numpy(box_hits).float(),
            },
            "segment_presence": segment_presence.float(),
            "box_presence": box_presence.float(),
            "transcript": transcript_payload,
            "labels": {
                "classification": {
                    "index": classification.index,
                    "name": classification.name,
                    "classes": classification.classes,
                    "one_hot": classification.one_hot(),
                    "ambiguous": classification.ambiguous,
                    "per_class": per_class,
                    "positives": classification.positives,
                },
                "single_index": torch.tensor(classification.index, dtype=torch.long),
                "single_name": classification.name,
                "single_class_names": classification.classes,
                "final_diagnosis": final_dx,
                "diagnoses": diagnoses,
            },
            "boxes": boxes,
            "meta": {
                "segment_names": segment_names,
                "box_class_names": box_class_names,
                "segmentation_height": int(height),
                "segmentation_width": int(width),
                "image_height": int(image_tensor.shape[-2]),
                "image_width": int(image_tensor.shape[-1]),
            },
        }
        return sample


# -------------------------------------------------------------------------- #
#   Collate function (unchanged – only minor comment tweaks)
# -------------------------------------------------------------------------- #
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length gaze sequences and stack fixed tensors."""
    dicom_ids = [item["dicom_id"] for item in batch]
    lengths = torch.tensor([item["fixations"]["xy"].shape[0] for item in batch], dtype=torch.long)

    xy      = pad_sequence([item["fixations"]["xy"]       for item in batch], batch_first=True)
    xy_norm = pad_sequence([item["fixations"]["xy_norm"]  for item in batch], batch_first=True)
    dwell   = pad_sequence([item["fixations"]["dwell"]    for item in batch], batch_first=True)
    times   = pad_sequence([item["fixations"]["time"]     for item in batch], batch_first=True)
    seg_h   = pad_sequence([item["fixations"]["seg_hits"] for item in batch], batch_first=True)
    box_h   = pad_sequence([item["fixations"]["box_hits"] for item in batch], batch_first=True)

    images = torch.stack([item["image"] for item in batch], dim=0)
    seg_pres = torch.stack([item["segment_presence"] for item in batch], dim=0)
    box_pres = torch.stack([item["box_presence"] for item in batch], dim=0)
    transcripts = [item["transcript"] for item in batch]

    # labels
    one_hot = torch.stack([item["labels"]["classification"]["one_hot"] for item in batch], dim=0)
    idxs    = torch.tensor([item["labels"]["classification"]["index"] for item in batch], dtype=torch.long)
    names   = [item["labels"]["classification"]["name"] for item in batch]
    ambig   = [item["labels"]["classification"]["ambiguous"] for item in batch]
    percls  = [item["labels"]["classification"]["per_class"] for item in batch]
    pos     = [item["labels"]["classification"]["positives"] for item in batch]
    class_names = list(batch[0]["labels"]["classification"]["classes"])

    single_idx = torch.stack([item["labels"]["single_index"] for item in batch], dim=0)
    single_nam = [item["labels"]["single_name"] for item in batch]

    labels_dict = {
        "final_diagnosis": [item["labels"]["final_diagnosis"] for item in batch],
        "diagnoses": [item["labels"]["diagnoses"] for item in batch],
        "classification": {
            "one_hot": one_hot, "indices": idxs, "names": names,
            "classes": class_names, "ambiguous": ambig,
            "per_class": percls, "positives": pos,
            "index": idxs, "name": names,
        },
        "single_index": single_idx,
        "single_name": single_nam,
        "single_class_names": class_names,
    }

    meta = {
        "segment_names": batch[0]["meta"]["segment_names"],
        "box_class_names": batch[0]["meta"]["box_class_names"],
        "segmentation_height": batch[0]["meta"].get("segmentation_height"),
        "segmentation_width": batch[0]["meta"].get("segmentation_width"),
        "image_height": batch[0]["meta"].get("image_height"),
        "image_width": batch[0]["meta"].get("image_width"),
    }

    return {
        "dicom_ids": dicom_ids,
        "images": images,
        "segment_presence": seg_pres,
        "box_presence": box_pres,
        "fixations": {
            "xy": xy,
            "xy_norm": xy_norm,
            "dwell": dwell,
            "time": times,
            "seg_hits": seg_h,
            "box_hits": box_h,
            "lengths": lengths,
        },
        "transcripts": transcripts,
        "labels": labels_dict,
        "meta": meta,
        "boxes": [item["boxes"] for item in batch],
    }


def create_dataloader(
    dataset: EGDCXRDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
) -> DataLoader:
    """Convenient DataLoader factory using the custom collate."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
