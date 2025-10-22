#!/usr/bin/env python3
"""
PyTorch dataset for loading EGD-CXR gaze frames and clinical labels.

This module bundles the fixation processing logic (cropped DICOM frames),
label extraction from `master_sheet.csv`, and a YAML-backed configuration
loader so downstream training or experimentation code can treat the dataset
as a standard `torch.utils.data.Dataset`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from torch.utils.data import Dataset
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore[assignment]
    HAS_PYDICOM = False

TIME_COLUMN = "Time (in secs)"
X_COLUMN = "FPOGX"
Y_COLUMN = "FPOGY"
DURATION_COLUMN = "FPOGD"


class Logger:
    """Lightweight logger that prints formatted status messages."""

    @staticmethod
    def error(message: str) -> None:
        print(f"⚠ {message}")

    @staticmethod
    def success(message: str) -> None:
        print(f"✓ {message}")

    @staticmethod
    def info(message: str) -> None:
        print(message)


class ConfigLoader:
    """Utility wrapper around YAML configuration files."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path).expanduser()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        Logger.info(f"\nLoading configuration from: {self.config_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        Logger.info(json.dumps(config, indent=4))
        return config

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


@dataclass
class CropConfig:
    radius_scale: float
    min_radius: float
    max_radius: Optional[float]
    fixed_radius: Optional[float]
    fade_decay_multiplier: float
    min_fade_ratio: float
    edge_width: float
    highlight_color: Tuple[int, int, int]
    invert_image: bool


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
    """Schema describing how to build label vectors."""

    class_columns: List[str]


class LabelProcessor:
    """Process labels from master_sheet.csv."""

    def __init__(self, master_sheet_csv: Path, schema: Optional[LabelSchema] = None):
        self.master_sheet_csv = Path(master_sheet_csv).expanduser()
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        self.df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.df.empty:
            raise ValueError(f"No rows found in {self.master_sheet_csv}")
        self.schema = schema or LabelSchema(
            class_columns=self.__discover_binary_columns(self.df)
        )

    def get_labels(self, case_id: str) -> CaseLabels:
        row = self.__get_row(case_id)
        diagnoses = self.__extract_diagnoses(row)
        final_diagnosis = diagnoses[0] if diagnoses else None
        binary_labels = self.__labels_dict(row)
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
        dx_columns = [
            col
            for col in case_row.index
            if col.startswith("dx") and "_icd" not in col
        ]
        diagnoses: List[str] = []
        for col in sorted(
            dx_columns, key=lambda name: int(name[2:]) if name[2:].isdigit() else 0
        ):
            value = case_row[col]
            if isinstance(value, str) and value.strip():
                diagnoses.append(value.strip())
        return diagnoses

    def __labels_dict(self, case_row: pd.Series) -> Dict[str, int]:
        cols = self.schema.class_columns
        labels: Dict[str, int] = {}
        for col in cols:
            if col not in case_row.index:
                continue
            value = case_row[col]
            if pd.isna(value):
                continue
            try:
                labels[col] = int(value)
            except (TypeError, ValueError):
                labels[col] = 1 if str(value).strip() not in ("", "0", "nan", "None") else 0
        return labels


class FixationProcessor:
    """Process gaze fixations to generate circularly cropped frames over a DICOM image."""

    def __init__(self, config: CropConfig, dicom_root: Path, fixations_csv: Path):
        self.config = config
        self.dicom_root = Path(dicom_root).expanduser()
        self.fixations_csv = Path(fixations_csv).expanduser()

        if not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM root directory not found: {self.dicom_root}")
        if not self.dicom_root.is_dir():
            raise NotADirectoryError(f"DICOM root is not a directory: {self.dicom_root}")
        if not self.fixations_csv.exists():
            raise FileNotFoundError(f"Fixations CSV not found: {self.fixations_csv}")

    def get_frames(
        self,
        case_id: str,
        *,
        dicom_path: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        dicom_path_resolved = self.__resolve_dicom_path(case_id, dicom_path)
        fixations = self.__load_fixations(case_id, limit)
        dicom_image = self.__load_dicom_image(dicom_path_resolved)
        frames, metadata = self.__circular_crop_frames(dicom_image, fixations)
        return frames, metadata

    def __resolve_dicom_path(self, case_id: str, override: Optional[Path]) -> Path:
        if override is not None:
            candidate = Path(override)
            if not candidate.is_absolute():
                candidate = self.dicom_root / candidate
        else:
            candidate = self.dicom_root / f"{case_id}.dcm"

        candidate = candidate.expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Could not locate DICOM file at {candidate}")

        return candidate.resolve()

    def __load_fixations(self, case_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(self.fixations_csv, engine="python")
        df = df[df["DICOM_ID"] == case_id].copy()

        if df.empty:
            raise ValueError(
                f"No fixations found for DICOM_ID {case_id} in {self.fixations_csv}"
            )

        valid_mask = (
            df[X_COLUMN].between(0.0, 1.0)
            & df[Y_COLUMN].between(0.0, 1.0)
            & df[DURATION_COLUMN].notna()
            & (df[DURATION_COLUMN] > 0)
        )
        df = df[valid_mask].copy()

        if df.empty:
            raise ValueError(
                f"Fixations for {case_id} are present but none have valid coordinates/duration."
            )

        df.sort_values(by=[TIME_COLUMN, "CNT"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        if limit is not None:
            df = df.iloc[:limit].copy()

        return df

    def __load_dicom_image(self, dicom_path: Path) -> np.ndarray:
        if HAS_PYDICOM:
            ds = pydicom.dcmread(str(dicom_path))
            arr = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            def _get_window_value(
                dicom_obj: "pydicom.dataset.FileDataset",
                attr: str,
                default: Optional[float],
            ) -> Optional[float]:
                val = getattr(dicom_obj, attr, default)
                try:
                    if isinstance(val, (list, tuple)):
                        val = val[0]
                    return float(val)
                except Exception:
                    return float(default) if default is not None else None

            center = _get_window_value(ds, "WindowCenter", None)
            width = _get_window_value(ds, "WindowWidth", None)

            if center is not None and width is not None and width > 0:
                min_val = center - width / 2.0
                max_val = center + width / 2.0
                arr = np.clip(arr, min_val, max_val)
                arr = (arr - min_val) / max((max_val - min_val), 1e-8)
            else:
                arr -= arr.min()
                max_val = arr.max()
                if max_val > 0:
                    arr /= max_val
        else:
            Logger.info("pydicom not available; using imageio fallback for DICOM loading.")
            image = imageio.imread(dicom_path)
            if image.ndim != 2:
                raise ValueError(
                    f"Expected grayscale 2D DICOM, got shape {image.shape} from {dicom_path}"
                )
            arr = image.astype(np.float32)
            arr -= arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr /= max_val

        if self.config.invert_image:
            arr = 1.0 - arr
        return arr

    def __compute_radius_px(
        self, duration_seconds: float, image_shape: Tuple[int, int]
    ) -> float:
        height, width = image_shape
        if self.config.fixed_radius is not None:
            radius = self.config.fixed_radius
        else:
            radius = duration_seconds * self.config.radius_scale
            radius = max(radius, self.config.min_radius)

        diagonal = math.hypot(height, width)
        max_radius_allowed = (
            self.config.max_radius if self.config.max_radius is not None else diagonal
        )
        radius = min(radius, max_radius_allowed)
        return radius

    def __circular_crop_frames(
        self, image: np.ndarray, fixations: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        height, width = image.shape
        yy, xx = np.ogrid[:height, :width]
        grayscale = (image * 255.0).clip(0, 255).astype(np.uint8)
        frames: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []

        fixation_indices = fixations["CNT"].to_numpy(dtype=int)
        times = fixations[TIME_COLUMN].to_numpy(dtype=float)
        durations = fixations[DURATION_COLUMN].to_numpy(dtype=float)
        x_norms = fixations[X_COLUMN].to_numpy(dtype=float)
        y_norms = fixations[Y_COLUMN].to_numpy(dtype=float)

        for idx, (cnt, time_sec, duration, x_norm, y_norm) in enumerate(
            zip(fixation_indices, times, durations, x_norms, y_norms), start=1
        ):
            cx = x_norm * (width - 1)
            cy = y_norm * (height - 1)
            radius = self.__compute_radius_px(duration, image.shape)

            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            dist = np.sqrt(dist_sq)

            frame = np.stack([grayscale, grayscale, grayscale], axis=2).astype(
                np.float32
            )

            fade = np.ones_like(dist, dtype=np.float32)
            falloff = max(radius * self.config.fade_decay_multiplier, 1.0)
            outside = dist > radius
            if np.any(outside):
                delta = (dist[outside] - radius) / falloff
                gaussian = np.exp(-(delta ** 2))
                fade[outside] = np.maximum(gaussian, self.config.min_fade_ratio)
            fade = np.clip(fade, self.config.min_fade_ratio, 1.0)

            frame *= fade[..., None]
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            if self.config.edge_width > 0:
                half_width = self.config.edge_width / 2.0
                edge_mask = (dist >= radius - half_width) & (dist <= radius + half_width)
                frame[edge_mask] = np.array(self.config.highlight_color, dtype=np.uint8)

            frames.append(frame)
            metadata.append(
                {
                    "frame_index": idx,
                    "fixation_index": int(cnt),
                    "time_seconds": float(time_sec),
                    "duration_seconds": float(duration),
                    "center_x_px": float(cx),
                    "center_y_px": float(cy),
                    "radius_px": float(radius),
                }
            )

        return frames, metadata


DEFAULT_CROP_CONFIG: Dict[str, Any] = {
    "radius_scale": 120.0,
    "min_radius": 32.0,
    "max_radius": None,
    "fixed_radius": None,
    "fade_decay_multiplier": 0.5,
    "min_fade_ratio": 0.2,
    "edge_width": 4.0,
    "highlight_color": (255, 0, 0),
    "invert_image": False,
}


class MimicDataset(Dataset):
    """PyTorch dataset wrapping fixation crops and labels for the EGD-CXR dataset."""

    def __init__(
        self,
        config_path: Path,
        *,
        max_fixations: Optional[int] = None,
        load_frames: bool = True,
    ):
        config_loader = ConfigLoader(Path(config_path))
        self.max_fixations = max_fixations
        self.load_frames = load_frames

        dicom_root = config_loader.get("input_path", "dicom_raw")
        if dicom_root is None:
            raise ValueError("Configuration missing 'input_path.dicom_raw'.")
        self.dicom_root = Path(dicom_root).expanduser()
        if not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.dicom_root}")

        gaze_path = Path(config_loader.get("input_path", "gaze_raw")).expanduser()
        if not gaze_path.exists():
            raise FileNotFoundError(f"Gaze directory not found: {gaze_path}")
        self.master_sheet_csv = gaze_path / "master_sheet.csv"
        self.bounding_boxes_csv = gaze_path / "bounding_boxes.csv"
        self.fixations_csv = gaze_path / "fixations.csv"

        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(
                f"master_sheet.csv not found at {self.master_sheet_csv}"
            )
        if not self.fixations_csv.exists():
            raise FileNotFoundError(
                f"fixations.csv not found at {self.fixations_csv}"
            )

        self.master_sheet_df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.master_sheet_df.empty:
            raise ValueError(f"No rows found in {self.master_sheet_csv}")

        self.crop_config = self._build_crop_config(config_loader)
        self.fixation_processor: Optional[FixationProcessor]
        if self.load_frames:
            self.fixation_processor = FixationProcessor(
                config=self.crop_config,
                dicom_root=self.dicom_root,
                fixations_csv=self.fixations_csv,
            )
        else:
            self.fixation_processor = None
        self.label_processor = LabelProcessor(self.master_sheet_csv)

    def __len__(self) -> int:
        return len(self.master_sheet_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.master_sheet_df):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.master_sheet_df)}"
            )
        record = self.master_sheet_df.iloc[idx]
        dicom_id = record["dicom_id"]
        Logger.info(f"Fetching data for DICOM ID: {dicom_id}")

        dicom_path = self._resolve_dicom_path(dicom_id)

        if self.load_frames and self.fixation_processor is not None:
            frames, metadata = self.fixation_processor.get_frames(
                case_id=dicom_id,
                dicom_path=dicom_path,
                limit=self.max_fixations,
            )
        else:
            frames, metadata = [], []

        input_feature = {
            "dicom_id": dicom_id,
            "frames": frames,
            "metadata": metadata,
        }

        labels = self.label_processor.get_labels(dicom_id)

        return {"input_feature": input_feature, "label": labels}

    def _build_crop_config(self, config_loader: ConfigLoader) -> CropConfig:
        raw_overrides = config_loader.get("crop_config", default={}) or {}
        unknown_keys = set(raw_overrides) - set(DEFAULT_CROP_CONFIG)
        if unknown_keys:
            Logger.info(f"Ignoring unknown crop_config keys: {sorted(unknown_keys)}")
        overrides = {
            key: raw_overrides[key] for key in DEFAULT_CROP_CONFIG if key in raw_overrides
        }
        params = {**DEFAULT_CROP_CONFIG, **overrides}

        highlight_color = params.get(
            "highlight_color", DEFAULT_CROP_CONFIG["highlight_color"]
        )
        if isinstance(highlight_color, (list, tuple)):
            highlight_color = tuple(int(x) for x in highlight_color)
        else:
            raise ValueError("crop_config.highlight_color must be a sequence of three integers")
        if len(highlight_color) != 3:
            raise ValueError(
                "crop_config.highlight_color must contain exactly three values (R, G, B)"
            )
        params["highlight_color"] = highlight_color

        return CropConfig(**params)

    def _resolve_dicom_path(self, dicom_id: str) -> Path:
        return self.dicom_root / f"{dicom_id}.dcm"


def collate_case_labels(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Example collate function that converts the CaseLabels dataclass into dicts."""

    inputs: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    for sample in batch:
        inputs.append(sample["input_feature"])
        label_obj = sample["label"]
        if isinstance(label_obj, CaseLabels):
            label_dict = asdict(label_obj)
            label_dict["source_csv"] = str(label_dict.get("source_csv"))
            labels.append(label_dict)
        else:
            labels.append(label_obj)
    return {"input_feature": inputs, "label": labels}
