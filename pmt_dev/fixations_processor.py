#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import pydicom
import numpy as np
import pandas as pd

TIME_COLUMN = "Time (in secs)"
X_COLUMN = "FPOGX"
Y_COLUMN = "FPOGY"
DURATION_COLUMN = "FPOGD"


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

class FixationProcessor:
    """Process gaze fixations to generate circularly cropped frames over a DICOM image.

    Instance methods use self.config consistently; utilities are grouped for clarity.
    """

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
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """Load fixations and DICOM image, then generate cropped frames and metadata."""
        dicom_path_resolved = self.__resolve_dicom_path(case_id, dicom_path)
        fixations = self.__load_fixations(case_id, limit)
        dicom_image = self.__load_dicom_image(dicom_path_resolved)
        frames, metadata = self.__circular_crop_frames(dicom_image, fixations)
        return frames, metadata

    def __resolve_dicom_path(self, case_id: str, override: Optional[Path]) -> Path:
        """Resolve and validate the DICOM path for a given case id."""
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
        """Load and validate fixation rows for a case id.

        - Filters by DICOM_ID
        - Keeps rows with normalized coordinates in [0,1] and positive duration
        - Sorts by time then CNT
        - Applies optional limit
        """
        df = pd.read_csv(self.fixations_csv)
        df = df[df["DICOM_ID"] == case_id].copy()

        if df.empty:
            raise ValueError(f"No fixations found for DICOM_ID {case_id} in {self.fixations_csv}")

        valid_mask = (
            df[X_COLUMN].between(0.0, 1.0)
            & df[Y_COLUMN].between(0.0, 1.0)
            & df[DURATION_COLUMN].notna()
            & (df[DURATION_COLUMN] > 0)
        )
        df = df[valid_mask].copy()

        if df.empty:
            raise ValueError(f"Fixations for {case_id} are present but none have valid coordinates/duration.")

        df.sort_values(by=[TIME_COLUMN, "CNT"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        if limit is not None:
            df = df.iloc[:limit].copy()

        return df

    def __load_dicom_image(self, dicom_path: Path) -> np.ndarray:
        """Read and normalize a DICOM (grayscale) image to [0,1] with pydicom; optionally invert.

        Using pydicom avoids shape issues observed with imageio for certain DICOMs.
        """
        ds = pydicom.dcmread(str(dicom_path))
        arr = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present (common in DICOM)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept

        # Windowing if tags present, else min-max normalize
        def _get_window_value(dicom, attr, default):
            val = getattr(dicom, attr, default)
            try:
                if isinstance(val, (list, tuple)):
                    val = val[0]
                return float(val)
            except Exception:
                return float(default)

        center = _get_window_value(ds, 'WindowCenter', None)
        width = _get_window_value(ds, 'WindowWidth', None)

        if center is not None and width is not None and width > 0:
            min_val = center - width / 2.0
            max_val = center + width / 2.0
            arr = np.clip(arr, min_val, max_val)
            arr = (arr - min_val) / max((max_val - min_val), 1e-8)
        else:
            # Fallback: scale to [0,1]
            arr -= arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr /= max_val

        if self.config.invert_image:
            arr = 1.0 - arr
        return arr

    def __compute_radius_px(self, duration_seconds: float, image_shape: Tuple[int, int]) -> float:
        """Compute radius in pixels based on fixation duration and config constraints."""
        height, width = image_shape
        if self.config.fixed_radius is not None:
            radius = self.config.fixed_radius
        else:
            radius = duration_seconds * self.config.radius_scale
            radius = max(radius, self.config.min_radius)

        diagonal = math.hypot(height, width)
        max_radius_allowed = self.config.max_radius if self.config.max_radius is not None else diagonal
        radius = min(radius, max_radius_allowed)
        return radius

    def __circular_crop_frames(self, image: np.ndarray, fixations: pd.DataFrame) -> Tuple[List[np.ndarray], List[dict]]:
        """Generate frames with circular emphasis around each fixation and return metadata."""
        height, width = image.shape
        yy, xx = np.ogrid[:height, :width]  # efficient coordinate grids
        grayscale = (image * 255.0).clip(0, 255).astype(np.uint8)
        frames: List[np.ndarray] = []
        metadata: List[dict] = []

        fixation_indices = fixations["CNT"].to_numpy(dtype=int)
        times = fixations[TIME_COLUMN].to_numpy(dtype=float)
        durations = fixations[DURATION_COLUMN].to_numpy(dtype=float)
        x_norms = fixations[X_COLUMN].to_numpy(dtype=float)
        y_norms = fixations[Y_COLUMN].to_numpy(dtype=float)

        for idx, (cnt, time_sec, duration, x_norm, y_norm) in enumerate(
            zip(fixation_indices, times, durations, x_norms, y_norms), start=1
        ):
            # Convert normalised gaze coordinates to pixel indices.
            cx = x_norm * (width - 1)
            cy = y_norm * (height - 1)
            radius = self.__compute_radius_px(duration, image.shape)

            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            dist = np.sqrt(dist_sq)

            frame = np.stack([grayscale, grayscale, grayscale], axis=2).astype(np.float32)

            fade = np.ones_like(dist, dtype=np.float32)
            falloff = max(radius * self.config.fade_decay_multiplier, 1.0)
            outside = dist > radius
            if np.any(outside):
                # Gaussian falloff outside the circle to produce a gradual fade.
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

    