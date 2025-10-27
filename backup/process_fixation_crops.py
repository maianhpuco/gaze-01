#!/usr/bin/env python3
"""
Create circular crops around individual fixation points on EGD-CXR DICOM images.

For each fixation, the script keeps only the pixels inside a circular region
centred on the fixation and gradually fades the rest of the image toward a dim
background (70% transparency by default, adjustable via --min-fade). Each fixation is
exported as its own frame so downstream tooling can treat them as video frames.

Example
-------
python process_fixation_crops.py \\
    --case-id 24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4 \\
    --output-root plots/crops
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0")
DEFAULT_FIXATIONS_CSV = DEFAULT_DATA_ROOT / "fixations.csv"
DEFAULT_DICOM_ROOT = ROOT / "sample"
DEFAULT_OUTPUT_ROOT = ROOT / "plots" / "crops"
DEFAULT_CASE_ID = "24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4"
DEFAULT_FIXED_RADIUS = 200.0
DEFAULT_MIN_FADE_RATIO = 0.3  # Adjust to lighten/darken the outer background

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate circular crops for each fixation on a DICOM image."
    )
    parser.add_argument(
        "--case-id",
        default=DEFAULT_CASE_ID,
        help=f"DICOM case identifier (default: {DEFAULT_CASE_ID}).",
    )
    parser.add_argument(
        "--fixations-csv",
        type=Path,
        default=DEFAULT_FIXATIONS_CSV,
        help=f"Path to fixations CSV (default: {DEFAULT_FIXATIONS_CSV}).",
    )
    parser.add_argument(
        "--dicom-path",
        type=Path,
        help="Explicit path to the DICOM file. Supersedes --dicom-root.",
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=DEFAULT_DICOM_ROOT,
        help="Directory that contains <case-id>.dcm files (default: sample/).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store the generated crops (default: plots/crops).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N fixations after sorting by timestamp.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=900.0,
        help="Multiplier applied to fixation duration (seconds) to derive radius in pixels.",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=80.0,
        help="Minimum radius in pixels applied to every fixation.",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        help="Optional maximum radius in pixels. Defaults to the image diagonal.",
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=6.0,
        help="Thickness of the circular outline in pixels.",
    )
    parser.add_argument(
        "--fixed-radius",
        type=float,
        default=DEFAULT_FIXED_RADIUS,
        help=f"Apply the same radius (pixels) to every fixation, bypassing duration scaling (default: {DEFAULT_FIXED_RADIUS}).",
    )
    parser.add_argument(
        "--fade-decay",
        type=float,
        default=2.0,
        help="Controls how gradually the crop fades to black outside the circle (higher = slower fade).",
    )
    parser.add_argument(
        "--min-fade",
        type=float,
        default=DEFAULT_MIN_FADE_RATIO,
        help=(
            "Clamp the minimum brightness outside the fixation region (0…1). "
            f"Default keeps at least {DEFAULT_MIN_FADE_RATIO:.0%} brightness."
        ),
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Disable inversion (by default X-rays are inverted for better contrast).",
    )
    return parser.parse_args()


def resolve_dicom_path(case_id: str, dicom_path: Optional[Path], dicom_root: Path) -> Path:
    if dicom_path is not None:
        resolved = dicom_path
    else:
        resolved = dicom_root / f"{case_id}.dcm"
    if not resolved.exists():
        raise FileNotFoundError(f"Could not locate DICOM file at {resolved}")
    return resolved


def load_fixations(fixations_csv: Path, case_id: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(fixations_csv)
    df = df[df["DICOM_ID"] == case_id].copy()

    if df.empty:
        raise ValueError(f"No fixations found for DICOM_ID {case_id} in {fixations_csv}")

    # Keep only rows with valid, normalised gaze coordinates and duration.
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


def load_dicom_image(dicom_path: Path, invert: bool) -> np.ndarray:
    image = imageio.imread(dicom_path)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale 2D DICOM, got shape {image.shape} from {dicom_path}")

    image = image.astype(np.float32)
    image -= image.min()
    max_val = image.max()
    if max_val > 0:
        image /= max_val
    if invert:
        image = 1.0 - image
    return image


def compute_radius_px(
    duration_seconds: float,
    config: CropConfig,
    image_shape: Tuple[int, int],
) -> float:
    height, width = image_shape
    if config.fixed_radius is not None:
        radius = config.fixed_radius
    else:
        radius = duration_seconds * config.radius_scale
        radius = max(radius, config.min_radius)

    diagonal = math.hypot(height, width)
    max_radius_allowed = config.max_radius if config.max_radius is not None else diagonal
    radius = min(radius, max_radius_allowed)
    return radius


def circular_crop_frames(
    image: np.ndarray,
    fixations: pd.DataFrame,
    config: CropConfig,
) -> Tuple[List[np.ndarray], List[dict]]:
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
        radius = compute_radius_px(duration, config, image.shape)

        dist_sq = (xx - cx)**2 + (yy - cy)**2
        dist = np.sqrt(dist_sq)

        frame = np.stack([grayscale, grayscale, grayscale], axis=2).astype(np.float32)

        fade = np.ones_like(dist, dtype=np.float32)
        falloff = max(radius * config.fade_decay_multiplier, 1.0)
        outside = dist > radius
        if np.any(outside):
            # Gaussian falloff outside the circle to produce a gradual fade.
            delta = (dist[outside] - radius) / falloff
            gaussian = np.exp(-(delta**2))
            fade[outside] = np.maximum(gaussian, config.min_fade_ratio)
        fade = np.clip(fade, config.min_fade_ratio, 1.0)

        frame *= fade[..., None]
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        if config.edge_width > 0:
            half_width = config.edge_width / 2.0
            edge_mask = (dist >= radius - half_width) & (dist <= radius + half_width)
            frame[edge_mask] = np.array(config.highlight_color, dtype=np.uint8)

        frames.append(frame)
        metadata.append(
            {
                "frame_index": idx,
                "fixation_index": int(cnt),
                "time_seconds": float(time_sec),
                "duration_seconds": float(duration),
                "center_x_px": cx,
                "center_y_px": cy,
                "radius_px": radius,
            }
        )

    return frames, metadata


def save_frames(
    frames: Iterable[np.ndarray],
    metadata: List[dict],
    output_dir: Path,
    case_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, meta in zip(frames, metadata):
        filename = f"{meta['frame_index']:03d}_frame.png"
        imageio.imwrite(output_dir / filename, frame)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    config = CropConfig(
        radius_scale=args.radius_scale,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        fixed_radius=args.fixed_radius,
        fade_decay_multiplier=max(args.fade_decay, 1e-3),
        min_fade_ratio=float(np.clip(args.min_fade, 0.0, 1.0)),
        edge_width=args.edge_width,
        highlight_color=(255, 64, 64),
        invert_image=not args.no_invert,
    )

    dicom_path = resolve_dicom_path(args.case_id, args.dicom_path, args.dicom_root)
    fixations = load_fixations(args.fixations_csv, args.case_id, args.limit)
    image = load_dicom_image(dicom_path, invert=config.invert_image)

    print(f"Loaded image {dicom_path} with shape {image.shape}")
    print(f"Found {len(fixations)} fixation(s) for case {args.case_id}")

    frames, metadata = circular_crop_frames(image, fixations, config)
    output_dir = args.output_root / args.case_id
    save_frames(frames, metadata, output_dir, args.case_id)

    print(f"Saved {len(frames)} frames to {output_dir}")
    print(f"Saved metadata to {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
