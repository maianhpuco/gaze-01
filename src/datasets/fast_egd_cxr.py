#!/usr/bin/env python3
"""
Fast PNG-based dataset for EGD-CXR training.

This dataset loads preprocessed PNG images instead of DICOM files,
providing 5-10x faster loading speeds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# import imageio.v2 as imageio  # Using PIL instead for PNG loading
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Configure threading to avoid conflicts in multiprocessing environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Column names for eye tracking data in the fixations CSV
TIME_COLUMN = "Time (in secs)"
X_COLUMN = "FPOGX"
Y_COLUMN = "FPOGY"
DURATION_COLUMN = "FPOGD"
DEFAULT_IMAGE_SIZE = (224, 224)


@dataclass(frozen=True)
class ClassificationLabel:
    """Classification label following the notebook logic."""
    index: int  # -1 for unknown, 0 for CHF, 1 for pneumonia, 2 for Normal
    name: str
    classes: Tuple[str, ...] = ("CHF", "pneumonia", "Normal")
    ambiguous: bool = False

    def one_hot(self) -> torch.Tensor:
        """Convert to one-hot encoding."""
        vec = torch.zeros(len(self.classes), dtype=torch.float32)
        if self.index >= 0:
            vec[self.index] = 1.0
        return vec


class FastEGDCXRDataset(Dataset):
    """
    Fast PNG-based EGD-CXR dataset for rapid training.
    
    This dataset loads preprocessed PNG images instead of DICOM files,
    providing 5-10x faster loading speeds while maintaining the same
    data structure as the original dataset.
    """

    def __init__(
        self,
        *,
        root: Path,
        png_dir: Optional[Path] = None,  # Directory containing preprocessed PNG files
        seg_path: Path,
        transcripts_path: Optional[Path] = None,
        max_fixations: Optional[int] = None,
        case_ids: Optional[Sequence[str]] = None,
        classes: Sequence[str] = ("CHF", "Pneumonia", "Normal"),
        drop_unlabelled: bool = True,
    ) -> None:
        super().__init__()
        
        self.root = Path(root).expanduser()
        self.png_dir = Path(png_dir).expanduser() if png_dir else None
        self.seg_path = Path(seg_path).expanduser()
        self.transcripts_path = Path(transcripts_path).expanduser() if transcripts_path else self.seg_path
        self.max_fixations = max_fixations
        self.classes = tuple(classes)
        
        # Load master sheet
        master_sheet_path = self.root / "master_sheet.csv"
        if not master_sheet_path.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {master_sheet_path}")
        
        self.master_df = pd.read_csv(master_sheet_path, engine="python")
        if self.master_df.empty:
            raise ValueError(f"master_sheet.csv has no rows: {master_sheet_path}")
        
        # Reduced debug output - only show essential info
        
        # Filter by case_ids if provided
        if case_ids:
            self.master_df = self.master_df[self.master_df["dicom_id"].isin(case_ids)].reset_index(drop=True)
            print(f"📋 After filtering by case_ids: {len(self.master_df)} rows")
        
        # Load additional data files
        self.fixations_df = self._load_fixations()
        self.bbox_df = self._load_bounding_boxes()
        
        # Process cases and apply classification logic
        self._process_cases(drop_unlabelled)
        
        print(f"FastEGDCXRDataset ready: {len(self._indices)} cases | classes={len(self.classes)}")
        if self.png_dir:
            pass  # PNG directory available
        else:
            print("⚠️  No PNG directory specified - will use dummy images")
        
        # Add class_names attribute for compatibility
        self.class_names = self.classes

    def _load_fixations(self) -> pd.DataFrame:
        """Load fixations CSV file."""
        fixations_path = self.root / "fixations.csv"
        if not fixations_path.exists():
            print(f"⚠️  Fixations file not found: {fixations_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(fixations_path, engine="python")
        # Loaded fixation records
        return df

    def _load_bounding_boxes(self) -> pd.DataFrame:
        """Load bounding boxes CSV file."""
        bbox_path = self.root / "bounding_boxes.csv"
        if not bbox_path.exists():
            print(f"⚠️  Bounding boxes file not found: {bbox_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(bbox_path, engine="python")
        # Loaded bounding box records
        return df

    def _process_cases(self, drop_unlabelled: bool) -> None:
        """Process cases and apply classification logic."""
        self._indices = []
        self._targets = []
        
        # Processing cases...
        
        for idx, row in self.master_df.iterrows():
            dicom_id = row["dicom_id"]
            
            # Check if PNG file exists (if PNG directory is specified)
            if self.png_dir:
                png_path = self.png_dir / f"{dicom_id}.png"
                if not png_path.exists():
                    if drop_unlabelled:
                        continue
                    else:
                        print(f"⚠️  PNG not found for {dicom_id}: {png_path}")
            
            # Apply classification logic
            classification = self._classify_case(row)
            
            # Debug: show first few classifications (reduced output)
            
            if classification.index >= 0 or not drop_unlabelled:
                self._indices.append(idx)
                self._targets.append(classification.index)
        
        # Processed valid cases
        
        # Calculate class counts
        if self._targets:
            targets_array = np.array(self._targets)
            self.class_counts = torch.tensor([
                np.sum(targets_array == i) for i in range(len(self.classes))
            ], dtype=torch.long)
            print(f"📊 Class distribution: {self.class_counts.tolist()}")
        else:
            self.class_counts = torch.zeros(len(self.classes), dtype=torch.long)

    def _classify_case(self, row: pd.Series) -> ClassificationLabel:
        """Apply classification logic following the notebook."""
        # Based on the notebook: use CHF, pneumonia, Normal columns directly
        # Priority: CHF > pneumonia > Normal (first match wins)
        
        # CHF classification (priority 1)
        if row.get("CHF", 0) == 1:
            return ClassificationLabel(index=0, name="CHF", classes=self.classes)
        
        # Pneumonia classification (priority 2)  
        if row.get("pneumonia", 0) == 1:
            return ClassificationLabel(index=1, name="Pneumonia", classes=self.classes)
        
        # Normal classification (priority 3)
        if row.get("Normal", 0) == 1:
            return ClassificationLabel(index=2, name="Normal", classes=self.classes)
        
        # Unknown/unlabelled
        return ClassificationLabel(index=-1, name="Unknown", classes=self.classes)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        master_idx = self._indices[index]
        row = self.master_df.iloc[master_idx]
        dicom_id = row["dicom_id"]
        target = self._targets[index]
        
        # Load preprocessed PNG image (FAST!)
        image = self._load_png_image(dicom_id)
        
        # Load other modalities (same as original)
        segments = self._load_segmentation(dicom_id)
        box_masks = self._load_bounding_box_masks(dicom_id)
        fixations = self._load_fixation_data(dicom_id)
        transcript = self._load_transcript(dicom_id)
        
        # Create classification label
        classification = ClassificationLabel(
            index=target, 
            name=self.classes[target] if target >= 0 else "Unknown",
            classes=self.classes
        )
        
        return {
            "dicom_id": dicom_id,
            "image": image,  # Preprocessed PNG tensor
            "segments": segments,
            "box_masks": box_masks,
            "fixations": fixations,
            "transcript": transcript,
            "labels": {
                "classification": {
                    "index": classification.index,
                    "name": classification.name,
                    "classes": classification.classes,
                    "one_hot": classification.one_hot(),
                    "ambiguous": classification.ambiguous,
                },
                "single_index": torch.tensor(classification.index, dtype=torch.long),
                "single_name": classification.name,
                "single_class_names": classification.classes,
            },
        }

    def _load_png_image(self, dicom_id: str) -> torch.Tensor:
        """Load preprocessed PNG image (FAST!)."""
        if not self.png_dir:
            # Return dummy image if no PNG directory specified
            return torch.zeros((1, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
            
        png_path = self.png_dir / f"{dicom_id}.png"
        
        if not png_path.exists():
            # Return dummy image if PNG not found
            return torch.zeros((1, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
        
        try:
            # Load PNG using PIL (more reliable than imageio)
            from PIL import Image
            img_pil = Image.open(png_path)
            
            # Convert to grayscale if needed
            if img_pil.mode != 'L':
                img_pil = img_pil.convert('L')
            
            # Resize to DEFAULT_IMAGE_SIZE if needed
            if img_pil.size != DEFAULT_IMAGE_SIZE:
                img_pil = img_pil.resize(DEFAULT_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            image = np.array(img_pil).astype(np.float32) / 255.0
            
            # Convert to tensor and add channel dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"❌ Error loading PNG {png_path}: {e}")
            return torch.zeros((1, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)

    def _load_segmentation(self, dicom_id: str) -> torch.Tensor:
        """Load segmentation masks (same as original)."""
        case_dir = self.seg_path / dicom_id
        if not case_dir.exists():
            return torch.zeros((4, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
        
        region_names = sorted([png.stem for png in case_dir.glob("*.png")])
        if not region_names:
            return torch.zeros((4, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
        
        masks = []
        for region_name in region_names:
            mask_path = case_dir / f"{region_name}.png"
            try:
                from PIL import Image
                mask_pil = Image.open(mask_path)
                if mask_pil.mode != 'L':
                    mask_pil = mask_pil.convert('L')
                mask = np.array(mask_pil).astype(np.float32) / 255.0
                
                # Resize mask to DEFAULT_IMAGE_SIZE if it's not already that size
                if mask.shape != DEFAULT_IMAGE_SIZE:
                    from PIL import Image
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(DEFAULT_IMAGE_SIZE, Image.Resampling.NEAREST)
                    mask = np.array(mask_pil).astype(np.float32) / 255.0
                
                masks.append(mask)
            except Exception:
                masks.append(np.zeros(DEFAULT_IMAGE_SIZE, dtype=np.float32))
        
        if masks:
            return torch.from_numpy(np.stack(masks))
        else:
            return torch.zeros((4, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)

    def _load_bounding_box_masks(self, dicom_id: str) -> torch.Tensor:
        """Load bounding box masks for all 17 regions."""
        if self.bbox_df.empty:
            return torch.zeros((17, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
        
        rows = self.bbox_df[self.bbox_df["dicom_id"] == dicom_id]
        if rows.empty:
            return torch.zeros((17, *DEFAULT_IMAGE_SIZE), dtype=torch.float32)
        
        # Get all unique bbox names and create a mapping
        all_bbox_names = sorted(self.bbox_df['bbox_name'].unique())
        bbox_name_to_idx = {name: idx for idx, name in enumerate(all_bbox_names)}
        
        # Initialize masks for all 17 regions
        masks = [np.zeros(DEFAULT_IMAGE_SIZE, dtype=np.float32) for _ in range(17)]
        
        for _, row in rows.iterrows():
            bbox_name = row["bbox_name"]
            if bbox_name in bbox_name_to_idx:
                idx = bbox_name_to_idx[bbox_name]
                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                # Convert to DEFAULT_IMAGE_SIZE coordinates
                H, W = DEFAULT_IMAGE_SIZE
                x1_norm, x2_norm = int(x1 * W / 2048), int(x2 * W / 2048)  # Assuming original image is 2048x2048
                y1_norm, y2_norm = int(y1 * H / 2048), int(y2 * H / 2048)
                masks[idx][y1_norm:y2_norm, x1_norm:x2_norm] = 1.0
        
        return torch.from_numpy(np.stack(masks))

    def _load_fixation_data(self, dicom_id: str) -> Dict[str, torch.Tensor]:
        """Load fixation data with region hits (seg_hits and box_hits)."""
        case_fixations = self.fixations_df[self.fixations_df["DICOM_ID"] == dicom_id].copy()
        
        if case_fixations.empty:
            return {
                "xy": torch.zeros((0, 2), dtype=torch.float32),
                "time": torch.zeros(0, dtype=torch.float32),
                "dwell": torch.zeros(0, dtype=torch.float32),
                "seg_hits": torch.zeros((0, 4), dtype=torch.float32),  # 4 regions (heart, left_lung, right_lung, mediastinum)
                "box_hits": torch.zeros((0, 17), dtype=torch.float32),  # 17 bounding box regions
            }
        
        # Filter valid fixations
        case_fixations = case_fixations[
            case_fixations[X_COLUMN].between(0.0, 1.0) &
            case_fixations[Y_COLUMN].between(0.0, 1.0) &
            case_fixations[DURATION_COLUMN].notna() &
            (case_fixations[DURATION_COLUMN] > 0)
        ].copy()
        
        if case_fixations.empty:
            return {
                "xy": torch.zeros((0, 2), dtype=torch.float32),
                "time": torch.zeros(0, dtype=torch.float32),
                "dwell": torch.zeros(0, dtype=torch.float32),
                "seg_hits": torch.zeros((0, 4), dtype=torch.float32),
                "box_hits": torch.zeros((0, 17), dtype=torch.float32),
            }
        
        # Limit fixations if specified
        if self.max_fixations:
            case_fixations = case_fixations.head(self.max_fixations)
        
        xy = torch.tensor(case_fixations[[X_COLUMN, Y_COLUMN]].values, dtype=torch.float32)
        times = torch.tensor(case_fixations[TIME_COLUMN].values, dtype=torch.float32)
        dwell = torch.tensor(case_fixations[DURATION_COLUMN].values, dtype=torch.float32)
        
        # Load segmentation and bounding box masks to compute hits
        segments = self._load_segmentation(dicom_id)  # Shape: [4, H, W] or [0, H, W] if not available
        box_masks = self._load_bounding_box_masks(dicom_id)  # Shape: [17, H, W]
        
        # Convert normalized coordinates to pixel coordinates
        H, W = DEFAULT_IMAGE_SIZE
        pixel_coords = xy * torch.tensor([W, H], dtype=torch.float32)
        pixel_coords = pixel_coords.long()
        
        # Compute segmentation hits (4 regions: heart, left_lung, right_lung, mediastinum)
        seg_hits = torch.zeros((len(xy), 4), dtype=torch.float32)
        if segments.shape[0] > 0:  # If segmentation masks are available
            for i, (x, y) in enumerate(pixel_coords):
                x = max(0, min(x, W-1))  # Clamp to valid range
                y = max(0, min(y, H-1))
                seg_hits[i] = segments[:, y, x]  # Check all 4 regions at this pixel
        
        # Compute bounding box hits (17 regions)
        box_hits = torch.zeros((len(xy), 17), dtype=torch.float32)
        if box_masks.shape[0] > 0:  # If bounding box masks are available
            for i, (x, y) in enumerate(pixel_coords):
                x = max(0, min(x, W-1))  # Clamp to valid range
                y = max(0, min(y, H-1))
                box_hits[i] = box_masks[:, y, x]  # Check all 17 bounding boxes at this pixel
        
        return {
            "xy": xy,
            "time": times,
            "dwell": dwell,
            "seg_hits": seg_hits,
            "box_hits": box_hits,
        }

    def _load_transcript(self, dicom_id: str) -> str:
        """Load transcript data (same as original)."""
        transcript_path = self.transcripts_path / f"{dicom_id}.json"
        if not transcript_path.exists():
            return ""
        
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
            return data.get("transcript", "")
        except Exception:
            return ""

    def class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        counts = self.class_counts.float()
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        return weights

    def sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced sampling."""
        weights = self.class_weights()
        targets = torch.tensor(self._targets, dtype=torch.long)
        return weights[targets]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom batch collation function (same as original)."""
    # Extract case IDs and sequence lengths
    dicom_ids = [item["dicom_id"] for item in batch]
    lengths = torch.tensor([item["fixations"]["xy"].shape[0] for item in batch], dtype=torch.long)

    # Pad variable-length sequences
    xy = pad_sequence([item["fixations"]["xy"] for item in batch], batch_first=True)
    dwell = pad_sequence([item["fixations"]["dwell"] for item in batch], batch_first=True)
    times = pad_sequence([item["fixations"]["time"] for item in batch], batch_first=True)
    seg_hits = pad_sequence([item["fixations"]["seg_hits"] for item in batch], batch_first=True)
    box_hits = pad_sequence([item["fixations"]["box_hits"] for item in batch], batch_first=True)

    # Stack fixed-size tensors
    images = torch.stack([item["image"] for item in batch], dim=0)
    segments = torch.stack([item["segments"] for item in batch], dim=0)
    box_masks = torch.stack([item["box_masks"] for item in batch], dim=0)
    transcripts = [item["transcript"] for item in batch]

    # Stack classification labels
    classification_one_hot = torch.stack([item["labels"]["classification"]["one_hot"] for item in batch], dim=0)
    single_index = torch.stack([item["labels"]["single_index"] for item in batch], dim=0)

    return {
        "dicom_ids": dicom_ids,
        "lengths": lengths,
        "images": images,
        "segments": segments,
        "box_masks": box_masks,
        "transcripts": transcripts,
        "fixations": {
            "xy": xy,
            "dwell": dwell,
            "time": times,
            "seg_hits": seg_hits,
            "box_hits": box_hits,
        },
        "labels": {
            "classification": {
                "one_hot": classification_one_hot,
            },
            "single_index": single_index,
        },
    }


def create_fast_dataloader(
    dataset: FastEGDCXRDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 4,  # Can use more workers with PNG!
    pin_memory: bool = True,
) -> DataLoader:
    """Create optimized DataLoader for fast PNG dataset."""
    # Only use multiprocessing optimizations if num_workers > 0
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,        # Prefetch batches
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
