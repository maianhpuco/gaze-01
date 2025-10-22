from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict
import json
import pandas as pd
import yaml
from dataclasses import asdict

from label_processor import LabelProcessor
from fixations_processor import FixationProcessor, CropConfig

class Logger:
    
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
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        Logger.info(f"\nLoading configuration from: {self.config_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open('r') as file:
            config = yaml.safe_load(file)
        
        Logger.info(json.dumps(config, indent=4))
        return config
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


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
    def __init__(self, config_path: str):
        config_loader = ConfigLoader(config_path)

        # Resolve and validate primary paths.
        dicom_root = config_loader.get('input_path', 'dicom_raw')
        if dicom_root is None:
            raise ValueError("Configuration missing 'input_path.dicom_raw'.")
        self.dicom_root = Path(dicom_root).expanduser()
        if not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.dicom_root}")

        gaze_path = Path(config_loader.get('input_path', 'gaze_raw')).expanduser()
        if not gaze_path.exists():
            raise FileNotFoundError(f"Gaze directory not found: {gaze_path}")
        self.master_sheet_csv = gaze_path / 'master_sheet.csv'
        self.bounding_boxes_csv = gaze_path / 'bounding_boxes.csv'
        self.fixations_csv = gaze_path / 'fixations.csv'

        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        if not self.fixations_csv.exists():
            raise FileNotFoundError(f"fixations.csv not found at {self.fixations_csv}")

        self.master_sheet_df = pd.read_csv(self.master_sheet_csv)
        if self.master_sheet_df.empty:
            raise ValueError(f"No rows found in {self.master_sheet_csv}")

        self.crop_config = self._build_crop_config(config_loader)
        self.fixation_processor = FixationProcessor(
            config=self.crop_config,
            dicom_root=self.dicom_root,
            fixations_csv=self.fixations_csv,
        )
        self.label_processor = LabelProcessor(self.master_sheet_csv)

    def __len__(self):
        return len(self.master_sheet_df)

    def __getitem__(self, idx) -> Dict:
        # Bounds check
        if idx < 0 or idx >= len(self.master_sheet_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.master_sheet_df)}")
        record = self.master_sheet_df.iloc[idx]
        dicom_id = record['dicom_id']
        Logger.info(f"Fetching data for DICOM ID: {dicom_id}")

        dicom_path = self._resolve_dicom_path(dicom_id)

        # Load input features
        frames, metadata = self.fixation_processor.get_frames(
            case_id=dicom_id,
            dicom_path=dicom_path,
        )

        input_feature = {
            "dicom_id": dicom_id,
            "frames": frames,
            "metadata": metadata,
        }

        # Load labels
        labels = self.label_processor.get_labels(dicom_id)
        

        return {
            "input_feature": input_feature,
            "label": labels
        }

    def _build_crop_config(self, config_loader: ConfigLoader) -> CropConfig:
        raw_overrides = config_loader.get('crop_config', default={}) or {}
        unknown_keys = set(raw_overrides) - set(DEFAULT_CROP_CONFIG)
        if unknown_keys:
            Logger.info(f"Ignoring unknown crop_config keys: {sorted(unknown_keys)}")
        overrides = {key: raw_overrides[key] for key in DEFAULT_CROP_CONFIG if key in raw_overrides}
        params = {**DEFAULT_CROP_CONFIG, **overrides}

        highlight_color = params.get('highlight_color', DEFAULT_CROP_CONFIG['highlight_color'])
        if isinstance(highlight_color, (list, tuple)):
            highlight_color = tuple(int(x) for x in highlight_color)
        else:
            raise ValueError("crop_config.highlight_color must be a sequence of three integers")
        if len(highlight_color) != 3:
            raise ValueError("crop_config.highlight_color must contain exactly three values (R, G, B)")
        params['highlight_color'] = highlight_color

        return CropConfig(**params)

    def _resolve_dicom_path(self, dicom_id: str) -> Path:
        """Find the DICOM file for a row, falling back to <dicom_id>.dcm under dicom_root."""
        return self.dicom_root / f"{dicom_id}.dcm"
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mimic Dataset Preparation")
    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / "config/data_egd-cxr.yaml"
    print(config_path)
    parser.add_argument("--config-path", type=str, help="Path to the configuration file", default=config_path)
    args = parser.parse_args()

    config_path = args.config_path
    dataset = MimicDataset(config_path)
    sample = dataset.__getitem__(1)
    # Print a compact summary to avoid huge array dumps
    label_dict = asdict(sample["label"])
    label_dict["source_csv"] = str(label_dict.get("source_csv"))
    frames = sample["input_feature"]["frames"]
    meta = sample["input_feature"]["metadata"]
    summary = {
        "dicom_id": sample["input_feature"]["dicom_id"],
        "frames_count": len(frames),
        "first_frame_shape": list(frames[0].shape) if frames else None,
        "metadata_len": len(meta),
        "label": label_dict,
    }
    Logger.info(json.dumps(summary, indent=2))

    # save image frames to disk for visual inspection
    output_dir = base_dir / "output_frames"
    output_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(output_dir / f"{sample['input_feature']['dicom_id']}_frame_{i:03d}.png")
    Logger.success(f"Saved {len(frames)} frames to {output_dir}")