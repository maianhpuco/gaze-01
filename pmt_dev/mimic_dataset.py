from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
import pydicom
import yaml
import json
from typing import Dict, Tuple, Optional, List

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
        self.base_dir = Path(__file__).parent.resolve()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        Logger.info(f"\nLoading configuration from: {self.config_path}")
        
        # Check if path is absolute or relative
        if self.config_path.is_absolute():
            full_path = self.config_path
        else:
            full_path = self.base_dir / self.config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        with full_path.open('r') as file:
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

class DICOMImageLoader:
    
    @staticmethod
    def load(dicom_path: Path) -> np.ndarray:  # Fixed return type
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array.astype(np.float32)  # Convert to float
        
        # Apply windowing
        window_center = DICOMImageLoader._get_window_value(
            dicom, 'WindowCenter', default=40
        )
        window_width = DICOMImageLoader._get_window_value(
            dicom, 'WindowWidth', default=400
        )
        
        # Process pixel array
        pixel_array = DICOMImageLoader._apply_windowing(
            pixel_array, window_center, window_width
        )
        
        Logger.success(
            f"Loaded DICOM: {pixel_array.shape}, "
            f"range: {pixel_array.min():.3f}-{pixel_array.max():.3f}"
        )
        
        return pixel_array
    
    @staticmethod
    def _get_window_value(dicom: pydicom.Dataset, attr: str, default: int) -> float:  # Changed return type
        value = getattr(dicom, attr, default)
        try:
            if isinstance(value, (list, tuple)):
                value = value[0]
            return float(value)
        except Exception:
            Logger.error(f"Unable to parse DICOM tag {attr}; using default {default}")
            return float(default)
    
    @staticmethod
    def _apply_windowing(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
        min_val = center - width / 2
        max_val = center + width / 2
        
        # Clip and normalize
        pixel_array = np.clip(pixel_array, min_val, max_val)
        
        if max_val > min_val:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = pixel_array / (pixel_array.max() + 1e-8)  # Added epsilon to avoid division by zero
        
        # Invert for chest X-ray display
        return 1.0 - pixel_array

class FixationPruner:
    
    @staticmethod
    def prune(fixations: pd.DataFrame, bounding_boxes: pd.DataFrame) -> pd.DataFrame:
        # Validate necessary columns
        required_box_cols = {"x1", "y1", "x2", "y2"}
        required_fix_cols = {"FPOGX", "FPOGY"}
        if not required_box_cols.issubset(set(bounding_boxes.columns)):
            Logger.error("Bounding boxes missing required columns {x1,y1,x2,y2}; returning empty pruned fixations.")
            return pd.DataFrame(columns=fixations.columns)
        if not required_fix_cols.issubset(set(fixations.columns)):
            Logger.error("Fixations missing required columns {FPOGX,FPOGY}; returning empty pruned fixations.")
            return pd.DataFrame(columns=fixations.columns)

        pruned_fixations = []
        
        for _, box in bounding_boxes.iterrows():
            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']

            box_fixations = fixations[
                (fixations['FPOGX'] >= x1) & (fixations['FPOGX'] <= x2) &
                (fixations['FPOGY'] >= y1) & (fixations['FPOGY'] <= y2)
            ]
            
            pruned_fixations.append(box_fixations)
        
        if pruned_fixations:
            return pd.concat(pruned_fixations).drop_duplicates().reset_index(drop=True)  # Added drop_duplicates
        else:
            return pd.DataFrame(columns=fixations.columns)

class MimicDataset(Dataset):
    def __init__(self, config_path: str, pruner: Optional[FixationPruner] = None):
        config_loader = ConfigLoader(config_path)
        gaze_path = Path(config_loader.get('input_path', 'gaze_raw'))

        self.dicom_path = Path(config_loader.get('input_path', 'dicom_raw'))
        self.master_sheet = pd.read_csv(gaze_path / 'master_sheet.csv')
        self.bounding_boxes = pd.read_csv(gaze_path / 'bounding_boxes.csv')
        self.fixations = pd.read_csv(gaze_path / 'fixations.csv')
        self.pruner = pruner

    def __len__(self):
        return len(self.master_sheet)

    def __getitem__(self, idx) -> Dict:
        record = self.master_sheet.iloc[idx]
        dicom_id = record['dicom_id']
        dicom_image = DICOMImageLoader.load(self.dicom_path / f"{dicom_id}.dcm")
        fix_id_col = 'DICOM_ID' if 'DICOM_ID' in self.fixations.columns else ('dicom_id' if 'dicom_id' in self.fixations.columns else None)
        box_id_col = 'dicom_id' if 'dicom_id' in self.bounding_boxes.columns else ('DICOM_ID' if 'DICOM_ID' in self.bounding_boxes.columns else None)
        if fix_id_col is None:
            raise KeyError("Neither 'DICOM_ID' nor 'dicom_id' found in fixations.csv")
        if box_id_col is None:
            raise KeyError("Neither 'dicom_id' nor 'DICOM_ID' found in bounding_boxes.csv")

        fixations = self.fixations[self.fixations[fix_id_col] == dicom_id].copy()
        bounding_boxes = self.bounding_boxes[self.bounding_boxes[box_id_col] == dicom_id].copy()

        if self.pruner:
            fixations = self.pruner.prune(fixations, bounding_boxes)

        # Return a single sample dict (not nested by idx) for PyTorch collation
        return {
            'dicom_id': dicom_id,
            'image': dicom_image,
            'fixations': fixations,
            'bounding_boxes': bounding_boxes
        }


def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function to handle pandas DataFrames and variable-size images.
    - Keeps 'fixations' and 'bounding_boxes' as lists of DataFrames
    - Keeps 'image' as a list to avoid stacking failures for varying HxW
    - Aggregates 'dicom_id' as a list of strings
    """
    return {
        'dicom_id': [sample['dicom_id'] for sample in batch],
        'image': [sample['image'] for sample in batch],
        'fixations': [sample['fixations'] for sample in batch],
        'bounding_boxes': [sample['bounding_boxes'] for sample in batch],
    }

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / 'config/data_egd-cxr.yaml'
    pruner = FixationPruner()
    dataset = MimicDataset(config_path, pruner=pruner)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

    for batch in dataloader:
        Logger.info(f"Batch DICOM IDs: {batch['dicom_id']}")
        Logger.info(f"Batch Image Shapes: {[img.shape for img in batch['image']]}")
