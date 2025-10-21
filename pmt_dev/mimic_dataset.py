from dataclasses import asdict, dataclass
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import json
import numpy as np
import pandas as pd
import pydicom
import yaml

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

def get_rows_containing_dicom_id(csv_path: Path, dicom_key: str, dicom_id: str) -> pd.DataFrame:
    """Return rows from a CSV where dicom_key == dicom_id."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if dicom_key not in df.columns:
        raise KeyError(f"Column '{dicom_key}' not found in {csv_path}")
    return df[df[dicom_key] == dicom_id].copy()

def get_row_by_dicom_id(csv_path: Path, dicom_key: str, dicom_id: str) -> pd.Series:
    """Return a single row from a CSV where dicom_key == dicom_id."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if dicom_key not in df.columns:
        raise KeyError(f"Column '{dicom_key}' not found in {csv_path}")
    row = df[df[dicom_key] == dicom_id]
    if row.empty:
        raise ValueError(f"No entry found for DICOM ID: {dicom_id} in {csv_path}")
    return row.iloc[0]

@dataclass
class InputFeature:
    dicom_id: str
    dicom_image: np.ndarray
    fixations: pd.DataFrame
    bounding_boxes: pd.DataFrame

    @classmethod
    def from_csv_files(
        cls,
        dicom_id: str,
        dicom_path: Path,
        fixations_csv: Path,
        bounding_boxes_csv: Path,
        pruner: Optional[FixationPruner] = None
    ) -> "InputFeature":
        """Factory constructor that initializes InputFeature from file paths."""
        dicom_image = DICOMImageLoader.load(dicom_path / f"{dicom_id}.dcm")
        fixation_dicom_id = "DICOM_ID"
        bounding_box_dicom_id = "dicom_id"
        fixations = get_rows_containing_dicom_id(fixations_csv, fixation_dicom_id, dicom_id)
        bounding_boxes = get_rows_containing_dicom_id(bounding_boxes_csv, bounding_box_dicom_id, dicom_id)

        if pruner is not None:
            fixations = pruner.prune(fixations, bounding_boxes)

        return cls(
            dicom_id=dicom_id,
            dicom_image=dicom_image,
            fixations=fixations,
            bounding_boxes=bounding_boxes
        )

@dataclass
class Label:
    dicom_id: str
    final_diagnosis: Optional[str]
    diagnoses: List[str]
    binary_labels: Dict[str, int]
    cxr_exam_indication: Optional[str]

    @classmethod
    def from_master_sheet(cls, dicom_id: str, master_sheet_csv: Path) -> "Label":
        """Factory constructor that creates a Label from the master sheet."""
        master_sheet_dicom_id = "dicom_id"
        row = get_row_by_dicom_id(master_sheet_csv, master_sheet_dicom_id, dicom_id)
        if row.empty:
            raise ValueError(f"No entry found for DICOM ID: {dicom_id}")
        diagnoses = cls.extract_diagnoses(row)
        return cls(
            dicom_id=dicom_id,
            diagnoses=diagnoses,
            final_diagnosis=diagnoses[0] if diagnoses else None,
            binary_labels=cls.extract_binary_labels(row),
            cxr_exam_indication=cls.extract_cxr_exam_indication(row),
        )

    @classmethod
    def from_master_sheet_dataframe(cls, dicom_id: str, master_sheet_df: pd.DataFrame) -> "Label":
        """Factory constructor that creates a Label from a DataFrame."""
        row = master_sheet_df[master_sheet_df["dicom_id"] == dicom_id]
        if row.empty:
            raise ValueError(f"No entry found for DICOM ID: {dicom_id}")
        row_series = row.iloc[0]
        diagnoses = cls.extract_diagnoses(row_series)
        return cls(
            dicom_id=dicom_id,
            diagnoses=diagnoses,
            final_diagnosis=diagnoses[0] if diagnoses else None,
            binary_labels=cls.extract_binary_labels(row_series),
            cxr_exam_indication=cls.extract_cxr_exam_indication(row_series),
        )

    # --- Static helper methods ---
    @staticmethod
    def extract_cxr_exam_indication(case_row: pd.Series) -> Optional[str]:
        return case_row["cxr_exam_indication"] if not case_row.empty else None

    @staticmethod
    def extract_diagnoses(case_row: pd.Series) -> List[str]:
        dx_columns = [col for col in case_row.index if col.startswith("dx") and "_icd" not in col]
        diagnoses: List[str] = []
        for col in sorted(dx_columns, key=lambda name: int(name[2:]) if name[2:].isdigit() else 0):
            value = case_row[col]
            if isinstance(value, str) and value.strip():
                diagnoses.append(value.strip())
        return diagnoses

    @staticmethod
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

class MimicDataset(Dataset):
    def __init__(self, config_path: str, pruner: Optional[FixationPruner] = None):
        config_loader = ConfigLoader(config_path)

        # Image DICOM path
        self.dicom_path = Path(config_loader.get('input_path', 'dicom_raw'))

        # Load CSV files
        gaze_path = Path(config_loader.get('input_path', 'gaze_raw'))
        self.master_sheet_csv = gaze_path / 'master_sheet.csv'
        self.bounding_boxes_csv = gaze_path / 'bounding_boxes.csv'
        self.fixations_csv = gaze_path / 'fixations.csv'

        # Cache master_sheet to avoid re-reading on every item
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found at {self.master_sheet_csv}")
        self.master_sheet_df = pd.read_csv(self.master_sheet_csv)

        # Optional pruner
        self.pruner = pruner

    def __len__(self):
        return len(self.master_sheet_df)

    def __getitem__(self, idx) -> Dict:
        # Bounds check
        if idx < 0 or idx >= len(self.master_sheet_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.master_sheet_df)}")
        record = self.master_sheet_df.iloc[idx]
        dicom_id = record['dicom_id']
        Logger.info(f"Fetching data for DICOM ID: {dicom_id}")

        # Load input features
        input_feature = InputFeature.from_csv_files(
            dicom_id=dicom_id,
            dicom_path=self.dicom_path,
            fixations_csv=self.fixations_csv,
            bounding_boxes_csv=self.bounding_boxes_csv,
            pruner=self.pruner
        )

        # Load labels
        label = Label.from_master_sheet_dataframe(
            dicom_id=dicom_id,
            master_sheet_df=self.master_sheet_df
        )
        return {
            "input_feature": input_feature,
            "label": label
        }

if __name__ == "__main__":
    # Example usage: print input and label as JSON
    try:
        dataset = MimicDataset(config_path='config/data_egd-cxr.yaml', pruner=FixationPruner())
        sample = dataset.__getitem__(1)
        Logger.info(f"Sample keys: {list(sample.keys())}")
        label_json = asdict(sample["label"])
        input_feature_json = asdict(sample["input_feature"])
        Logger.info(json.dumps(label_json, indent=2))
    except Exception as e:
        Logger.error(f"Error fetching/printing data: {e}")