# Import required libraries
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
from PIL import Image


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
    def load(dicom_path: Path) -> Tuple[np.ndarray, pydicom.Dataset]:
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        
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
        
        return pixel_array, dicom
    
    @staticmethod
    def _get_window_value(dicom: pydicom.Dataset, attr: str, default: int) -> int:
        value = getattr(dicom, attr, default)
        return value[0] if isinstance(value, list) else value
    
    @staticmethod
    def _apply_windowing(pixel_array: np.ndarray, center: int, width: int) -> np.ndarray:
        min_val = center - width // 2
        max_val = center + width // 2
        
        # Clip and normalize
        pixel_array = np.clip(pixel_array, min_val, max_val)
        
        if max_val > min_val:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = pixel_array / pixel_array.max()
        
        # Invert for chest X-ray display
        return 1.0 - pixel_array


class AnatomicalMaskLoader:
    
    REGIONS = ['aortic_knob', 'left_lung', 'right_lung', 'mediastanum']
    
    @staticmethod
    def load(gaze_path: Path, dicom_id: str) -> Dict[str, np.ndarray]:
        audio_dir = gaze_path / 'audio_segmentation_transcripts' / dicom_id
        
        if not audio_dir.exists():
            raise FileNotFoundError(
                f"Audio segmentation directory not found for DICOM ID: {dicom_id}"
            )
        
        masks = {}
        for region in AnatomicalMaskLoader.REGIONS:
            img_path = audio_dir / f"{region}.png"
            if img_path.exists():
                img = Image.open(img_path)
                masks[region] = np.array(img)
                Logger.success(f"Loaded {region} mask: {masks[region].shape}")
        
        return masks


class GazeDataProcessor:
    
    def __init__(self, dicom_image: np.ndarray, gaze_data: pd.DataFrame):
        self.dicom_image = dicom_image
        self.gaze_data = gaze_data
        self.img_h, self.img_w = dicom_image.shape[:2]
    
    def get_image_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        gaze_x = self.gaze_data['FPOGX'] * self.img_w
        gaze_y = self.gaze_data['FPOGY'] * self.img_h
        return gaze_x, gaze_y
    
    def plot_fixation_analysis(self, ax: plt.Axes) -> None:
        if len(self.gaze_data) == 0:
            Logger.error("No gaze data available")
            return
        
        # Display DICOM image
        ax.imshow(self.dicom_image, cmap='gray')
        
        # Get coordinates
        gaze_x, gaze_y = self.get_image_coordinates()
        
        # Draw transition lines
        self._draw_transitions(ax, gaze_x, gaze_y)
        
        # Draw fixation points
        scatter = self._draw_fixations(ax, gaze_x, gaze_y)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Fixation Duration (seconds)', fontsize=12)
        
        # Add markers
        self._add_markers(ax, gaze_x, gaze_y)
        
        # Set title
        self._set_title(ax)
        ax.axis('off')
    
    def _draw_transitions(self, ax: plt.Axes, gaze_x: pd.Series, gaze_y: pd.Series) -> None:
        for i in range(len(self.gaze_data) - 1):
            x1, y1 = gaze_x.iloc[i], gaze_y.iloc[i]
            x2, y2 = gaze_x.iloc[i + 1], gaze_y.iloc[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=1)
    
    def _draw_fixations(self, ax: plt.Axes, gaze_x: pd.Series, gaze_y: pd.Series):
        return ax.scatter(
            gaze_x, gaze_y,
            c=self.gaze_data['FPOGD'],
            cmap='hot',
            s=self.gaze_data['FPOGD'] * 1000,
            alpha=0.8,
            edgecolors='white',
            linewidth=1
        )
    
    def _add_markers(self, ax: plt.Axes, gaze_x: pd.Series, gaze_y: pd.Series) -> None:
        if len(self.gaze_data) > 0:
            ax.scatter(gaze_x.iloc[0], gaze_y.iloc[0], c='green', s=200,
                      marker='o', label='Start', edgecolors='white', linewidth=2)
            ax.scatter(gaze_x.iloc[-1], gaze_y.iloc[-1], c='red', s=200,
                      marker='s', label='End', edgecolors='white', linewidth=2)
            ax.legend()
    
    def _set_title(self, ax: plt.Axes) -> None:
        total_duration = self.gaze_data["Time (in secs)"].max()
        ax.set_title(
            f'Fixation Analysis (Real DICOM Image)\n'
            f'Total Fixations: {len(self.gaze_data)}, Duration: {total_duration:.1f}s',
            fontsize=16, fontweight='bold'
        )


class CaseProcessor:
    
    def __init__(self, config: ConfigLoader, dicom_path: Path, gaze_path: Path):
        self.config = config
        self.dicom_path = dicom_path
        self.gaze_path = gaze_path
        self.base_dir = Path(__file__).parent.resolve()
    
    def process(
        self,
        case: pd.Series,
        bounding_boxes: pd.DataFrame,
        fixations: pd.DataFrame
    ) -> None:
        dicom_id = case['dicom_id']
        Logger.info(f"\nProcessing case: {dicom_id}")
        
        # Create output directory
        case_dir = self._create_case_directory(dicom_id)
        
        # Filter data for this case
        bboxes = bounding_boxes[bounding_boxes['dicom_id'] == dicom_id]
        gaze_data = fixations[fixations['DICOM_ID'] == dicom_id]
        
        # Load data
        anatomical_masks = AnatomicalMaskLoader.load(self.gaze_path, dicom_id)
        dicom_image, dicom = self._load_dicom(dicom_id)
        
        # Print summary
        self._print_summary(bboxes, gaze_data, anatomical_masks, dicom)
        
        # Create visualizations
        self._create_fixation_plot(gaze_data, dicom_image, case_dir)
    
    def _create_case_directory(self, dicom_id: str) -> Path:
        case_dir = (
            self.base_dir /
            self.config.get('output_path', 'base_plots_dir', default='output') /
            dicom_id
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        Logger.success(f"Created case directory: {case_dir}")
        return case_dir
    
    def _load_dicom(self, dicom_id: str) -> Tuple[np.ndarray, pydicom.Dataset]:

        dicom_file = self.dicom_path / f"{dicom_id}.dcm"
        dicom_image, dicom = DICOMImageLoader.load(dicom_file)
        Logger.success(f"Successfully loaded DICOM image: {dicom_file}")
        return dicom_image, dicom
    
    def _print_summary(
        self,
        bboxes: pd.DataFrame,
        gaze_data: pd.DataFrame,
        anatomical_masks: Dict,
        dicom: pydicom.Dataset
    ) -> None:

        Logger.info(f"\nData Summary:")
        Logger.info(f"• Bounding boxes: {len(bboxes)}")
        Logger.info(f"• Gaze fixations: {len(gaze_data)}")
        Logger.info(f"• Anatomical masks: {len(anatomical_masks)}")
        Logger.info(f"• Patient ID: {getattr(dicom, 'PatientID', 'N/A')}")
        Logger.info(f"• Study Date: {getattr(dicom, 'StudyDate', 'N/A')}")
        Logger.info(f"• Image size: {dicom.pixel_array.shape}")
    
    def _create_fixation_plot(
        self,
        gaze_data: pd.DataFrame,
        dicom_image: np.ndarray,
        case_dir: Path
    ) -> None:
        Logger.info("\nCreating Plot: Fixation Analysis...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        processor = GazeDataProcessor(dicom_image, gaze_data)
        processor.plot_fixation_analysis(ax)
        
        plot_path = case_dir / "fixation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        Logger.success(f"Saved: {plot_path}")
        plt.close()


class DataPipeline:
    def __init__(self, config_path: str):
        self.config = ConfigLoader(config_path)
        self._load_paths()
        self._load_dataframes()
    
    def _load_paths(self) -> None:
        self.dicom_path = Path(self.config.get('input_path', 'dicom_raw'))
        self.gaze_path = Path(self.config.get('input_path', 'gaze_raw'))
        
        if not self.gaze_path.exists() or not self.dicom_path.exists():
            raise FileNotFoundError("Gaze or DICOM raw data path does not exist.")
    
    def _load_dataframes(self) -> None:
        self.master_sheet = pd.read_csv(self.gaze_path / 'master_sheet.csv')
        self.bounding_boxes = pd.read_csv(self.gaze_path / 'bounding_boxes.csv')
        self.fixations = pd.read_csv(self.gaze_path / 'fixations.csv')
    
    def run(self) -> None:
        case = self.master_sheet.iloc[2] # Process the second case as an example
        processor = CaseProcessor(self.config, self.dicom_path, self.gaze_path)
        processor.process(case, self.bounding_boxes, self.fixations)


def main():
    try:
        pipeline = DataPipeline('config/data_egd-cxr.yaml')
        pipeline.run()
    except Exception as e:
        Logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()