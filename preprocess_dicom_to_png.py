#!/usr/bin/env python3
"""
Preprocess DICOM images to PNG format for faster training.

This script converts all DICOM files to preprocessed PNG images with:
- Medical windowing applied
- Normalization to [0, 1]
- Resizing to 224x224
- Grayscale format

Usage:
    # Use default paths from configs/data_egd_cxr_single_label.yaml
    python preprocess_dicom_to_png.py
    
    # Or specify custom paths
    python preprocess_dicom_to_png.py --dicom-dir /path/to/dicom --output-dir /path/to/png
    
    # Dry run to see what would be processed
    python preprocess_dicom_to_png.py --dry-run
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Repo-local imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

try:
    from egd_cxr_dataset import ConfigLoader  # type: ignore
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    print("❌ pydicom not installed. Run: pip install pydicom")
    HAS_PYDICOM = False
    exit(1)


def process_dicom_to_png(
    dicom_path: Path, 
    output_path: Path, 
    target_size: tuple = (224, 224)
) -> bool:
    """
    Convert a single DICOM file to preprocessed PNG.
    
    Args:
        dicom_path: Path to DICOM file
        output_path: Path to save PNG file
        target_size: Target image size (width, height)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load DICOM with error handling for corrupted files
        try:
            ds = pydicom.dcmread(str(dicom_path))
        except Exception as e:
            print(f"⚠️  Skipping corrupted DICOM {dicom_path.name}: {e}")
            return False
            
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"⚠️  Skipping DICOM with invalid pixel data {dicom_path.name}: {e}")
            return False
        
        # Apply medical windowing
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        
        # Get window center and width
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
        
        # Ensure single channel
        if arr.ndim == 3:
            arr = arr[..., 0]
        
        # Convert to PIL Image and resize
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save as PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format='PNG', optimize=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dicom_path}: {e}")
        return False


def process_batch(args_tuple):
    """Process a batch of DICOM files."""
    dicom_files, output_dir, target_size = args_tuple
    results = []
    skipped_files = []
    
    for dicom_path in dicom_files:
        dicom_id = dicom_path.stem
        output_path = output_dir / f"{dicom_id}.png"
        
        # Skip if already processed
        if output_path.exists():
            results.append(True)
            continue
            
        success = process_dicom_to_png(dicom_path, output_path, target_size)
        if not success:
            skipped_files.append(dicom_path.name)
        results.append(success)
    
    return results, skipped_files


def load_default_paths(config_path: Optional[Path] = None) -> tuple[Path, Path, Path]:
    """Load default paths from config file."""
    if config_path is None:
        config_path = ROOT / "configs" / "data_egd_cxr_single_label.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not HAS_CONFIG_LOADER:
        raise ImportError("ConfigLoader not available. Install egd_cxr_dataset package.")
    
    cfg = ConfigLoader(config_path)
    
    dicom_dir = Path(cfg.get("input_path", "dicom_raw"))
    png_dir = Path(cfg.get("input_path", "png_raw"))
    gaze_raw = Path(cfg.get("input_path", "gaze_raw"))
    master_sheet = gaze_raw / "master_sheet.csv"
    
    return dicom_dir, png_dir, master_sheet


def main():
    parser = argparse.ArgumentParser(description="Preprocess DICOM images to PNG")
    parser.add_argument("--config", type=Path, default=None,
                       help="Config file path (default: configs/data_egd_cxr_single_label.yaml)")
    parser.add_argument("--dicom-dir", type=Path, default=None, 
                       help="Directory containing DICOM files (uses config if not provided)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Directory to save PNG files (uses config if not provided)")
    parser.add_argument("--master-sheet", type=Path, default=None,
                       help="Master sheet CSV to filter cases (uses config if not provided)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for parallel processing")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224],
                       help="Target image size (width height)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually processing")
    
    args = parser.parse_args()
    
    # Load default paths from config if not provided
    try:
        default_dicom, default_png, default_master = load_default_paths(args.config)
        print(f"📋 Loaded default paths from config:")
        print(f"   DICOM: {default_dicom}")
        print(f"   PNG: {default_png}")
        print(f"   Master sheet: {default_master}")
    except Exception as e:
        print(f"⚠️  Could not load default paths: {e}")
        default_dicom = default_png = default_master = None
    
    # Use provided paths or defaults
    dicom_dir = args.dicom_dir or default_dicom
    output_dir = args.output_dir or default_png
    master_sheet = args.master_sheet or default_master
    
    if not dicom_dir:
        print("❌ DICOM directory not specified and no default available")
        return 1
    if not output_dir:
        print("❌ Output directory not specified and no default available")
        return 1
    
    if not HAS_PYDICOM:
        print("❌ pydicom is required but not installed")
        return 1
    
    # Setup
    dicom_dir = dicom_dir.expanduser()
    output_dir = output_dir.expanduser()
    target_size = tuple(args.target_size)
    
    if not dicom_dir.exists():
        print(f"❌ DICOM directory not found: {dicom_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find DICOM files
    dicom_files = []
    for ext in [".dcm", ".dicom"]:
        dicom_files.extend(dicom_dir.glob(f"*{ext}"))
    
    if not dicom_files:
        print(f"❌ No DICOM files found in {dicom_dir}")
        return 1
    
    print(f"📁 Found {len(dicom_files)} DICOM files")
    
    # Filter by master sheet if provided
    if master_sheet and master_sheet.exists():
        master_df = pd.read_csv(master_sheet)
        valid_ids = set(master_df["dicom_id"].tolist())
        dicom_files = [f for f in dicom_files if f.stem in valid_ids]
        print(f"📋 Filtered to {len(dicom_files)} files using master sheet")
    
    # Check existing PNG files
    existing_pngs = set(output_dir.glob("*.png"))
    existing_ids = {f.stem for f in existing_pngs}
    dicom_ids = {f.stem for f in dicom_files}
    
    to_process = [f for f in dicom_files if f.stem not in existing_ids]
    already_processed = len(dicom_ids.intersection(existing_ids))
    
    print(f"✅ Already processed: {already_processed}")
    print(f"🔄 To process: {len(to_process)}")
    
    if args.dry_run:
        print("🔍 Dry run - would process:")
        for f in to_process[:10]:  # Show first 10
            print(f"  {f.name} -> {output_dir / f.stem}.png")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return 0
    
    if not to_process:
        print("✅ All files already processed!")
        return 0
    
    # Process in batches
    num_workers = args.num_workers or min(mp.cpu_count(), 8)
    batch_size = args.batch_size
    
    # Create batches
    batches = []
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        batches.append((batch, output_dir, target_size))
    
    print(f"🚀 Processing {len(to_process)} files using {num_workers} workers")
    print(f"📦 {len(batches)} batches of size {batch_size}")
    
    start_time = time.time()
    
    # Process batches in parallel
    with mp.Pool(num_workers) as pool:
        batch_outputs = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Flatten results and collect skipped files
    all_results = []
    all_skipped = []
    for batch_results, batch_skipped in batch_outputs:
        all_results.extend(batch_results)
        all_skipped.extend(batch_skipped)
    
    # Summary
    successful = sum(all_results)
    failed = len(all_results) - successful
    skipped = len(all_skipped)
    elapsed = time.time() - start_time
    
    print(f"\n📊 Processing Complete!")
    print(f"✅ Successful: {successful}")
    print(f"⚠️  Skipped (corrupted): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"🚀 Speed: {len(to_process)/elapsed:.1f} files/sec")
    
    if all_skipped:
        print(f"\n⚠️  Skipped files (corrupted DICOM):")
        for skipped_file in all_skipped[:10]:  # Show first 10
            print(f"   {skipped_file}")
        if len(all_skipped) > 10:
            print(f"   ... and {len(all_skipped) - 10} more")
    
    # Estimate speedup
    estimated_dicom_time = len(to_process) * 0.075  # ~75ms per DICOM
    estimated_png_time = len(to_process) * 0.007   # ~7ms per PNG
    speedup = estimated_dicom_time / estimated_png_time
    
    print(f"\n🎯 Expected Training Speedup:")
    print(f"   DICOM loading: ~{estimated_dicom_time:.1f}s")
    print(f"   PNG loading: ~{estimated_png_time:.1f}s")
    print(f"   Speedup: {speedup:.1f}x faster")
    
    print(f"\n✅ PNG files saved to: {output_dir}")
    print(f"🚀 You can now use fast training:")
    print(f"   python main_resnet_img_fast.py --config configs/data_egd_cxr_single_label.yaml")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
