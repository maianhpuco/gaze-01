#!/usr/bin/env python3
"""
Update train/test splits to match available PNG files.

This script checks which PNG files are actually available and updates
the split files to only include cases that have corresponding PNG files.
"""

import os
from pathlib import Path
from typing import List, Set

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


def load_default_paths() -> tuple[Path, Path]:
    """Load PNG directory and master sheet from config."""
    config_path = ROOT / "configs" / "data_egd_cxr_single_label.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not HAS_CONFIG_LOADER:
        raise ImportError("ConfigLoader not available. Install egd_cxr_dataset package.")
    
    cfg = ConfigLoader(config_path)
    
    png_dir = Path(cfg.get("input_path", "png_raw"))
    gaze_raw = Path(cfg.get("input_path", "gaze_raw"))
    master_sheet = gaze_raw / "master_sheet.csv"
    
    return png_dir, master_sheet


def get_available_png_ids(png_dir: Path) -> Set[str]:
    """Get set of dicom_ids that have corresponding PNG files."""
    png_files = list(png_dir.glob("*.png"))
    return {f.stem for f in png_files}


def load_split_ids(split_file: Path) -> List[str]:
    """Load case IDs from a split file."""
    if not split_file.exists():
        return []
    
    ids = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def save_split_ids(split_file: Path, ids: List[str]) -> None:
    """Save case IDs to a split file."""
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'w', encoding='utf-8') as f:
        for case_id in ids:
            f.write(f"{case_id}\n")


def main():
    print("🔄 Updating train/test splits to match available PNG files...")
    
    # Load paths
    try:
        png_dir, master_sheet = load_default_paths()
        print(f"📁 PNG directory: {png_dir}")
        print(f"📋 Master sheet: {master_sheet}")
    except Exception as e:
        print(f"❌ Error loading paths: {e}")
        return 1
    
    # Get available PNG files
    available_ids = get_available_png_ids(png_dir)
    print(f"📊 Found {len(available_ids)} PNG files")
    
    if len(available_ids) == 0:
        print("❌ No PNG files found!")
        return 1
    
    # Load current splits
    splits_dir = ROOT / "configs" / "splits"
    train_ids = load_split_ids(splits_dir / "train_ids.txt")
    val_ids = load_split_ids(splits_dir / "val_ids.txt")
    test_ids = load_split_ids(splits_dir / "test_ids.txt")
    
    print(f"📋 Current splits:")
    print(f"   Train: {len(train_ids)}")
    print(f"   Val: {len(val_ids)}")
    print(f"   Test: {len(test_ids)}")
    print(f"   Total: {len(train_ids) + len(val_ids) + len(test_ids)}")
    
    # Filter splits to only include available PNG files
    train_available = [id for id in train_ids if id in available_ids]
    val_available = [id for id in val_ids if id in available_ids]
    test_available = [id for id in test_ids if id in available_ids]
    
    # Count removed cases
    train_removed = len(train_ids) - len(train_available)
    val_removed = len(val_ids) - len(val_available)
    test_removed = len(test_ids) - len(test_available)
    total_removed = train_removed + val_removed + test_removed
    
    print(f"\n📊 After filtering to available PNG files:")
    print(f"   Train: {len(train_available)} (removed {train_removed})")
    print(f"   Val: {len(val_available)} (removed {val_removed})")
    print(f"   Test: {len(test_available)} (removed {test_removed})")
    print(f"   Total: {len(train_available) + len(val_available) + len(test_available)} (removed {total_removed})")
    
    if total_removed > 0:
        print(f"\n⚠️  Removed cases (no PNG file):")
        all_original = set(train_ids + val_ids + test_ids)
        missing_cases = all_original - available_ids
        for case_id in sorted(missing_cases)[:10]:  # Show first 10
            print(f"   {case_id}")
        if len(missing_cases) > 10:
            print(f"   ... and {len(missing_cases) - 10} more")
    
    # Save updated splits
    print(f"\n💾 Saving updated splits...")
    save_split_ids(splits_dir / "train_ids.txt", train_available)
    save_split_ids(splits_dir / "val_ids.txt", val_available)
    save_split_ids(splits_dir / "test_ids.txt", test_available)
    
    print(f"✅ Updated splits saved!")
    print(f"🎯 Ready for training with {len(train_available) + len(val_available) + len(test_available)} cases")
    
    return 0


if __name__ == "__main__":
    exit(main())
