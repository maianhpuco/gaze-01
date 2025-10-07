#!/usr/bin/env python3
"""
EGD-CXR Dataset Exploration Script

This script explores the EGD-CXR (Eye Gaze Dataset for Chest X-Ray) dataset structure,
prints directory trees, and analyzes sample data to understand the dataset contents.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(item_path) and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item_path, next_prefix, max_depth, current_depth + 1)
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")

def explore_csv_file(file_path, max_rows=5):
    """Explore a CSV file and print basic statistics."""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Read first few rows to understand structure
        df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for analysis
        
        print(f"Shape (first 1000 rows): {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Show first few rows
        print(f"\nFirst {max_rows} rows:")
        print(df.head(max_rows).to_string())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics for numeric columns:")
            print(df[numeric_cols].describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\nNo missing values found in first 1000 rows")
            
        return df
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def explore_directory_structure(directory_path):
    """Explore directory structure and file types."""
    print(f"\n{'='*60}")
    print(f"DIRECTORY STRUCTURE: {directory_path}")
    print(f"{'='*60}")
    
    file_types = defaultdict(int)
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory_path):
        level = root.replace(directory_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Show first 10 files per directory
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file)[1]
            file_types[file_ext] += 1
            total_files += 1
            total_size += file_size
            
            size_str = f"({file_size / (1024*1024):.1f}MB)" if file_size > 1024*1024 else f"({file_size / 1024:.1f}KB)"
            print(f"{subindent}{file} {size_str}")
        
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")
    
    print(f"\nFile type summary:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext or 'no extension'}: {count} files")
    
    print(f"\nTotal: {total_files} files, {total_size / (1024*1024*1024):.2f} GB")

def analyze_gaze_data(gaze_file):
    """Analyze eye gaze data specifically."""
    print(f"\n{'='*60}")
    print(f"EYE GAZE DATA ANALYSIS")
    print(f"{'='*60}")
    
    try:
        # Read a sample of the gaze data
        df = pd.read_csv(gaze_file, nrows=10000)
        
        print(f"Gaze data sample shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Look for common gaze data columns
        gaze_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['gaze', 'eye', 'x', 'y', 'timestamp', 'time'])]
        print(f"Potential gaze-related columns: {gaze_columns}")
        
        # Show sample data
        print(f"\nSample gaze data:")
        print(df[gaze_columns].head(10).to_string())
        
        # Basic statistics
        if len(gaze_columns) > 0:
            print(f"\nGaze data statistics:")
            print(df[gaze_columns].describe())
        
        return df
        
    except Exception as e:
        print(f"Error analyzing gaze data: {str(e)}")
        return None

def analyze_fixations_data(fixations_file):
    """Analyze fixations data specifically."""
    print(f"\n{'='*60}")
    print(f"FIXATIONS DATA ANALYSIS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(fixations_file, nrows=5000)
        
        print(f"Fixations data sample shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Look for fixation-related columns
        fixation_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['fixation', 'duration', 'x', 'y', 'timestamp'])]
        print(f"Potential fixation-related columns: {fixation_columns}")
        
        # Show sample data
        print(f"\nSample fixations data:")
        print(df[fixation_columns].head(10).to_string())
        
        return df
        
    except Exception as e:
        print(f"Error analyzing fixations data: {str(e)}")
        return None

def analyze_master_sheet(master_file):
    """Analyze the master sheet for dataset overview."""
    print(f"\n{'='*60}")
    print(f"MASTER SHEET ANALYSIS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(master_file)
        
        print(f"Master sheet shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        # Look for unique values in key columns
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"\nColumn '{col}': {unique_count} unique values")
            if unique_count <= 20:  # Show values if not too many
                print(f"  Values: {list(df[col].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing master sheet: {str(e)}")
        return None

def main():
    """Main exploration function."""
    print("EGD-CXR Dataset Exploration")
    print("=" * 60)
    
    # Load configuration
    config_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/config_maui/data_egd-cxr.yaml"
    config = load_config(config_path)
    
    dataset_path = config['path']['raw']
    print(f"Dataset path: {dataset_path}")
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return
    
    # Print directory tree
    print(f"\n{'='*60}")
    print("DIRECTORY TREE")
    print(f"{'='*60}")
    print_tree(dataset_path, max_depth=2)
    
    # Explore directory structure
    explore_directory_structure(dataset_path)
    
    # Explore main CSV files
    main_files = {
        'eye_gaze.csv': 'Eye gaze tracking data',
        'fixations.csv': 'Eye fixation data', 
        'master_sheet.csv': 'Master dataset sheet',
        'bounding_boxes.csv': 'Bounding box annotations'
    }
    
    for filename, description in main_files.items():
        filepath = os.path.join(dataset_path, filename)
        if os.path.exists(filepath):
            print(f"\n{description}:")
            explore_csv_file(filepath)
    
    # Special analysis for gaze and fixations data
    gaze_file = os.path.join(dataset_path, 'eye_gaze.csv')
    if os.path.exists(gaze_file):
        analyze_gaze_data(gaze_file)
    
    fixations_file = os.path.join(dataset_path, 'fixations.csv')
    if os.path.exists(fixations_file):
        analyze_fixations_data(fixations_file)
    
    master_file = os.path.join(dataset_path, 'master_sheet.csv')
    if os.path.exists(master_file):
        analyze_master_sheet(master_file)
    
    # Explore subdirectories
    subdirs = ['audio_segmentation_transcripts', 'inclusion_exclusion_criteria_outputs']
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.exists(subdir_path):
            print(f"\n{'='*60}")
            print(f"EXPLORING SUBDIRECTORY: {subdir}")
            print(f"{'='*60}")
            explore_directory_structure(subdir_path)
            
            # Show a few sample files from subdirectories
            files = os.listdir(subdir_path)[:3]
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if file.endswith('.csv'):
                    explore_csv_file(file_path, max_rows=3)
                elif file.endswith(('.txt', '.json')):
                    print(f"\nSample content from {file}:")
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read(500)  # First 500 characters
                            print(content)
                            if len(content) == 500:
                                print("... (truncated)")
                    except Exception as e:
                        print(f"Error reading {file}: {str(e)}")

if __name__ == "__main__":
    main()
