#!/usr/bin/env python3
"""
EGD-CXR Dataset Sampling Script

This script samples 50 diverse examples from the EGD-CXR dataset and creates
a structured subset for processing and analysis.

Author: AI Assistant
Date: 2024
"""

import os
import yaml
import pandas as pd
import numpy as np
import shutil
import json
from pathlib import Path
import random
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EGDCXRSampler:
    """Class to handle sampling from the EGD-CXR dataset."""
    
    def __init__(self, config_path):
        """Initialize the sampler with configuration."""
        self.config = self.load_config(config_path)
        self.raw_path = self.config['path']['raw']
        self.output_path = self.config['path']['sampling_data']
        self.sample_size = 50
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        logger.info(f"Initialized EGD-CXR Sampler")
        logger.info(f"Raw data path: {self.raw_path}")
        logger.info(f"Output path: {self.output_path}")
    
    def load_config(self, config_path):
        """Load dataset configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_master_sheet(self):
        """Load the master sheet with all study metadata."""
        master_file = os.path.join(self.raw_path, 'master_sheet.csv')
        if not os.path.exists(master_file):
            raise FileNotFoundError(f"Master sheet not found: {master_file}")
        
        df = pd.read_csv(master_file)
        logger.info(f"Loaded master sheet: {df.shape[0]} studies, {df.shape[1]} columns")
        return df
    
    def stratify_samples(self, df, n_samples=50):
        """Create stratified sample ensuring diversity across conditions."""
        logger.info("Creating stratified sample...")
        
        # Define stratification groups
        conditions = ['Normal', 'CHF', 'pneumonia']
        age_groups = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60 - 70', '70 - 80', '> 80']
        genders = ['M', 'F']
        
        samples = []
        
        # Sample from each condition
        for condition in conditions:
            if condition in df.columns:
                condition_data = df[df[condition] == 1]
                if len(condition_data) > 0:
                    # Sample proportionally, but ensure minimum representation
                    n_condition = max(5, min(len(condition_data), n_samples // len(conditions)))
                    condition_sample = condition_data.sample(n=n_condition, random_state=42)
                    samples.append(condition_sample)
                    logger.info(f"Sampled {len(condition_sample)} {condition} cases")
        
        # If we don't have enough samples, fill with random selection
        if len(samples) < n_samples:
            remaining = n_samples - sum(len(s) for s in samples)
            used_ids = set()
            for sample_df in samples:
                used_ids.update(sample_df['dicom_id'].tolist())
            
            remaining_data = df[~df['dicom_id'].isin(used_ids)]
            if len(remaining_data) > 0:
                additional_sample = remaining_data.sample(n=min(remaining, len(remaining_data)), random_state=42)
                samples.append(additional_sample)
                logger.info(f"Added {len(additional_sample)} additional random samples")
        
        # Combine all samples
        final_sample = pd.concat(samples, ignore_index=True)
        
        # Ensure we have exactly n_samples
        if len(final_sample) > n_samples:
            final_sample = final_sample.sample(n=n_samples, random_state=42)
        
        logger.info(f"Final stratified sample: {len(final_sample)} studies")
        return final_sample
    
    def copy_audio_transcripts(self, sample_df):
        """Copy audio transcript files for sampled studies."""
        logger.info("Copying audio transcript files...")
        
        audio_dir = os.path.join(self.output_path, 'audio_segmentation_transcripts')
        os.makedirs(audio_dir, exist_ok=True)
        
        source_audio_dir = os.path.join(self.raw_path, 'audio_segmentation_transcripts')
        
        copied_count = 0
        for _, row in sample_df.iterrows():
            dicom_id = row['dicom_id']
            source_dir = os.path.join(source_audio_dir, dicom_id)
            target_dir = os.path.join(audio_dir, dicom_id)
            
            if os.path.exists(source_dir):
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                copied_count += 1
            else:
                logger.warning(f"Audio transcript directory not found: {source_dir}")
        
        logger.info(f"Copied {copied_count} audio transcript directories")
    
    def sample_gaze_data(self, sample_df):
        """Sample eye gaze data for the selected studies."""
        logger.info("Sampling eye gaze data...")
        
        gaze_file = os.path.join(self.raw_path, 'eye_gaze.csv')
        fixations_file = os.path.join(self.raw_path, 'fixations.csv')
        
        sample_dicom_ids = set(sample_df['dicom_id'].tolist())
        
        # Sample gaze data
        if os.path.exists(gaze_file):
            logger.info("Processing eye gaze data...")
            gaze_chunks = []
            chunk_size = 10000
            
            for chunk in pd.read_csv(gaze_file, chunksize=chunk_size):
                chunk_filtered = chunk[chunk['DICOM_ID'].isin(sample_dicom_ids)]
                if len(chunk_filtered) > 0:
                    gaze_chunks.append(chunk_filtered)
            
            if gaze_chunks:
                gaze_sample = pd.concat(gaze_chunks, ignore_index=True)
                gaze_output = os.path.join(self.output_path, 'eye_gaze_sample.csv')
                gaze_sample.to_csv(gaze_output, index=False)
                logger.info(f"Saved {len(gaze_sample)} gaze records to {gaze_output}")
        
        # Sample fixations data
        if os.path.exists(fixations_file):
            logger.info("Processing fixations data...")
            fixations_chunks = []
            
            for chunk in pd.read_csv(fixations_file, chunksize=chunk_size):
                chunk_filtered = chunk[chunk['DICOM_ID'].isin(sample_dicom_ids)]
                if len(chunk_filtered) > 0:
                    fixations_chunks.append(chunk_filtered)
            
            if fixations_chunks:
                fixations_sample = pd.concat(fixations_chunks, ignore_index=True)
                fixations_output = os.path.join(self.output_path, 'fixations_sample.csv')
                fixations_sample.to_csv(fixations_output, index=False)
                logger.info(f"Saved {len(fixations_sample)} fixation records to {fixations_output}")
    
    def sample_bounding_boxes(self, sample_df):
        """Sample bounding box data for the selected studies."""
        logger.info("Sampling bounding box data...")
        
        bbox_file = os.path.join(self.raw_path, 'bounding_boxes.csv')
        if not os.path.exists(bbox_file):
            logger.warning(f"Bounding boxes file not found: {bbox_file}")
            return
        
        sample_dicom_ids = set(sample_df['dicom_id'].tolist())
        
        # Read and filter bounding boxes
        bbox_df = pd.read_csv(bbox_file)
        bbox_sample = bbox_df[bbox_df['dicom_id'].isin(sample_dicom_ids)]
        
        bbox_output = os.path.join(self.output_path, 'bounding_boxes_sample.csv')
        bbox_sample.to_csv(bbox_output, index=False)
        logger.info(f"Saved {len(bbox_sample)} bounding box records to {bbox_output}")
    
    def create_sample_metadata(self, sample_df):
        """Create metadata file for the sampled data."""
        logger.info("Creating sample metadata...")
        
        # Basic statistics
        metadata = {
            'sample_info': {
                'total_studies': len(sample_df),
                'sample_date': pd.Timestamp.now().isoformat(),
                'source_dataset': 'EGD-CXR v1.0.0'
            },
            'demographics': {
                'gender_distribution': sample_df['gender'].value_counts().to_dict(),
                'age_distribution': sample_df['anchor_age'].value_counts().to_dict()
            },
            'clinical_conditions': {
                'normal_cases': int(sample_df['Normal'].sum()) if 'Normal' in sample_df.columns else 0,
                'chf_cases': int(sample_df['CHF'].sum()) if 'CHF' in sample_df.columns else 0,
                'pneumonia_cases': int(sample_df['pneumonia'].sum()) if 'pneumonia' in sample_df.columns else 0
            },
            'sampled_studies': sample_df['dicom_id'].tolist()
        }
        
        # Save metadata
        metadata_file = os.path.join(self.output_path, 'sample_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
        return metadata
    
    def create_summary_report(self, sample_df, metadata):
        """Create a summary report of the sampled data."""
        logger.info("Creating summary report...")
        
        report_file = os.path.join(self.output_path, 'sampling_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("EGD-CXR Dataset Sampling Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Sample Size: {len(sample_df)} studies\n")
            f.write(f"Sampling Date: {metadata['sample_info']['sample_date']}\n\n")
            
            f.write("Demographics:\n")
            f.write("-" * 20 + "\n")
            for gender, count in metadata['demographics']['gender_distribution'].items():
                f.write(f"  {gender}: {count} ({count/len(sample_df)*100:.1f}%)\n")
            
            f.write("\nAge Distribution:\n")
            f.write("-" * 20 + "\n")
            for age, count in metadata['demographics']['age_distribution'].items():
                f.write(f"  {age}: {count} ({count/len(sample_df)*100:.1f}%)\n")
            
            f.write("\nClinical Conditions:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Normal: {metadata['clinical_conditions']['normal_cases']}\n")
            f.write(f"  CHF: {metadata['clinical_conditions']['chf_cases']}\n")
            f.write(f"  Pneumonia: {metadata['clinical_conditions']['pneumonia_cases']}\n")
            
            f.write(f"\nSampled Study IDs:\n")
            f.write("-" * 20 + "\n")
            for i, study_id in enumerate(metadata['sampled_studies'], 1):
                f.write(f"  {i:2d}. {study_id}\n")
        
        logger.info(f"Saved summary report to {report_file}")
    
    def run_sampling(self):
        """Run the complete sampling process."""
        logger.info("Starting EGD-CXR dataset sampling...")
        
        try:
            # Load master sheet
            master_df = self.load_master_sheet()
            
            # Create stratified sample
            sample_df = self.stratify_samples(master_df, self.sample_size)
            
            # Save sample master sheet
            sample_master_file = os.path.join(self.output_path, 'master_sheet_sample.csv')
            sample_df.to_csv(sample_master_file, index=False)
            logger.info(f"Saved sample master sheet to {sample_master_file}")
            
            # Copy audio transcripts
            self.copy_audio_transcripts(sample_df)
            
            # Sample gaze and fixations data
            self.sample_gaze_data(sample_df)
            
            # Sample bounding boxes
            self.sample_bounding_boxes(sample_df)
            
            # Create metadata
            metadata = self.create_sample_metadata(sample_df)
            
            # Create summary report
            self.create_summary_report(sample_df, metadata)
            
            logger.info("Sampling completed successfully!")
            logger.info(f"Output directory: {self.output_path}")
            
            return sample_df, metadata
            
        except Exception as e:
            logger.error(f"Error during sampling: {str(e)}")
            raise

def main():
    """Main function to run the sampling process."""
    config_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/config_maui/data_egd-cxr.yaml"
    
    try:
        sampler = EGDCXRSampler(config_path)
        sample_df, metadata = sampler.run_sampling()
        
        print("\n" + "="*60)
        print("EGD-CXR SAMPLING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Sample size: {len(sample_df)} studies")
        print(f"Output directory: {sampler.output_path}")
        print(f"Clinical conditions:")
        print(f"  - Normal: {metadata['clinical_conditions']['normal_cases']}")
        print(f"  - CHF: {metadata['clinical_conditions']['chf_cases']}")
        print(f"  - Pneumonia: {metadata['clinical_conditions']['pneumonia_cases']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())



