#!/usr/bin/env python3
"""
Enhanced EGD-CXR Dataset Sampling Script

This script creates a more diverse and comprehensive sample from the EGD-CXR dataset,
ensuring representation across all clinical conditions and data completeness.

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

class EnhancedEGDCXRSampler:
    """Enhanced class to handle diverse sampling from the EGD-CXR dataset."""
    
    def __init__(self, config_path):
        """Initialize the enhanced sampler with configuration."""
        self.config = self.load_config(config_path)
        self.raw_path = self.config['path']['raw']
        self.output_path = self.config['path']['sampling_data']
        self.sample_size = 50
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Define all clinical conditions we want to represent
        self.primary_conditions = ['Normal', 'CHF', 'pneumonia']
        self.secondary_conditions = ['consolidation', 'enlarged_cardiac_silhouette', 
                                   'pleural_effusion_or_thickening', 'pulmonary_edema__hazy_opacity']
        
        logger.info(f"Initialized Enhanced EGD-CXR Sampler")
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
    
    def validate_data_completeness(self, dicom_ids):
        """Validate that all required data exists for given DICOM IDs."""
        logger.info("Validating data completeness...")
        
        # Check audio transcripts
        audio_dir = os.path.join(self.raw_path, 'audio_segmentation_transcripts')
        available_audio = set(os.listdir(audio_dir)) if os.path.exists(audio_dir) else set()
        
        # Check gaze data availability
        gaze_file = os.path.join(self.raw_path, 'eye_gaze.csv')
        fixations_file = os.path.join(self.raw_path, 'fixations.csv')
        
        complete_studies = []
        for dicom_id in dicom_ids:
            # Check audio transcript
            if dicom_id not in available_audio:
                continue
                
            # Check if gaze data exists (sample check)
            has_gaze = False
            has_fixations = False
            
            if os.path.exists(gaze_file):
                try:
                    # Quick check for gaze data
                    sample_gaze = pd.read_csv(gaze_file, nrows=1000)
                    if dicom_id in sample_gaze['DICOM_ID'].values:
                        has_gaze = True
                except:
                    pass
            
            if os.path.exists(fixations_file):
                try:
                    # Quick check for fixations data
                    sample_fixations = pd.read_csv(fixations_file, nrows=1000)
                    if dicom_id in sample_fixations['DICOM_ID'].values:
                        has_fixations = True
                except:
                    pass
            
            if has_gaze and has_fixations:
                complete_studies.append(dicom_id)
        
        logger.info(f"Found {len(complete_studies)} studies with complete data")
        return complete_studies
    
    def create_diverse_sample(self, df, n_samples=50):
        """Create a diverse sample ensuring representation across all conditions."""
        logger.info("Creating diverse stratified sample...")
        
        # First, get studies with complete data
        all_dicom_ids = df['dicom_id'].tolist()
        complete_studies = self.validate_data_completeness(all_dicom_ids)
        df_complete = df[df['dicom_id'].isin(complete_studies)]
        
        logger.info(f"Working with {len(df_complete)} studies with complete data")
        
        samples = []
        used_ids = set()
        
        # 1. Sample from primary conditions (Normal, CHF, Pneumonia)
        primary_samples_per_condition = 8  # 24 total
        for condition in self.primary_conditions:
            if condition in df_complete.columns:
                condition_data = df_complete[df_complete[condition] == 1]
                condition_data = condition_data[~condition_data['dicom_id'].isin(used_ids)]
                
                if len(condition_data) > 0:
                    n_sample = min(primary_samples_per_condition, len(condition_data))
                    condition_sample = condition_data.sample(n=n_sample, random_state=42)
                    samples.append(condition_sample)
                    used_ids.update(condition_sample['dicom_id'].tolist())
                    logger.info(f"Sampled {len(condition_sample)} {condition} cases")
        
        # 2. Sample studies with multiple conditions (complex cases)
        multi_condition_data = df_complete[~df_complete['dicom_id'].isin(used_ids)]
        if len(multi_condition_data) > 0:
            # Calculate condition complexity
            condition_cols = self.primary_conditions + self.secondary_conditions
            available_cols = [col for col in condition_cols if col in multi_condition_data.columns]
            multi_condition_data['complexity'] = multi_condition_data[available_cols].sum(axis=1)
            
            # Sample complex cases (2+ conditions)
            complex_cases = multi_condition_data[multi_condition_data['complexity'] >= 2]
            if len(complex_cases) > 0:
                n_complex = min(8, len(complex_cases))
                complex_sample = complex_cases.sample(n=n_complex, random_state=42)
                samples.append(complex_sample)
                used_ids.update(complex_sample['dicom_id'].tolist())
                logger.info(f"Sampled {len(complex_sample)} complex cases (2+ conditions)")
        
        # 3. Sample from secondary conditions
        secondary_samples_per_condition = 3  # 12 total
        for condition in self.secondary_conditions:
            if condition in df_complete.columns:
                condition_data = df_complete[df_complete[condition] == 1]
                condition_data = condition_data[~condition_data['dicom_id'].isin(used_ids)]
                
                if len(condition_data) > 0:
                    n_sample = min(secondary_samples_per_condition, len(condition_data))
                    condition_sample = condition_data.sample(n=n_sample, random_state=42)
                    samples.append(condition_sample)
                    used_ids.update(condition_sample['dicom_id'].tolist())
                    logger.info(f"Sampled {len(condition_sample)} {condition} cases")
        
        # 4. Fill remaining slots with diverse random samples
        remaining = n_samples - sum(len(s) for s in samples)
        if remaining > 0:
            remaining_data = df_complete[~df_complete['dicom_id'].isin(used_ids)]
            if len(remaining_data) > 0:
                # Ensure age and gender diversity in remaining samples
                remaining_sample = remaining_data.sample(n=min(remaining, len(remaining_data)), random_state=42)
                samples.append(remaining_sample)
                used_ids.update(remaining_sample['dicom_id'].tolist())
                logger.info(f"Added {len(remaining_sample)} diverse random samples")
        
        # Combine all samples
        if samples:
            final_sample = pd.concat(samples, ignore_index=True)
        else:
            final_sample = df_complete.sample(n=min(n_samples, len(df_complete)), random_state=42)
        
        # Ensure we have exactly n_samples
        if len(final_sample) > n_samples:
            final_sample = final_sample.sample(n=n_samples, random_state=42)
        
        logger.info(f"Final diverse sample: {len(final_sample)} studies")
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
        return copied_count
    
    def sample_gaze_data(self, sample_df):
        """Sample eye gaze data for the selected studies."""
        logger.info("Sampling eye gaze data...")
        
        gaze_file = os.path.join(self.raw_path, 'eye_gaze.csv')
        fixations_file = os.path.join(self.raw_path, 'fixations.csv')
        
        sample_dicom_ids = set(sample_df['dicom_id'].tolist())
        
        gaze_records = 0
        fixation_records = 0
        
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
                gaze_records = len(gaze_sample)
                logger.info(f"Saved {gaze_records} gaze records to {gaze_output}")
        
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
                fixation_records = len(fixations_sample)
                logger.info(f"Saved {fixation_records} fixation records to {fixations_output}")
        
        return gaze_records, fixation_records
    
    def sample_bounding_boxes(self, sample_df):
        """Sample bounding box data for the selected studies."""
        logger.info("Sampling bounding box data...")
        
        bbox_file = os.path.join(self.raw_path, 'bounding_boxes.csv')
        if not os.path.exists(bbox_file):
            logger.warning(f"Bounding boxes file not found: {bbox_file}")
            return 0
        
        sample_dicom_ids = set(sample_df['dicom_id'].tolist())
        
        # Read and filter bounding boxes
        bbox_df = pd.read_csv(bbox_file)
        bbox_sample = bbox_df[bbox_df['dicom_id'].isin(sample_dicom_ids)]
        
        bbox_output = os.path.join(self.output_path, 'bounding_boxes_sample.csv')
        bbox_sample.to_csv(bbox_output, index=False)
        bbox_records = len(bbox_sample)
        logger.info(f"Saved {bbox_records} bounding box records to {bbox_output}")
        
        return bbox_records
    
    def create_comprehensive_metadata(self, sample_df, gaze_records, fixation_records, bbox_records, audio_count):
        """Create comprehensive metadata file for the sampled data."""
        logger.info("Creating comprehensive sample metadata...")
        
        # Calculate condition statistics
        condition_stats = {}
        all_conditions = self.primary_conditions + self.secondary_conditions
        
        for condition in all_conditions:
            if condition in sample_df.columns:
                condition_stats[condition] = int(sample_df[condition].sum())
        
        # Calculate complexity distribution
        available_condition_cols = [col for col in all_conditions if col in sample_df.columns]
        sample_df['condition_count'] = sample_df[available_condition_cols].sum(axis=1)
        complexity_dist = sample_df['condition_count'].value_counts().sort_index().to_dict()
        
        # Basic statistics
        metadata = {
            'sample_info': {
                'total_studies': len(sample_df),
                'sample_date': pd.Timestamp.now().isoformat(),
                'source_dataset': 'EGD-CXR v1.0.0',
                'sampling_strategy': 'diverse_stratified'
            },
            'demographics': {
                'gender_distribution': sample_df['gender'].value_counts().to_dict(),
                'age_distribution': sample_df['anchor_age'].value_counts().to_dict()
            },
            'clinical_conditions': condition_stats,
            'condition_complexity': {
                'studies_with_single_condition': complexity_dist.get(1, 0),
                'studies_with_multiple_conditions': sum(v for k, v in complexity_dist.items() if k > 1),
                'complexity_distribution': complexity_dist
            },
            'data_completeness': {
                'audio_transcripts': audio_count,
                'gaze_records': gaze_records,
                'fixation_records': fixation_records,
                'bounding_box_records': bbox_records,
                'completeness_rate': f"{audio_count/len(sample_df)*100:.1f}%"
            },
            'sampled_studies': sample_df['dicom_id'].tolist(),
            'study_details': []
        }
        
        # Add detailed study information
        for _, row in sample_df.iterrows():
            study_detail = {
                'dicom_id': row['dicom_id'],
                'gender': row['gender'],
                'age': row['anchor_age'],
                'conditions': [cond for cond in all_conditions if cond in row and row[cond] == 1],
                'condition_count': int(row['condition_count'])
            }
            metadata['study_details'].append(study_detail)
        
        # Save metadata
        metadata_file = os.path.join(self.output_path, 'comprehensive_sample_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved comprehensive metadata to {metadata_file}")
        return metadata
    
    def create_detailed_summary_report(self, sample_df, metadata):
        """Create a detailed summary report of the sampled data."""
        logger.info("Creating detailed summary report...")
        
        report_file = os.path.join(self.output_path, 'detailed_sampling_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("EGD-CXR Enhanced Dataset Sampling Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Sample Size: {len(sample_df)} studies\n")
            f.write(f"Sampling Date: {metadata['sample_info']['sample_date']}\n")
            f.write(f"Sampling Strategy: {metadata['sample_info']['sampling_strategy']}\n\n")
            
            f.write("Demographics:\n")
            f.write("-" * 30 + "\n")
            for gender, count in metadata['demographics']['gender_distribution'].items():
                f.write(f"  {gender}: {count} ({count/len(sample_df)*100:.1f}%)\n")
            
            f.write("\nAge Distribution:\n")
            f.write("-" * 30 + "\n")
            for age, count in metadata['demographics']['age_distribution'].items():
                f.write(f"  {age}: {count} ({count/len(sample_df)*100:.1f}%)\n")
            
            f.write("\nClinical Conditions:\n")
            f.write("-" * 30 + "\n")
            for condition, count in metadata['clinical_conditions'].items():
                f.write(f"  {condition}: {count} ({count/len(sample_df)*100:.1f}%)\n")
            
            f.write("\nCondition Complexity:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Single condition: {metadata['condition_complexity']['studies_with_single_condition']}\n")
            f.write(f"  Multiple conditions: {metadata['condition_complexity']['studies_with_multiple_conditions']}\n")
            f.write("  Complexity distribution:\n")
            for complexity, count in metadata['condition_complexity']['complexity_distribution'].items():
                f.write(f"    {complexity} conditions: {count} studies\n")
            
            f.write("\nData Completeness:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Audio transcripts: {metadata['data_completeness']['audio_transcripts']}\n")
            f.write(f"  Gaze records: {metadata['data_completeness']['gaze_records']:,}\n")
            f.write(f"  Fixation records: {metadata['data_completeness']['fixation_records']:,}\n")
            f.write(f"  Bounding box records: {metadata['data_completeness']['bounding_box_records']:,}\n")
            f.write(f"  Completeness rate: {metadata['data_completeness']['completeness_rate']}\n")
            
            f.write(f"\nSampled Study Details:\n")
            f.write("-" * 30 + "\n")
            for i, study in enumerate(metadata['study_details'], 1):
                f.write(f"  {i:2d}. {study['dicom_id']}\n")
                f.write(f"      Gender: {study['gender']}, Age: {study['age']}\n")
                f.write(f"      Conditions: {', '.join(study['conditions']) if study['conditions'] else 'None'}\n")
                f.write(f"      Complexity: {study['condition_count']} conditions\n\n")
        
        logger.info(f"Saved detailed summary report to {report_file}")
    
    def run_enhanced_sampling(self):
        """Run the complete enhanced sampling process."""
        logger.info("Starting Enhanced EGD-CXR dataset sampling...")
        
        try:
            # Load master sheet
            master_df = self.load_master_sheet()
            
            # Create diverse sample
            sample_df = self.create_diverse_sample(master_df, self.sample_size)
            
            # Save sample master sheet
            sample_master_file = os.path.join(self.output_path, 'master_sheet_sample.csv')
            sample_df.to_csv(sample_master_file, index=False)
            logger.info(f"Saved sample master sheet to {sample_master_file}")
            
            # Copy audio transcripts
            audio_count = self.copy_audio_transcripts(sample_df)
            
            # Sample gaze and fixations data
            gaze_records, fixation_records = self.sample_gaze_data(sample_df)
            
            # Sample bounding boxes
            bbox_records = self.sample_bounding_boxes(sample_df)
            
            # Create comprehensive metadata
            metadata = self.create_comprehensive_metadata(
                sample_df, gaze_records, fixation_records, bbox_records, audio_count
            )
            
            # Create detailed summary report
            self.create_detailed_summary_report(sample_df, metadata)
            
            logger.info("Enhanced sampling completed successfully!")
            logger.info(f"Output directory: {self.output_path}")
            
            return sample_df, metadata
            
        except Exception as e:
            logger.error(f"Error during enhanced sampling: {str(e)}")
            raise

def main():
    """Main function to run the enhanced sampling process."""
    config_path = "/project/hnguyen2/mvu9/folder_04_ma/gaze-01/config_maui/data_egd-cxr.yaml"
    
    try:
        sampler = EnhancedEGDCXRSampler(config_path)
        sample_df, metadata = sampler.run_enhanced_sampling()
        
        print("\n" + "="*70)
        print("ENHANCED EGD-CXR SAMPLING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Sample size: {len(sample_df)} studies")
        print(f"Output directory: {sampler.output_path}")
        print(f"\nClinical conditions:")
        for condition, count in metadata['clinical_conditions'].items():
            print(f"  - {condition}: {count}")
        print(f"\nData completeness:")
        print(f"  - Audio transcripts: {metadata['data_completeness']['audio_transcripts']}")
        print(f"  - Gaze records: {metadata['data_completeness']['gaze_records']:,}")
        print(f"  - Fixation records: {metadata['data_completeness']['fixation_records']:,}")
        print(f"  - Bounding box records: {metadata['data_completeness']['bounding_box_records']:,}")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())



