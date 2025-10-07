#!/usr/bin/env python3
"""
GPU Test Script for EGD-CXR Dataset Processing
This script tests GPU availability and creates a simple tensor operation.
"""

import sys
import os

def test_gpu_availability():
    """Test if GPU is available and working"""
    print("=== GPU Availability Test ===")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ Current GPU: {torch.cuda.current_device()}")
            print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test simple GPU operation
            print("\n=== Testing GPU Operations ===")
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print(f"✓ GPU tensor operation successful: {z.shape}")
            print(f"✓ Result tensor device: {z.device}")
            
            return True
        else:
            print("✗ CUDA not available")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main function"""
    print("Starting GPU test for EGD-CXR dataset processing...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        print("\n🎉 GPU session is ready for EGD-CXR dataset processing!")
        print("You can now run GPU-accelerated code for:")
        print("- Eye gaze data processing")
        print("- Image analysis")
        print("- Machine learning model training")
        print("- Data augmentation")
    else:
        print("\n⚠️  GPU not available. Running on CPU.")
    
    return gpu_available

if __name__ == "__main__":
    main()
