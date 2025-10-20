import langchain_openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

llm = langchain_openai.ChatOpenAI(
    model="gpt-5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=os.getenv("OPENAI_API_KEY")
)

def system_prompt(num_frames, initial_dicom_image, fixation_images_data):
    """
    Create system prompt for the VLM with initial DICOM image and fixation frames
    
    Args:
        num_frames (int): Number of fixation frames
        initial_dicom_image (str): Base64 encoded initial DICOM image
        fixation_images_data (list): List of base64 encoded fixation images
        
    Returns:
        dict: System prompt message for the VLM
    """
    content = [
        {
            "type": "text",
            "text": f"""You are an expert radiologist tasked to predict fixations frames of a dicom image.
The task fixations are between 0 and {num_frames}. We provide several examples of the robot performing the task at various stages and their corresponding task completion percentages.
Note that these frames are in random order, so please pay attention to the individual frames when reasoning about task completion percentage.
Initial dicom image:"""
        },
        {
            "type": "image",
            "source_type": "base64",
            "data": initial_dicom_image,
            "mime_type": "image/jpeg"
        },
        {
            "type": "text",
            "text": f"""
For each fixation frame, format your response as follows:
Frame i: 
- Frame Description: [Detailed description of the anatomical region or clinical feature being examined]
- Frame Number Prediction: [Position in the gaze sequence, e.g., "Frame 3 of 8"]
- Clinical Relevance: [Why this fixation is important for diagnosis]

Analyze the following fixation frames:
"""
        }
    ]
    
    # Add each fixation image to the content
    for i, image in enumerate(fixation_images_data):
        content.append({
            "type": "image",
            "source_type": "base64",
            "data": image,
            "mime_type": "image/jpeg"
        })
    
    return {
        "role": "user",
        "content": content
    }

def get_fixation_images_data(dicom_path, fixation_path):
    """
    Extract fixation frames from DICOM and fixation data, create visualizations, and convert to base64
    
    Args:
        dicom_path (str): Path to DICOM file
        fixation_path (str): Path to fixations CSV file
        
    Returns:
        list: List of base64 encoded images showing fixation visualizations
    """
    try:
        # Read DICOM file
        dicom_file = pydicom.dcmread(dicom_path)
        image_data = dicom_file.pixel_array
        img_height, img_width = image_data.shape
        
        # Read fixation data
        fixations_df = pd.read_csv(fixation_path)
        
        # Extract DICOM ID from the file path (assuming it's in the filename)
        dicom_filename = os.path.basename(dicom_path)
        dicom_id = dicom_filename.replace('.dcm', '')
        
        # Filter fixations for this DICOM ID
        dicom_fixations = fixations_df[fixations_df['DICOM_ID'] == dicom_id].copy()
        
        if len(dicom_fixations) == 0:
            print(f"No fixations found for DICOM ID: {dicom_id}")
            return []
        
        # Sort fixations by time to get proper sequence
        dicom_fixations = dicom_fixations.sort_values('Time (in secs)').reset_index(drop=True)
        
        base64_images = []
        
        # Create visualization for each fixation
        for idx, fixation in dicom_fixations.iterrows():
            # Convert normalized coordinates to pixel coordinates
            pixel_x = fixation['FPOGX'] * img_width
            pixel_y = fixation['FPOGY'] * img_height
            
            # Create the visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Display DICOM image
            ax.imshow(image_data, cmap='gray')
            
            # Highlight the fixation with a circle
            duration = fixation['FPOGD']
            circle_size = max(20, min(100, duration * 30))  # Scale based on duration
            
            # Draw the fixation circle
            circle = patches.Circle((pixel_x, pixel_y), circle_size/2, 
                                  color='red', alpha=0.6, 
                                  linewidth=4, edgecolor='yellow')
            ax.add_patch(circle)
            
            # Convert plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='PNG', dpi=150, bbox_inches='tight')
            buf.seek(0)
            base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
            base64_images.append(base64_string)
            
            plt.close()
        
        print(f"Created {len(base64_images)} fixation visualizations")
        return base64_images
        
    except Exception as e:
        print(f"Error creating fixation images: {e}")
        return []

def get_initial_dicom_image(dicom_path):
    """
    Convert DICOM image to base64 string
    
    Args:
        dicom_path (str): Path to DICOM file
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        # Read DICOM file
        dicom_file = pydicom.dcmread(dicom_path)
        image_data = dicom_file.pixel_array
        
        # Create a simple visualization of the DICOM image
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image_data, cmap='gray')
        ax.set_title('Initial DICOM Image', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', dpi=150, bbox_inches='tight')
        buf.seek(0)
        base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        plt.close()
        print(f"Created initial DICOM image base64 (length: {len(base64_string):,} chars)")
        return base64_string
        
    except Exception as e:
        print(f"Error creating initial DICOM image: {e}")
        return ""

def get_num_frames(dicom_path, fixation_path):
    """
    Get the number of fixation frames for a specific DICOM image
    
    Args:
        dicom_path (str): Path to DICOM file
        fixation_path (str): Path to fixations CSV file
        
    Returns:
        int: Number of fixation frames for the DICOM image
    """
    try:
        # Read fixation data
        fixations_df = pd.read_csv(fixation_path)
        
        # Extract DICOM ID from the file path
        dicom_filename = os.path.basename(dicom_path)
        dicom_id = dicom_filename.replace('.dcm', '')
        
        # Filter fixations for this DICOM ID
        dicom_fixations = fixations_df[fixations_df['DICOM_ID'] == dicom_id]
        
        num_frames = len(dicom_fixations)
        print(f"Found {num_frames} fixation frames for DICOM ID: {dicom_id}")
        return num_frames
        
    except Exception as e:
        print(f"Error getting number of frames: {e}")
        return 0

def start_asking():
    fixation_path = "/home/hiro/Downloads/Workspace/gz_lab/egd-cxr-1.0.0/1.0.0/fixations.csv"
    dicom_path = "/home/hiro/Downloads/Workspace/gz_lab/dicom_raw/0dda5129-7c030387-3c83d925-fae90635-d2f2c42f.dcm"
    num_frames = get_num_frames(dicom_path, fixation_path)
    initial_dicom_image = get_initial_dicom_image(dicom_path)
    print("Getting fixation images data")
    fixation_images_data = get_fixation_images_data(dicom_path, fixation_path)[:25]
    print("Starting to ask the LLM")
    system_prompt_msg = system_prompt(num_frames, initial_dicom_image, fixation_images_data)
    print("Formed system prompt message")
    response = llm.invoke([system_prompt_msg])
    print(response.text)
    return response.text

start_asking()