# 从edit_transfer数据集当中获取新数据

import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def process_images(data_dir):
    # Get all image files from the data directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(data_dir).glob(f'*{ext}')))
    
    print(f'Found {len(image_files)} images to process')
    
    # Process each image
    for img_path in image_files:
        # Get the image name without extension to use as folder name
        img_name = img_path.stem
        
        # Create output directories
        output_dir = os.path.join(data_dir, img_name)
        source_dir = os.path.join(output_dir, 'source')
        target_dir = os.path.join(output_dir, 'target')
        
        # Create directories if they don't exist
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Could not read image: {img_path}')
            continue
            
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Calculate the midpoints
        mid_h, mid_w = height // 2, width // 2
        
        # Split the image into four quadrants
        top_left = img[0:mid_h, 0:mid_w]
        top_right = img[0:mid_h, mid_w:width]
        bottom_left = img[mid_h:height, 0:mid_w]
        bottom_right = img[mid_h:height, mid_w:width]
        
        # Save the quadrants to their respective folders
        # Top-left to source folder as 7001.jpg
        cv2.imwrite(os.path.join(source_dir, '7001.jpg'), top_left)
        
        # Bottom-left to source folder as 7002.jpg
        cv2.imwrite(os.path.join(source_dir, '7002.jpg'), bottom_left)
        
        # Top-right to target folder as 7001.jpg
        cv2.imwrite(os.path.join(target_dir, '7001.jpg'), top_right)
        
        # Bottom-right to target folder as 7002.jpg
        cv2.imwrite(os.path.join(target_dir, '7002.jpg'), bottom_right)
        
        # Check for corresponding .txt file and copy it to the output directory
        txt_file_path = img_path.with_suffix('.txt')
        if txt_file_path.exists():
            # Copy the .txt file to the output directory (at the same level as source and target)
            shutil.copy(str(txt_file_path), os.path.join(output_dir, txt_file_path.name))
            print(f'Copied text file: {txt_file_path.name} to {output_dir}')
        
        print(f'Processed image: {img_name}')

if __name__ == '__main__':
    # Path to the data directory containing images
    data_directory = '/Users/cuikq/Downloads/Edit-Transfer-main/data/edit_transfer'
    
    # Process the images
    process_images(data_directory)
    
    print('Image processing completed!')