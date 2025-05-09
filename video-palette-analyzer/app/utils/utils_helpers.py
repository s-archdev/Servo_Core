"""
Helper functions for the Video Color Palette Analyzer.
"""

import os
import shutil
import datetime
import cv2
from typing import List, Tuple


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
    
    Returns:
        File extension (without the dot)
    """
    return os.path.splitext(filename)[1][1:].lower()


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a valid video file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file extension
    valid_extensions = ["mp4", "avi", "mov", "mkv", "webm", "flv"]
    extension = get_file_extension(file_path)
    if extension not in valid_extensions:
        return False, f"Invalid file extension: {extension}. Must be one of: {', '.join(valid_extensions)}"
    
    # Try to open the file with OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "Could not open video file with OpenCV"
        
        # Check if the file has frames
        ret, frame = cap.read()
        if not ret:
            return False, "Video file does not contain any frames"
        
        cap.release()
        
    except Exception as e:
        return False, f"Error validating video file: {str(e)}"
    
    return True, ""


def cleanup_temp_files(max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than the specified age.
    
    Args:
        max_age_hours: Maximum age of temporary files in hours
    
    Returns:
        Number of directories removed
    """
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        return 0
    
    count = 0
    current_time = datetime.datetime.now()
    
    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        # Get the modification time
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))
        age = current_time - mod_time
        
        # Remove if older than max_age_hours
        if age > datetime.timedelta(hours=max_age_hours):
            try:
                shutil.rmtree(item_path)
                count += 1
            except Exception:
                # Skip if can't remove
                pass
    
    return count
