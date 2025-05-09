#!/usr/bin/env python3
"""
Video Color Palette Analyzer

This script analyzes the color palette of a video file by extracting the dominant 
color at a defined frame sampling rate.

Usage:
    python video_color_palette.py --input video.mp4 --output palette.png 
                                 [--method kmeans] [--sample-rate 1] 
                                 [--block-size 50] [--metadata palette_data.json]
"""

import argparse
import cv2
import json
import numpy as np
import os
from datetime import timedelta
from typing import Dict, List, Tuple, Union, Optional
from sklearn.cluster import KMeans
from PIL import Image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract color palette from a video')
    
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', required=True, help='Output palette image path')
    parser.add_argument('--method', '-m', choices=['kmeans', 'histogram'], default='kmeans',
                        help='Method for dominant color extraction (default: kmeans)')
    parser.add_argument('--sample-rate', '-s', type=float, default=1.0,
                        help='Sample rate in seconds (default: 1.0)')
    parser.add_argument('--sample-frames', '-f', type=int,
                        help='Alternative: sample every N frames')
    parser.add_argument('--block-size', '-b', type=int, default=50,
                        help='Size of color blocks in output image (default: 50)')
    parser.add_argument('--metadata', help='Output JSON metadata file path')
    
    return parser.parse_args()


def extract_frames(
    video_path: str,
    sample_rate: Optional[float] = None,
    sample_frames: Optional[int] = None
) -> List[Tuple[np.ndarray, float]]:
    """
    Extract frames from a video file at the specified sampling rate.
    
    Args:
        video_path: Path to the video file
        sample_rate: Time in seconds between sampled frames
        sample_frames: Number of frames to skip between samples
    
    Returns:
        List of tuples containing (frame, timestamp)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame interval
    if sample_rate is not None:
        frame_interval = int(fps * sample_rate)
    elif sample_frames is not None:
        frame_interval = sample_frames
    else:
        frame_interval = int(fps)  # Default: sample once per second
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            frames.append((frame, timestamp))
            
        frame_number += 1
    
    cap.release()
    
    return frames


def get_dominant_color_kmeans(frame: np.ndarray, n_clusters: int = 1) -> np.ndarray:
    """
    Extract dominant color using K-means clustering.
    
    Args:
        frame: Input frame
        n_clusters: Number of clusters (colors) to extract
    
    Returns:
        RGB color as numpy array
    """
    # Reshape the frame to a list of pixels
    pixels = frame.reshape(-1, 3)
    
    # Convert from BGR to RGB
    pixels = pixels[:, ::-1]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_
    
    # Get the most dominant color (largest cluster)
    counts = np.bincount(kmeans.labels_)
    dominant_color = colors[np.argmax(counts)]
    
    return dominant_color.astype(np.uint8)


def get_dominant_color_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Extract dominant color using histogram peak detection.
    
    Args:
        frame: Input frame
        bins: Number of bins for the histogram
    
    Returns:
        RGB color as numpy array
    """
    # Convert from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create histograms for each channel
    hist_r = cv2.calcHist([frame_rgb], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([frame_rgb], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([frame_rgb], [2], None, [bins], [0, 256])
    
    # Find the peak values
    r_peak = np.argmax(hist_r) * (256 // bins) + (256 // bins) // 2
    g_peak = np.argmax(hist_g) * (256 // bins) + (256 // bins) // 2
    b_peak = np.argmax(hist_b) * (256 // bins) + (256 // bins) // 2
    
    return np.array([r_peak, g_peak, b_peak], dtype=np.uint8)


def analyze_colors(
    frames: List[Tuple[np.ndarray, float]],
    method: str = 'kmeans'
) -> List[Tuple[np.ndarray, float]]:
    """
    Analyze each frame to extract the dominant color.
    
    Args:
        frames: List of tuples containing (frame, timestamp)
        method: Method for color extraction ('kmeans' or 'histogram')
    
    Returns:
        List of tuples containing (dominant_color, timestamp)
    """
    color_data = []
    
    for frame, timestamp in frames:
        if method == 'kmeans':
            dominant_color = get_dominant_color_kmeans(frame)
        else:  # histogram
            dominant_color = get_dominant_color_histogram(frame)
            
        color_data.append((dominant_color, timestamp))
        
    return color_data


def create_color_palette(
    color_data: List[Tuple[np.ndarray, float]],
    block_size: int = 50
) -> Image.Image:
    """
    Create a color palette image from the dominant colors.
    
    Args:
        color_data: List of tuples containing (dominant_color, timestamp)
        block_size: Size of each color block in pixels
    
    Returns:
        PIL Image object containing the color palette
    """
    num_colors = len(color_data)
    
    # Create a new image with the appropriate size
    palette_img = Image.new('RGB', (num_colors * block_size, block_size))
    
    # Fill the image with color blocks
    for i, (color, _) in enumerate(color_data):
        color_block = Image.new('RGB', (block_size, block_size), tuple(color))
        palette_img.paste(color_block, (i * block_size, 0))
    
    return palette_img


def generate_metadata(
    color_data: List[Tuple[np.ndarray, float]],
    input_file: str
) -> Dict:
    """
    Generate metadata for the color palette.
    
    Args:
        color_data: List of tuples containing (dominant_color, timestamp)
        input_file: Path to the input video file
    
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "source_video": os.path.basename(input_file),
        "analysis_date": str(np.datetime64('now')),
        "total_frames_analyzed": len(color_data),
        "colors": []
    }
    
    for color, timestamp in color_data:
        # Format timestamp as HH:MM:SS.ms
        time_formatted = str(timedelta(seconds=timestamp))
        
        metadata["colors"].append({
            "timestamp": timestamp,
            "timestamp_formatted": time_formatted,
            "rgb": color.tolist(),
            "hex": "#{:02x}{:02x}{:02x}".format(*color)
        })
    
    return metadata


def save_metadata(metadata: Dict, output_file: str) -> None:
    """
    Save metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing metadata
        output_file: Path to the output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def process_video(
    input_file: str,
    output_file: str,
    method: str = 'kmeans',
    sample_rate: Optional[float] = None,
    sample_frames: Optional[int] = None,
    block_size: int = 50,
    metadata_file: Optional[str] = None
) -> Dict:
    """
    Process a video file to extract and visualize its color palette.
    
    Args:
        input_file: Path to the input video file
        output_file: Path to the output palette image
        method: Method for color extraction ('kmeans' or 'histogram')
        sample_rate: Time in seconds between sampled frames
        sample_frames: Number of frames to skip between samples
        block_size: Size of each color block in pixels
        metadata_file: Path to the output metadata file
    
    Returns:
        Dictionary containing metadata
    """
    # Extract frames
    frames = extract_frames(input_file, sample_rate, sample_frames)
    
    # Analyze colors
    color_data = analyze_colors(frames, method)
    
    # Create color palette
    palette_img = create_color_palette(color_data, block_size)
    
    # Save palette image
    palette_img.save(output_file)
    
    # Generate metadata
    metadata = generate_metadata(color_data, input_file)
    
    # Save metadata if requested
    if metadata_file:
        save_metadata(metadata, metadata_file)
    
    return metadata


def main():
    """Main function to run the script from command line."""
    args = parse_arguments()
    
    try:
        metadata = process_video(
            input_file=args.input,
            output_file=args.output,
            method=args.method,
            sample_rate=args.sample_rate,
            sample_frames=args.sample_frames,
            block_size=args.block_size,
            metadata_file=args.metadata
        )
        
        print(f"Processed video: {args.input}")
        print(f"Color palette saved to: {args.output}")
        if args.metadata:
            print(f"Metadata saved to: {args.metadata}")
            
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
