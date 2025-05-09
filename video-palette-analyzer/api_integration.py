#!/usr/bin/env python3
"""
API Integration for Video Color Palette Analyzer

This module provides FastAPI and Flask integrations for the video color palette analyzer.
"""

import os
import uuid
import shutil
from typing import Optional, List, Dict

# FastAPI integration
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Flask integration
from flask import Flask, request, jsonify, send_file

# Import the video color palette analyzer functions
from video_color_palette import process_video


# Define models for FastAPI
class ColorPaletteRequest(BaseModel):
    method: str = "kmeans"
    sample_rate: Optional[float] = 1.0
    sample_frames: Optional[int] = None
    block_size: int = 50


class ColorPaletteResponse(BaseModel):
    palette_image_url: str
    metadata_url: Optional[str]
    metadata: Dict


# Create FastAPI app
fastapi_app = FastAPI(
    title="Video Color Palette Analyzer API",
    description="API for analyzing color palettes in videos",
    version="1.0.0"
)


@fastapi_app.post("/analyze", response_model=ColorPaletteResponse)
async def analyze_video(
    file: UploadFile = File(...),
    method: str = Form("kmeans"),
    sample_rate: float = Form(1.0),
    sample_frames: Optional[int] = Form(None),
    block_size: int = Form(50)
):
    """
    Analyze a video file and generate a color palette image.
    
    Args:
        file: The video file to analyze
        method: Method for color extraction ('kmeans' or 'histogram')
        sample_rate: Time in seconds between sampled frames
        sample_frames: Number of frames to skip between samples
        block_size: Size of each color block in pixels
    
    Returns:
        ColorPaletteResponse object containing URLs and metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Create temporary directory for this analysis
    os.makedirs("temp", exist_ok=True)
    upload_dir = os.path.join("temp", analysis_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    input_path = os.path.join(upload_dir, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Define output paths
    output_image = os.path.join(upload_dir, "palette.png")
    output_metadata = os.path.join(upload_dir, "metadata.json")
    
    # Process the video
    try:
        metadata = process_video(
            input_file=input_path,
            output_file=output_image,
            method=method,
            sample_rate=sample_rate,
            sample_frames=sample_frames,
            block_size=block_size,
            metadata_file=output_metadata
        )
    except Exception as e:
        # Clean up
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    # Generate URLs for the results
    base_url = f"/files/{analysis_id}"
    palette_image_url = f"{base_url}/palette.png"
    metadata_url = f"{base_url}/metadata.json"
    
    return ColorPaletteResponse(
        palette_image_url=palette_image_url,
        metadata_url=metadata_url,
        metadata=metadata
    )


@fastapi_app.get("/files/{analysis_id}/{filename}")
async def get_file(analysis_id: str, filename: str):
    """
    Retrieve a generated file.
    
    Args:
        analysis_id: Unique ID for the analysis
        filename: Name of the file to retrieve
    
    Returns:
        The requested file
    """
    file_path = os.path.join("temp", analysis_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


# Create Flask app
flask_app = Flask(__name__)


@flask_app.route("/analyze", methods=["POST"])
def flask_analyze_video():
    """
    Flask endpoint to analyze a video file and generate a color palette image.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Get parameters
    method = request.form.get("method", "kmeans")
    sample_rate = float(request.form.get("sample_rate", 1.0))
    sample_frames = request.form.get("sample_frames")
    if sample_frames:
        sample_frames = int(sample_frames)
    block_size = int(request.form.get("block_size", 50))
    
    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Create temporary directory for this analysis
    os.makedirs("temp", exist_ok=True)
    upload_dir = os.path.join("temp", analysis_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    input_path = os.path.join(upload_dir, file.filename)
    file.save(input_path)
    
    # Define output paths
    output_image = os.path.join(upload_dir, "palette.png")
    output_metadata = os.path.join(upload_dir, "metadata.json")
    
    # Process the video
    try:
        metadata = process_video(
            input_file=input_path,
            output_file=output_image,
            method=method,
            sample_rate=sample_rate,
            sample_frames=sample_frames,
            block_size=block_size,
            metadata_file=output_metadata
        )
    except Exception as e:
        # Clean up
        shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500
    
    # Generate URLs for the results
    base_url = f"/files/{analysis_id}"
    palette_image_url = f"{base_url}/palette.png"
    metadata_url = f"{base_url}/metadata.json"
    
    return jsonify({
        "palette_image_url": palette_image_url,
        "metadata_url": metadata_url,
        "metadata": metadata
    })


@flask_app.route("/files/<analysis_id>/<filename>")
def flask_get_file(analysis_id, filename):
    """
    Flask endpoint to retrieve a generated file.
    """
    file_path = os.path.join("temp", analysis_id, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path)


# Run the Flask app if executed directly
if __name__ == "__main__":
    flask_app.run(debug=True)
