"""
FastAPI implementation for the Video Color Palette Analyzer.
"""

import os
import uuid
import shutil
from typing import Optional, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.video_color_palette import process_video


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
app = FastAPI(
    title="Video Color Palette Analyzer API",
    description="API for analyzing color palettes in videos",
    version="1.0.0"
)


def create_app():
    """Create and configure the FastAPI application."""
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create required directories
    os.makedirs("temp", exist_ok=True)
    
    # Mount static files directory
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except:
        pass
    
    return app


@app.post("/analyze", response_model=ColorPaletteResponse)
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


@app.get("/files/{analysis_id}/{filename}")
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


# Default route for health check
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Video Color Palette Analyzer API is running"}
