"""
Core functionality for the Video Color Palette Analyzer.
"""

from app.core.video_color_palette import (
    extract_frames,
    get_dominant_color_kmeans,
    get_dominant_color_histogram,
    analyze_colors,
    create_color_palette,
    generate_metadata,
    save_metadata,
    process_video
)

__all__ = [
    "extract_frames",
    "get_dominant_color_kmeans",
    "get_dominant_color_histogram",
    "analyze_colors",
    "create_color_palette",
    "generate_metadata",
    "save_metadata",
    "process_video"
]
