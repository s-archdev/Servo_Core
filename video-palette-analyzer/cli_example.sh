# Basic usage - sample every 1 second with kmeans method
python video_color_palette.py --input my_video.mp4 --output palette.png --metadata colors.json

# Using histogram method instead of kmeans
python video_color_palette.py --input my_video.mp4 --output palette.png --method histogram

# Sample every 5 seconds
python video_color_palette.py --input my_video.mp4 --output palette.png --sample-rate 5.0

# Sample every 30 frames
python video_color_palette.py --input my_video.mp4 --output palette.png --sample-frames 30

# Create larger color blocks (100x100 pixels)
python video_color_palette.py --input my_video.mp4 --output palette.png --block-size 100

# Combined example with all parameters
python video_color_palette.py --input my_video.mp4 --output palette.png --method histogram --sample-rate 2.5 --block-size 75 --metadata video_colors.json

# For FastAPI server (with uvicorn)
uvicorn api_integration:fastapi_app --reload

# For Flask server
python api_integration.py
