# Video Frame Slicer

A web application that takes a video, slices it into equidistant frames, and creates a new video with these frames - perfect for generating thumbnails and promotional material.

## Features

- Upload videos in various formats (MP4, AVI, MOV, WEBM, MKV)
- Extract frames at equidistant intervals from the video
- Create a new video with extracted frames displayed for customizable durations
- View both original and processed videos side by side
- Display a gallery of all extracted frames
- Download the processed video

## Requirements

- Python 3.6+
- Flask
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/video-frame-slicer.git
cd video-frame-slicer
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```
pip install flask opencv-python numpy
```

## Running the Application

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

## Usage Guide

1. **Upload a Video**
   - Click the "Choose File" button and select a video
   - Click "Upload" and wait for the video to be processed

2. **Configure Settings**
   - Set the number of frames to extract
   - Set the display duration for each frame in the output video
   - Click "Process Video"

3. **View Results**
   - Watch the original and processed videos side by side
   - Browse through the gallery of extracted frames
   - Download the processed video

## Project Structure

```
video-frame-slicer/
│
├── app.py                 # Main Flask application
├── static/                # Static files
│   ├── uploads/           # Uploaded videos
│   └── results/           # Processed videos and frames
└── templates/             # HTML templates
    └── index.html         # Main page
```

## How It Works

1. The application uses OpenCV to extract frames at equidistant intervals from the uploaded video
2. It then creates a new video by displaying each frame for a specified duration
3. The resulting video is a sequence of still frames from the original video, useful for creating thumbnails or promotional materials

## License

This project is licensed under the MIT License - see the LICENSE file for details.
