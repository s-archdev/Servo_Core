# Video Color Palette Analyzer

A tool for analyzing and extracting color palettes from video files. This application can identify dominant colors across video frames and generate visual representations of the color palette over time.

## Features

- Extract dominant colors from video files
- Analyze color palette changes over time
- Generate visual representations of color data
- Web interface for easy interaction and visualization
- Support for both FastAPI and Flask server backends

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/video-color-palette-analyzer.git
   cd video-color-palette-analyzer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. For development installation:
   ```
   pip install -e .
   ```

## Usage

### Starting the server

Run with FastAPI (default):
```
python main.py
```

Run with specific options:
```
python main.py --server [fastapi|flask] --host 127.0.0.1 --port 8000 --debug
```

### API Endpoints

Once the server is running, you can access:

- Web Interface: `http://localhost:8000/`
- API Documentation (FastAPI): `http://localhost:8000/docs`
- API Endpoints:
  - Upload Video: POST `/api/upload`
  - Analyze Video: POST `/api/analyze`
  - Get Results: GET `/api/results/{job_id}`

## Project Structure

```
video-color-palette-analyzer/
├── app/
│   ├── api/
│   │   ├── fastapi_app.py
│   │   └── flask_app.py
│   ├── core/
│   │   ├── color_analyzer.py
│   │   ├── frame_extractor.py
│   │   └── palette_generator.py
│   ├── utils/
│   │   ├── file_manager.py
│   │   └── video_utils.py
│   └── web/
│       ├── static/
│       └── templates/
├── data/
│   ├── input/
│   └── output/
├── temp/
├── tests/
│   ├── test_color_analyzer.py
│   └── test_frame_extractor.py
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Development

### Running Tests

```
pytest
```

### Code Formatting

```
black .
isort .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for video processing
- scikit-learn for color clustering algorithms
- FastAPI and Flask for web server frameworks