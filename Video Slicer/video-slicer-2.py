from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import cv2
import numpy as np
import uuid
import time
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, num_frames, session_id):
    """Extract equidistant frames from video"""
    # Create folder for frames
    frames_folder = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame positions for equidistant sampling
    if num_frames > frame_count:
        num_frames = frame_count
    
    frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    # Extract frames
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frames_folder, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    cap.release()
    
    return frame_paths, fps, frame_count, duration

def create_sliced_video(frame_paths, fps, session_id, display_duration=1.0):
    """Create video from frames with each frame displayed for display_duration seconds"""
    if not frame_paths:
        return None
    
    # Read first frame to get dimensions
    sample_frame = cv2.imread(frame_paths[0])
    height, width, _ = sample_frame.shape
    
    # Output video path
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_sliced.mp4")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video, displaying for display_duration seconds
    frames_per_image = int(fps * display_duration)
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        for _ in range(frames_per_image):
            out.write(frame)
    
    out.release()
    
    return output_path

def get_video_metadata(video_path):
    """Get basic metadata about the video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': f"{duration:.2f} seconds",
        'duration_raw': duration
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Get video metadata
        metadata = get_video_metadata(file_path)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'file_path': file_path,
            'metadata': metadata
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    session_id = data.get('session_id')
    file_path = data.get('file_path')
    num_frames = int(data.get('num_frames', 10))
    frame_duration = float(data.get('frame_duration', 1.0))
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        # Extract frames
        frame_paths, fps, frame_count, duration = extract_frames(file_path, num_frames, session_id)
        
        # Create sliced video
        output_path = create_sliced_video(frame_paths, fps, session_id, frame_duration)
        
        if output_path:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'original_video': os.path.basename(file_path),
                'sliced_video': os.path.basename(output_path),
                'num_frames': len(frame_paths),
                'frame_paths': [os.path.basename(path) for path in frame_paths],
                'frames_folder': f"{session_id}_frames"
            })
        else:
            return jsonify({'error': 'Failed to create sliced video'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clean', methods=['POST'])
def clean_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
    
    try:
        # Delete session files
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.startswith(session_id):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        
        for file in os.listdir(app.config['RESULTS_FOLDER']):
            if file.startswith(session_id):
                if os.path.isdir(os.path.join(app.config['RESULTS_FOLDER'], file)):
                    shutil.rmtree(os.path.join(app.config['RESULTS_FOLDER'], file))
                else:
                    os.remove(os.path.join(app.config['RESULTS_FOLDER'], file))
        
        return jsonify({'success': True, 'message': 'Session files cleaned'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
