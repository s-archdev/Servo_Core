"""
Flask implementation for the Video Color Palette Analyzer.
"""

import os
import uuid
import shutil
from typing import Optional

from flask import Flask, request, jsonify, send_file, render_template

from app.core.video_color_palette import process_video


def create_app():
    """Create and configure the Flask application."""
    
    # Create Flask app
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates"),
                static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static"))
    
    # Create required directories
    os.makedirs("temp", exist_ok=True)
    
    # Register routes
    
    @app.route("/")
    def index():
        """Render the index page."""
        try:
            return render_template("index.html")
        except:
            return jsonify({"status": "ok", "message": "Video Color Palette Analyzer API is running"})
    
    @app.route("/analyze", methods=["POST"])
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
    
    
    @app.route("/files/<analysis_id>/<filename>")
    def flask_get_file(analysis_id, filename):
        """
        Flask endpoint to retrieve a generated file.
        """
        file_path = os.path.join("temp", analysis_id, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path)
    
    return app


# Create app instance for direct execution
flask_app = create_app()

# Run the Flask app if executed directly
if __name__ == "__main__":
    flask_app.run(debug=True)
