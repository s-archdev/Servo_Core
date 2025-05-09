#!/usr/bin/env python3
"""
Video Color Palette Analyzer

Main entry point for running the application as FastAPI or Flask server.
"""

import argparse
import os
import sys

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        "data/input",
        "data/output",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Created necessary directories.")

def main():
    """Main function to run either FastAPI or Flask server."""
    parser = argparse.ArgumentParser(description="Video Color Palette Analyzer Server")
    parser.add_argument("--server", choices=["fastapi", "flask"], default="fastapi",
                        help="Server type to run (default: fastapi)")
    parser.add_argument("--host", default="127.0.0.1", 
                        help="Host to run the server on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on (default: 8000)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Add the project root to the Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.server == "fastapi":
        import uvicorn
        from app.api.fastapi_app import create_app
        
        app = create_app()
        print(f"Starting FastAPI server at http://{args.host}:{args.port}")
        uvicorn.run("app.api.fastapi_app:app", host=args.host, port=args.port, reload=args.debug)
        
    else:  # flask
        from app.api.flask_app import create_app
        
        app = create_app()
        print(f"Starting Flask server at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
