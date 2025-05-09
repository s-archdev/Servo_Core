#!/usr/bin/env python3
"""
Automated Highlight Reel Generator

This script processes video files to detect and remove periods of silence,
creating a more concise highlight reel from the original content.
"""

import os
import sys
import time
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from pydub import AudioSegment
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('highlight_generator')


@dataclass
class SilenceSegment:
    """Represents a detected silence segment in the video."""
    start_time: float  # in seconds
    end_time: float    # in seconds
    
    @property
    def duration(self) -> float:
        """Get the duration of this silence segment in seconds."""
        return self.end_time - self.start_time


@dataclass
class VideoProcessingConfig:
    """Configuration parameters for video processing."""
    silence_threshold_db: float = -40.0  # silence threshold in dB
    min_silence_duration: float = 1.5    # minimum silence duration in seconds
    audio_sample_rate: int = 44100       # sample rate for audio analysis
    padding: float = 0.05                # seconds to keep around non-silent segments
    output_codec: str = 'libx264'        # video codec for output
    crf_value: int = 18                  # Constant Rate Factor (lower = higher quality)
    audio_codec: str = 'aac'             # audio codec for output
    audio_bitrate: str = '192k'          # audio bitrate for output
    temp_dir: Optional[str] = None       # directory for temporary files


class AudioAnalyzer:
    """Handles audio extraction and silence detection."""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
    
    def extract_audio(self, video_path: str) -> AudioSegment:
        """Extract audio from video file for analysis."""
        logger.info(f"Extracting audio from {video_path}")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.config.temp_dir) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(temp_audio_path, format='wav', ar=self.config.audio_sample_rate)
                .run(quiet=True, overwrite_output=True)
            )
            
            # Load the audio file using pydub
            audio = AudioSegment.from_file(temp_audio_path)
            return audio
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def detect_silence(self, audio: AudioSegment) -> List[SilenceSegment]:
        """
        Detect silence segments in the audio.
        
        Args:
            audio: AudioSegment object containing the audio data
            
        Returns:
            List of SilenceSegment objects representing silent periods
        """
        logger.info("Detecting silence segments")
        
        # Convert to numpy array for processing
        samples = np.array(audio.get_array_of_samples())
        if audio.channels > 1:
            # Convert stereo to mono by averaging channels
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            
        # Convert to float and normalize
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        
        # Calculate RMS amplitude in decibels
        chunk_size = int(self.config.audio_sample_rate * 0.025)  # 25ms chunks
        num_chunks = len(samples) // chunk_size
        
        silence_segments = []
        is_silent = False
        silence_start = 0
        
        for i in range(num_chunks):
            chunk = samples[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) == 0:
                continue
                
            # Calculate RMS power in dB
            rms = np.sqrt(np.mean(chunk ** 2))
            db = 20 * np.log10(rms) if rms > 0 else -100
            
            # Detect transition to silence
            if not is_silent and db < self.config.silence_threshold_db:
                silence_start = i * chunk_size / self.config.audio_sample_rate
                is_silent = True
                
            # Detect transition from silence to sound
            elif is_silent and db >= self.config.silence_threshold_db:
                silence_end = i * chunk_size / self.config.audio_sample_rate
                duration = silence_end - silence_start
                
                if duration >= self.config.min_silence_duration:
                    silence_segments.append(SilenceSegment(silence_start, silence_end))
                    
                is_silent = False
        
        # Check if the file ends with silence
        if is_silent:
            silence_end = len(samples) / self.config.audio_sample_rate
            duration = silence_end - silence_start
            
            if duration >= self.config.min_silence_duration:
                silence_segments.append(SilenceSegment(silence_start, silence_end))
        
        logger.info(f"Detected {len(silence_segments)} silence segments")
        return silence_segments


class VideoProcessor:
    """Handles video processing and segment trimming."""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video file information using ffprobe."""
        logger.info(f"Getting video information for {video_path}")
        
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        # Extract relevant information
        info = {
            'width': int(video_info['width']),
            'height': int(video_info['height']),
            'duration': float(probe['format']['duration']),
            'fps': eval(video_info['r_frame_rate']),  # converts string like '30000/1001' to float
        }
        
        return info
    
    def generate_keep_segments(self, video_duration: float, silence_segments: List[SilenceSegment]) -> List[Tuple[float, float]]:
        """
        Convert silence segments to segments to keep.
        
        Args:
            video_duration: Total duration of the video in seconds
            silence_segments: List of SilenceSegment objects representing silent periods
            
        Returns:
            List of (start_time, end_time) tuples representing segments to keep
        """
        if not silence_segments:
            return [(0, video_duration)]
            
        keep_segments = []
        current_pos = 0
        
        for segment in silence_segments:
            # Add segment before silence (with padding adjustment)
            if segment.start_time > current_pos:
                keep_start = current_pos
                keep_end = max(keep_start, segment.start_time - self.config.padding)
                if keep_end > keep_start:
                    keep_segments.append((keep_start, keep_end))
            
            current_pos = segment.end_time
        
        # Add final segment after the last silence
        if current_pos < video_duration:
            keep_segments.append((current_pos, video_duration))
            
        return keep_segments
    
    def create_trimmed_video(self, input_path: str, output_path: str, keep_segments: List[Tuple[float, float]]) -> None:
        """
        Create a new video with only the segments to keep.
        
        Args:
            input_path: Path to the input video file
            output_path: Path where the output video should be saved
            keep_segments: List of (start_time, end_time) tuples representing segments to keep
        """
        if not keep_segments:
            logger.warning("No segments to keep, output would be empty")
            return
            
        logger.info(f"Creating trimmed video with {len(keep_segments)} segments")
        
        # Create temporary directory for segment files
        with tempfile.TemporaryDirectory(dir=self.config.temp_dir) as temp_dir:
            segment_files = []
            
            # Extract each segment to a separate file
            for i, (start, end) in enumerate(keep_segments):
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                segment_files.append(segment_path)
                
                # Use ffmpeg to extract segment
                (
                    ffmpeg
                    .input(input_path, ss=start, to=end)
                    .output(
                        segment_path,
                        c='copy',  # Use copy codec for speed when creating segments
                        avoid_negative_ts='make_zero'
                    )
                    .run(quiet=True, overwrite_output=True)
                )
            
            # Create a file list for concatenation
            concat_file_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file_path, 'w') as f:
                for segment_path in segment_files:
                    f.write(f"file '{segment_path}'\n")
            
            # Concatenate segments and apply final encoding
            (
                ffmpeg
                .input(concat_file_path, format='concat', safe=0)
                .output(
                    output_path,
                    c=self.config.output_codec,
                    crf=self.config.crf_value,
                    preset='medium',
                    acodec=self.config.audio_codec,
                    audio_bitrate=self.config.audio_bitrate
                )
                .run(quiet=True, overwrite_output=True)
            )
            
        logger.info(f"Trimmed video saved to {output_path}")


class HighlightGenerator:
    """Main class that orchestrates the highlight generation process."""
    
    def __init__(self, config: VideoProcessingConfig = None):
        self.config = config or VideoProcessingConfig()
        self.audio_analyzer = AudioAnalyzer(self.config)
        self.video_processor = VideoProcessor(self.config)
        
    def process_video(self, input_path: str, output_path: str) -> Dict:
        """
        Process a video file to generate a highlight reel.
        
        Args:
            input_path: Path to the input video file
            output_path: Path where the output video should be saved
            
        Returns:
            Dict containing processing statistics
        """
        start_time = time.time()
        
        # Check if input file exists
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get video information
        video_info = self.video_processor.get_video_info(input_path)
        video_duration = video_info['duration']
        
        # Extract audio and detect silence
        audio = self.audio_analyzer.extract_audio(input_path)
        silence_segments = self.audio_analyzer.detect_silence(audio)
        
        # Calculate segments to keep
        keep_segments = self.video_processor.generate_keep_segments(video_duration, silence_segments)
        
        # Create the trimmed video
        self.video_processor.create_trimmed_video(input_path, output_path, keep_segments)
        
        # Calculate statistics
        total_silence_duration = sum(segment.duration for segment in silence_segments)
        original_duration = video_duration
        new_duration = sum(end - start for start, end in keep_segments)
        
        elapsed_time = time.time() - start_time
        
        # Log removal details
        for i, segment in enumerate(silence_segments):
            logger.info(f"Removed silence #{i+1}: {segment.start_time:.2f}s - {segment.end_time:.2f}s (duration: {segment.duration:.2f}s)")
        
        # Return statistics
        stats = {
            'input_file': input_path,
            'output_file': output_path,
            'original_duration': original_duration,
            'new_duration': new_duration,
            'silence_removed': total_silence_duration,
            'reduction_percent': (total_silence_duration / original_duration) * 100 if original_duration > 0 else 0,
            'silence_segments': len(silence_segments),
            'processing_time': elapsed_time
        }
        
        logger.info(f"Video processing complete. Original: {original_duration:.2f}s, "
                   f"New: {new_duration:.2f}s, Removed: {total_silence_duration:.2f}s "
                   f"({stats['reduction_percent']:.1f}%)")
        
        return stats


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated Highlight Reel Generator")
    
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path for the output video file")
    parser.add_argument("--threshold", type=float, default=-40.0,
                        help="Silence threshold in dB (default: -40.0)")
    parser.add_argument("--min-silence", type=float, default=1.5,
                        help="Minimum silence duration in seconds (default: 1.5)")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Padding around silence segments in seconds (default: 0.05)")
    parser.add_argument("--output-codec", default="libx264",
                        help="Video codec for output (default: libx264)")
    parser.add_argument("--crf", type=int, default=18,
                        help="Constant Rate Factor for video quality (default: 18, lower is better)")
    parser.add_argument("--audio-codec", default="aac",
                        help="Audio codec for output (default: aac)")
    parser.add_argument("--audio-bitrate", default="192k",
                        help="Audio bitrate for output (default: 192k)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--temp-dir", help="Directory for temporary files")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create configuration
    config = VideoProcessingConfig(
        silence_threshold_db=args.threshold,
        min_silence_duration=args.min_silence,
        padding=args.padding,
        output_codec=args.output_codec,
        crf_value=args.crf,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        temp_dir=args.temp_dir
    )
    
    # Process the video
    try:
        generator = HighlightGenerator(config)
        stats = generator.process_video(args.input, args.output)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Input file: {stats['input_file']}")
        print(f"Output file: {stats['output_file']}")
        print(f"Original duration: {stats['original_duration']:.2f} seconds")
        print(f"New duration: {stats['new_duration']:.2f} seconds")
        print(f"Silence removed: {stats['silence_removed']:.2f} seconds ({stats['reduction_percent']:.1f}%)")
        print(f"Silent segments detected: {stats['silence_segments']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
