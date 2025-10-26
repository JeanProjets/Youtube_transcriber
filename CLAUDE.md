# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python YouTube transcription tool that uses OpenAI Whisper for high-quality video transcription. The application provides a Tkinter GUI for batch processing YouTube videos with GPU acceleration support (NVIDIA CUDA).

## Key Architecture

### Core Components

- **main.py** - Tkinter GUI application with threading for non-blocking UI
  - Handles user input (YouTube URLs)
  - Manages progress tracking and logging display
  - Uses queue-based communication between GUI and transcription threads
  - Features dual progress bars (global and detailed)

- **transcriber.py** - Core transcription logic
  - `YouTubeTranscriber` class handles the complete pipeline
  - Audio download via `yt-dlp` with temporary file management
  - Whisper model loading and transcription
  - Automatic cleanup of temporary files

### Data Flow

1. GUI collects YouTube URLs from user input
2. `YouTubeTranscriber` downloads audio using `yt-dlp` to temporary directory
3. Whisper processes audio with configurable model size and device
4. Transcriptions saved to `transcriptions/` directory with timestamps
5. Temporary files cleaned up automatically

## Setup and Running

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r youtube_transcriber_requirements.txt
```

### GPU Setup (CUDA)
For NVIDIA GPU acceleration, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### FFmpeg Requirement
FFmpeg must be installed system-wide:
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg` 
- **Windows**: Download from ffmpeg.org and add to PATH

### Running the Application
```bash
python main.py
```

## Configuration

### Whisper Model Selection
Models can be changed in main.py line 155:
- `tiny` - ~1GB VRAM, fastest but lowest quality
- `base` - ~1GB VRAM, good balance for basic use
- `small` - ~2GB VRAM, better quality
- `medium` - ~5GB VRAM, high quality
- `large-v3` - ~10GB VRAM, highest quality (default)

### Device Selection
Change device in main.py line 156:
- `'cuda'` - Use GPU acceleration (default)
- `'cpu'` - Force CPU usage (slower but works without GPU)

## Output Format

Transcriptions are saved in `transcriptions/` directory with format:
- Filename: `{cleaned_title}_{YYYYMMDD_HHMMSS}.txt`
- Content: Video title header + clean text transcription
- Automatic language detection included in logs

## Development Notes

- No formal testing framework - this is a standalone utility
- No build process - Python script execution only  
- GUI uses queue-based threading to prevent UI blocking
- Extensive error handling for network failures and invalid URLs
- Automatic temporary file cleanup prevents disk space issues
- Progress tracking via custom message queue system between threads