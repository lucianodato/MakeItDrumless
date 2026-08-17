import os
import tempfile
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from yt_dlp import YoutubeDL
from pydub import AudioSegment

from makeitdrumless.cli_utils.spinner import spinner


def get_default_temp_dir() -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "makeitdrumless")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_audio_input(input_source: str, output_folder: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Handles fetching audio from either a YouTube URL or a local audio file.

    Returns:
        (wav_file_path, metadata_dict)
    """
    target_dir = output_folder or get_default_temp_dir()
    os.makedirs(target_dir, exist_ok=True)

    # 1. Check if input is a local file
    if os.path.isfile(input_source):
        print(f"📂 Using local audio file: {input_source}")
        ext = os.path.splitext(input_source)[1].lower()
        base_name = os.path.splitext(os.path.basename(input_source))[0]
        target_wav = os.path.join(target_dir, f"{base_name}.wav")

        info_dict = {"title": base_name}
        if ext == ".wav":
            return os.path.abspath(input_source), info_dict

        # Convert to WAV if needed
        if not os.path.exists(target_wav):
            print(f"🔄 Converting {ext} to WAV format...")
            seg = AudioSegment.from_file(input_source)
            seg.export(target_wav, format="wav")
            print(f"✅ Converted: {target_wav}")

        return target_wav, info_dict

    # 2. Treat as URL (YouTube, etc.)
    return download_audio(input_source, output_folder=target_dir)


def download_audio(link: str, output_folder: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Downloads audio from YouTube/supported URL using yt-dlp."""
    target_dir = output_folder or get_default_temp_dir()
    os.makedirs(target_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(target_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    # Check if file for this URL is already present
    with YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        title = info_dict.get("title", "downloaded_track")
        expected_file = os.path.join(target_dir, f"{title}.wav")

    if os.path.exists(expected_file) and os.path.getsize(expected_file) > 0:
        print(f"✅ Audio already downloaded: {expected_file}")
        return expected_file, info_dict

    # Otherwise, download
    print(f"⚙️  Downloading audio from URL: {link}...")
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("Downloading audio", stop_event))
    spinner_thread.start()
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=True)
            title = info.get("title", "downloaded_track")
            filename = os.path.join(target_dir, f"{title}.wav")
    finally:
        stop_event.set()
        spinner_thread.join()

    print(f"✅ Audio downloaded: {filename}")
    return filename, info
