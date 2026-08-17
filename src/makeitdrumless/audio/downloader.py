import os
import tempfile
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from yt_dlp import YoutubeDL
from pydub import AudioSegment

from makeitdrumless.cli_utils.spinner import spinner


def get_default_output_base() -> str:
    """Returns the default macOS/system Music output directory (~/Music/MakeItDrumless)."""
    music_dir = Path.home() / "Music" / "MakeItDrumless"
    return str(music_dir)


def get_audio_input(input_source: str, output_folder: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Handles fetching audio from either a YouTube URL or a local audio file.

    Returns:
        (wav_file_path, metadata_dict)
    """
    target_dir = os.path.abspath(output_folder or get_default_output_base())
    os.makedirs(target_dir, exist_ok=True)

    # 1. Check if input is a local file
    if os.path.isfile(input_source):
        print(f"📂 Using local audio file: {input_source}")
        ext = os.path.splitext(input_source)[1].lower()
        base_name = os.path.splitext(os.path.basename(input_source))[0]
        safe_name = "".join(c for c in base_name if c not in r'\/:*?"<>|').strip()
        target_wav = os.path.join(target_dir, safe_name, f"{safe_name} (Original).wav")

        info_dict = {"title": safe_name}
        if ext == ".wav" and os.path.abspath(input_source) == os.path.abspath(target_wav):
            return os.path.abspath(input_source), info_dict

        # If already exists in track folder
        if os.path.exists(target_wav) and os.path.getsize(target_wav) > 0:
            print(f"✅ Using existing audio file: {target_wav}")
            return target_wav, info_dict

        # Convert/copy to WAV in target directory if needed
        os.makedirs(os.path.dirname(target_wav), exist_ok=True)
        if not os.path.exists(target_wav):
            print(f"🔄 Converting {ext} to WAV format in {os.path.dirname(target_wav)}...")
            seg = AudioSegment.from_file(input_source)
            seg.export(target_wav, format="wav")
            print(f"✅ Saved original WAV: {target_wav}")

        return target_wav, info_dict

    # 2. Treat as URL (YouTube, etc.)
    return download_audio(input_source, output_folder=target_dir)


def download_audio(link: str, output_folder: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Downloads audio from YouTube/supported URL using yt-dlp with caching and anti-bot headers."""
    target_dir = os.path.abspath(output_folder or get_default_output_base())
    os.makedirs(target_dir, exist_ok=True)

    ydl_extract_opts = {
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web', 'mweb'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        }
    }

    # Extract metadata without downloading
    try:
        with YoutubeDL(ydl_extract_opts) as ydl:
            info_dict = ydl.extract_info(link, download=False)
            title = info_dict.get("title", "downloaded_track")
    except Exception:
        # Fallback with minimal options if extraction has quirks
        with YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info_dict = ydl.extract_info(link, download=False)
            title = info_dict.get("title", "downloaded_track")

    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()

    # Check potential cached locations before downloading
    candidate_paths = [
        os.path.join(target_dir, safe_title, f"{safe_title} (Original).wav"),
        os.path.join(target_dir, f"{safe_title} (Original).wav"),
        os.path.join(target_dir, safe_title, f"{title}.wav"),
        os.path.join(target_dir, f"{title}.wav"),
        os.path.join(tempfile.gettempdir(), "makeitdrumless", f"{safe_title}.wav"),
        os.path.join(tempfile.gettempdir(), "makeitdrumless", f"{title}.wav"),
    ]

    for cand in candidate_paths:
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            print(f"✅ Audio already downloaded: {cand}")
            return cand, info_dict

    # Destination file in dedicated song directory
    song_dir = os.path.join(target_dir, safe_title)
    os.makedirs(song_dir, exist_ok=True)
    expected_file = os.path.join(song_dir, f"{safe_title} (Original).wav")

    ydl_download_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(song_dir, f"{safe_title} (Original).%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web', 'mweb'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
        'quiet': True,
        'no_warnings': True,
    }

    # Download
    print(f"⚙️  Downloading audio from URL: {link}...")
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("Downloading audio", stop_event))
    spinner_thread.start()
    try:
        with YoutubeDL(ydl_download_opts) as ydl:
            info = ydl.extract_info(link, download=True)
    finally:
        stop_event.set()
        spinner_thread.join()

    print(f"✅ Audio downloaded: {expected_file}")
    return expected_file, info_dict
