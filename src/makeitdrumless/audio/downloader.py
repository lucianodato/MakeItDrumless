import os
import tempfile
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

from makeitdrumless.cli_utils.spinner import spinner


def get_default_output_base() -> str:
    """Returns the default macOS/system Music output directory (~/Music/MakeItDrumless)."""
    music_dir = Path.home() / "Music" / "MakeItDrumless"
    return str(music_dir)


def clean_audio_title(title_or_path: str) -> str:
    """Strips suffixes like (Original) or (Drumless) and removes invalid path characters."""
    base = os.path.splitext(os.path.basename(title_or_path))[0]
    cleaned = base.replace(" (Original)", "").replace(" (Drumless)", "").strip()
    return "".join(c for c in cleaned if c not in r'\/:*?"<>|').strip()


def parse_artist_title(raw_title: str) -> Tuple[Optional[str], str]:
    """Splits 'Artist - Track' into artist and track title if separator is found."""
    clean = raw_title.replace(" (Original)", "").replace(" (Drumless)", "").strip()
    if ' - ' in clean:
        parts = clean.split(' - ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
        return artist, title
    return None, clean


def get_audio_input(input_source: str, output_folder: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Handles fetching audio from a YouTube URL, a local audio file, or an existing track directory.

    Returns:
        (wav_file_path, metadata_dict)
    """
    target_dir = os.path.abspath(output_folder or get_default_output_base())
    os.makedirs(target_dir, exist_ok=True)

    # 1. Check if input is an existing directory
    if os.path.isdir(input_source):
        dir_path = os.path.abspath(input_source)
        print(f"📂 Using song directory: {dir_path}")
        dir_name = os.path.basename(dir_path)
        artist, title = parse_artist_title(dir_name)
        safe_title = clean_audio_title(title if title else dir_name)

        # Look for existing WAV/audio in this folder
        candidate_names = [
            f"{safe_title} (Original).wav",
            f"{dir_name} (Original).wav",
            f"{safe_title}.wav",
            f"{dir_name}.wav",
        ]
        for cname in candidate_names:
            cpath = os.path.join(dir_path, cname)
            if os.path.exists(cpath) and os.path.getsize(cpath) > 0:
                print(f"✅ Using existing audio file: {cpath}")
                return cpath, {"title": safe_title, "artist": artist}

        # Search for any audio file in the directory
        for file in sorted(os.listdir(dir_path)):
            if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")) and not file.lower().endswith("(drumless).mp3"):
                audio_file = os.path.join(dir_path, file)
                if file.lower().endswith(".wav"):
                    return audio_file, {"title": safe_title, "artist": artist}
                # Convert to WAV
                target_wav = os.path.join(dir_path, f"{safe_title} (Original).wav")
                print(f"🔄 Converting {file} to WAV format: {target_wav}...")
                seg = AudioSegment.from_file(audio_file)
                seg.export(target_wav, format="wav")
                return target_wav, {"title": safe_title, "artist": artist}

    # 2. Check if input is a local file
    if os.path.isfile(input_source):
        print(f"📂 Using local audio file: {input_source}")
        ext = os.path.splitext(input_source)[1].lower()
        base_name = os.path.splitext(os.path.basename(input_source))[0]
        artist, title = parse_artist_title(base_name)
        safe_name = clean_audio_title(title if title else base_name)
        target_wav = os.path.join(target_dir, safe_name, f"{safe_name} (Original).wav")

        info_dict = {"title": safe_name, "artist": artist}
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

    # 3. Check if input_source refers to a track folder name under target_dir
    possible_track_dir = os.path.join(target_dir, input_source)
    if os.path.isdir(possible_track_dir):
        return get_audio_input(possible_track_dir, output_folder=target_dir)

    # 4. Treat as URL (YouTube, etc.)
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

    artist = info_dict.get("artist") or info_dict.get("uploader") or info_dict.get("channel")
    track_field = info_dict.get("track")
    if not artist and title and ' - ' in title:
        parsed_artist, parsed_title = parse_artist_title(title)
        artist = parsed_artist
        clean_title = clean_audio_title(parsed_title)
    elif track_field:
        clean_title = clean_audio_title(track_field)
    else:
        clean_title = clean_audio_title(title)

    raw_safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()

    # Check potential cached locations before downloading
    candidate_paths = [
        os.path.join(target_dir, clean_title, f"{clean_title} (Original).wav"),
        os.path.join(target_dir, raw_safe_title, f"{raw_safe_title} (Original).wav"),
        os.path.join(target_dir, f"{clean_title} (Original).wav"),
        os.path.join(target_dir, f"{raw_safe_title} (Original).wav"),
        os.path.join(target_dir, clean_title, f"{title}.wav"),
        os.path.join(target_dir, raw_safe_title, f"{title}.wav"),
        os.path.join(tempfile.gettempdir(), "makeitdrumless", f"{clean_title}.wav"),
        os.path.join(tempfile.gettempdir(), "makeitdrumless", f"{raw_safe_title}.wav"),
    ]

    for cand in candidate_paths:
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            print(f"✅ Audio already downloaded: {cand}")
            return cand, info_dict

    # Destination file in dedicated song directory
    song_dir = os.path.join(target_dir, clean_title)
    os.makedirs(song_dir, exist_ok=True)
    expected_file = os.path.join(song_dir, f"{clean_title} (Original).wav")

    ydl_download_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(song_dir, f"{clean_title} (Original).%(ext)s"),
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
    spinner_thread = threading.Thread(target=spinner, args=("Downloading audio", stop_event), daemon=True)
    spinner_thread.start()
    try:
        with YoutubeDL(ydl_download_opts) as ydl:
            info = ydl.extract_info(link, download=True)
    finally:
        stop_event.set()
        spinner_thread.join(timeout=1.0)

    print(f"✅ Audio downloaded: {expected_file}")
    return expected_file, info_dict
