import os
import threading
import time
from urllib.request import urlretrieve
from yt_dlp import YoutubeDL
from src.utils.spinner import spinner

def download_audio(link: str, output_folder: str = "temp", project_root=".") -> str:
    output_folder_abs = os.path.join(project_root, output_folder)
    os.makedirs(output_folder_abs, exist_ok=True)
    # Always download if the file for this URL is not present
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder_abs, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    # Check if file for this URL is present (by extracting info first)
    with YoutubeDL({'quiet': True}) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        title = info_dict.get("title", "downloaded_track")
        expected_file = os.path.join(output_folder_abs, f"{title}.wav")
    if os.path.exists(expected_file):
        print(f"✅ Audio already downloaded: {expected_file}")
        return expected_file, info_dict
    # Otherwise, download
    print("⚙️  Downloading audio...")
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("Downloading audio", stop_event))
    spinner_thread.start()
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=True)
            title = info.get("title", "downloaded_track")
            filename = os.path.join(output_folder_abs, f"{title}.wav")
    finally:
        stop_event.set()
        spinner_thread.join()
    print(f"✅ Audio downloaded: {filename}")
    return filename, info