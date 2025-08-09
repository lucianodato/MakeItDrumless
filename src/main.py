# ----------------------------------------
# Standard library imports
import os
import sys
import subprocess
import importlib.util
import platform
import zipfile
import time
from urllib.request import urlretrieve
import argparse
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------
# Internal imports
from src.ffmpeg.manager import setup_ffmpeg_binary
from src.scnet_integration.repo_manager import ensure_scnet_installed
from src.audio.processing import mix_stems_without_drums, set_mp3_metadata
from src.audio.downloader import download_audio
from src.scnet_integration.inference import separate_stems_scnet

# Setup ffmpeg before importing pydub
setup_ffmpeg_binary(PROJECT_ROOT)

# Ensure SCNet is downloaded and ready
ensure_scnet_installed(PROJECT_ROOT)

# ----------------------------------------
# Main pipeline
if __name__ == "__main__":
    print("\n=== Drumless Track Generator ===\n")
    parser = argparse.ArgumentParser(description="Drumless Track Generator")
    parser.add_argument("url", help="YouTube URL")
    args = parser.parse_args()
    url = args.url
    downloaded, info = download_audio(url, project_root=PROJECT_ROOT)
    stems = separate_stems_scnet(downloaded, project_root=PROJECT_ROOT)
    # Determine pretty output filename based on metadata
    out_title = None
    out_artist = None
    if info:
        out_title = info.get("track") or info.get("title")
        out_artist = info.get("artist") or info.get("uploader") or info.get("channel")
        if not out_artist and out_title and ' - ' in out_title:
            out_artist, out_title = out_title.split(' - ', 1)
    if out_title:
        out_filename = f"{out_title.strip()} (Drumless).mp3"
    else:
        out_filename = os.path.splitext(os.path.basename(downloaded))[0] + "_no_drums.mp3"
    output = os.path.join(PROJECT_ROOT, "output", out_filename)
    mix_stems_without_drums(stems, output, project_root=PROJECT_ROOT)
    set_mp3_metadata(output, info)
    print(f"\n🎉 All done! Check the output folder for your no-drum track: {output}\n")
