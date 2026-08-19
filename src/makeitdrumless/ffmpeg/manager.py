import os
import subprocess
import zipfile
from urllib.request import urlretrieve
import platform
import sys
import threading
from pathlib import Path

from makeitdrumless.cli_utils.spinner import spinner


def is_ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def download_ffmpeg_windows(dest_folder: str | None = None) -> str:
    if dest_folder is None:
        dest_folder = str(Path.home() / ".cache" / "makeitdrumless" / "ffmpeg_bin")

    dest_folder_abs = os.path.abspath(dest_folder)
    # Check if ffmpeg.exe already exists in the expected location
    if os.path.exists(dest_folder_abs):
        for d in os.listdir(dest_folder_abs):
            if d.startswith("ffmpeg"):
                bin_dir = os.path.join(dest_folder_abs, d, "bin")
                ffmpeg_exe = os.path.join(bin_dir, "ffmpeg.exe")
                if os.path.isfile(ffmpeg_exe):
                    print(f"✅ ffmpeg already present at {ffmpeg_exe}")
                    return bin_dir

    print("⚙️  ffmpeg not found, downloading portable version for Windows...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(dest_folder_abs, "ffmpeg.zip")
    os.makedirs(dest_folder_abs, exist_ok=True)

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("Downloading ffmpeg", stop_event), daemon=True)
    spinner_thread.start()
    try:
        if not os.path.exists(zip_path):
            urlretrieve(url, zip_path)
    finally:
        stop_event.set()
        spinner_thread.join(timeout=1.0)

    print(f"📂 Extracting ffmpeg to {dest_folder_abs} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder_abs)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    extracted_dirs = [d for d in os.listdir(dest_folder_abs) if d.startswith("ffmpeg")]
    if not extracted_dirs:
        print("❌ ffmpeg extraction failed.")
        sys.exit(1)

    ffmpeg_root = os.path.join(dest_folder_abs, extracted_dirs[0])
    ffmpeg_bin = os.path.join(ffmpeg_root, "bin")
    print(f"✅ ffmpeg downloaded and extracted to {ffmpeg_root}")
    return ffmpeg_bin


def setup_ffmpeg_binary(project_root: str | None = None) -> str | None:
    if is_ffmpeg_installed():
        return None
    else:
        if platform.system() == "Windows":
            ffmpeg_bin = download_ffmpeg_windows()
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_bin, "ffmpeg.exe")
            return os.environ["FFMPEG_BINARY"]
        else:
            print("❌ ffmpeg not found in PATH.")
            print("💡 On macOS, run: brew install ffmpeg")
            print("💡 On Linux, run: sudo apt install ffmpeg")
            sys.exit(1)
