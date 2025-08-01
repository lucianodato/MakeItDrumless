import os
import subprocess
import importlib.util
import requests
import sys
import threading
import time

from src.utils.spinner import spinner

def ensure_scnet_installed(project_root="."):
    # Check if SCNet repo folder exists first
    scnet_dir = os.path.join(project_root, "SCNet")
    if os.path.exists(scnet_dir):
        print("✅ SCNet repo already present.")
    else:
        print("⚙️  SCNet not found. Cloning SCNet...")
        repo_url = "https://github.com/starrytong/SCNet.git"
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=("Cloning SCNet repository", stop_event))
        spinner_thread.start()
        try:
            subprocess.check_call(["git", "clone", repo_url, scnet_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            stop_event.set()
            spinner_thread.join()
        print("✅ SCNet repo cloned.")
    # Check if scnet is importable (for sys.path)
    scnet_spec = importlib.util.find_spec("scnet")
    if scnet_spec is not None:
        scnet_dir = os.path.abspath(os.path.dirname(scnet_spec.origin))
    if scnet_dir not in sys.path:
        sys.path.insert(0, scnet_dir)
    print("✅ SCNet ready.")

    # Ensure checkpoints folder and required files
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "SCNet-large.th")
    config_path = os.path.join(checkpoints_dir, "config.yaml")

    # URLs for the files
    checkpoint_url = "https://drive.google.com/file/d/1s7QvQwn8ag9oVstGDBQ6KZvacJkvyK7t/view?usp=drivesdk"
    config_url = "https://drive.google.com/uc?export=download&id=1qxK7SZx6-Gsp1s3wCrj98X7--UcI4O3K"

    if not os.path.exists(checkpoint_path):
        print("❗ SCNet checkpoint file not found.")
        print("Please download the SCNet checkpoint manually from the following URL:")
        print(checkpoint_url)
        print(f"and place it as '{checkpoint_path}'")
        print("After placing the file, rerun this script.")
        sys.exit(1)
    else:
        print("✅ SCNet checkpoint already present.")

    if not os.path.exists(config_path):
        print("⚙️  Downloading SCNet config file...")
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=("Downloading SCNet config", stop_event))
        spinner_thread.start()
        try:
            with requests.get(config_url, stream=True) as r:
                r.raise_for_status()
                with open(config_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"❌ Failed to download config file: {e}")
            sys.exit(1)
        finally:
            stop_event.set()
            spinner_thread.join()
    else:
        print("✅ SCNet config file already present.")

    return scnet_dir
