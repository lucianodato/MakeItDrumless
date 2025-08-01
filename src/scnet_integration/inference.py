import os
import sys
import subprocess
import time
import threading

from src.utils.spinner import spinner

def separate_stems_scnet(input_wav, output_folder="separated_scnet", project_root="."):
    os.makedirs(os.path.join(project_root, output_folder), exist_ok=True)
    model_checkpoint = os.path.abspath(os.path.join(project_root, "checkpoints", "SCNet-large.th"))
    config = os.path.abspath(os.path.join(project_root, "checkpoints", "config.yaml"))
    if not os.path.exists(model_checkpoint):
        print(f"❌ SCNet checkpoint not found at {model_checkpoint}. Please ensure it is downloaded.")
        sys.exit(1)
    if not os.path.exists(config):
        print(f"❌ SCNet config file not found at {config}. Please ensure it is downloaded.")
        sys.exit(1)
    print("🎚️ Running SCNet inference using official CLI...")
    # Only run inference if stems are not already present
    if os.path.isfile(input_wav):
        input_dir = os.path.abspath(os.path.dirname(input_wav))
        track = os.path.splitext(os.path.basename(input_wav))[0]
        expected_stem = os.path.join(project_root, output_folder, track, "vocals.wav")
        if os.path.exists(expected_stem):
            print(f"✅ Stems already separated for {track}.")
            return os.path.join(project_root, output_folder, track)
    else:
        input_dir = os.path.abspath(input_wav)
        track = None
    output_dir_abs = os.path.abspath(os.path.join(project_root, output_folder))
    scnet_dir = os.path.join(project_root, "SCNet")
    start_time = time.time()
    command = [
        sys.executable, "-m", "scnet.inference",
        "--input_dir", input_dir,
        "--output_dir", output_dir_abs,
        "--checkpoint_path", model_checkpoint,
        "--config_path", config
    ]
    print(f"Running SCNet inference... (this may take a while)")
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("SCNet is processing", stop_event))
    spinner_thread.start()
    try:
        subprocess.run(command, check=True, cwd=scnet_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        stop_event.set()
        spinner_thread.join()
    elapsed = time.time() - start_time
    print(f"⏱️  SCNet inference completed in {elapsed:.2f} seconds.")
    if track:
        return os.path.join(project_root, output_folder, track)
    else:
        return os.path.join(project_root, output_folder)
