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
import requests

# ----------------------------------------
# FFMPEG setup and dependency management
def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def download_ffmpeg_windows(dest_folder="ffmpeg"):
    # Check if ffmpeg.exe already exists in the expected location
    expected_bin_dir = None
    if os.path.exists(dest_folder):
        for d in os.listdir(dest_folder):
            if d.startswith("ffmpeg"):
                bin_dir = os.path.join(dest_folder, d, "bin")
                ffmpeg_exe = os.path.join(bin_dir, "ffmpeg.exe")
                if os.path.isfile(ffmpeg_exe):
                    print(f"✅ ffmpeg already present at {ffmpeg_exe}")
                    return bin_dir

    print("⚙️  ffmpeg not found, downloading portable version...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(dest_folder, "ffmpeg.zip")
    os.makedirs(dest_folder, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"⬇️  Downloading ffmpeg from {url} ...")
        urlretrieve(url, zip_path)
    else:
        print(f"⚠️  ffmpeg zip already downloaded at {zip_path}")

    print(f"📂 Extracting ffmpeg to {dest_folder} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)

    os.remove(zip_path)

    extracted_dirs = [d for d in os.listdir(dest_folder) if d.startswith("ffmpeg")]
    if not extracted_dirs:
        print("❌ ffmpeg extraction failed.")
        sys.exit(1)

    ffmpeg_root = os.path.join(dest_folder, extracted_dirs[0])
    ffmpeg_bin = os.path.join(ffmpeg_root, "bin")
    print(f"✅ ffmpeg downloaded and extracted to {ffmpeg_root}")
    return ffmpeg_bin

def setup_ffmpeg_binary():
    if is_ffmpeg_installed():
        print("✅ ffmpeg found in system PATH.")
        return None
    else:
        if platform.system() == "Windows":
            ffmpeg_bin = download_ffmpeg_windows()
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_bin, "ffmpeg.exe")
            return os.environ["FFMPEG_BINARY"]
        else:
            print("❌ ffmpeg not found and automatic download only implemented for Windows.")
            print("Please install ffmpeg manually and add it to PATH.")
            sys.exit(1)

# Setup ffmpeg before importing pydub
setup_ffmpeg_binary()

# ----------------------------------------
# Ensure SCNet is installed
def ensure_scnet_installed():
    # Check if SCNet repo folder exists first
    scnet_dir = os.path.join(os.getcwd(), "SCNet")
    if os.path.exists(scnet_dir):
        print("✅ SCNet repo already present.")
    else:
        print("SCNet not found. Cloning SCNet...")
        repo_url = "https://github.com/starrytong/SCNet.git"
        subprocess.check_call(["git", "clone", repo_url, scnet_dir])
        print("✅ SCNet repo cloned.")
    # Check if scnet is importable (for sys.path)
    scnet_spec = importlib.util.find_spec("scnet")
    if scnet_spec is not None:
        scnet_dir = os.path.abspath(os.path.dirname(scnet_spec.origin))
    if scnet_dir not in sys.path:
        sys.path.insert(0, scnet_dir)
    print("SCNet ready.")

    # Ensure checkpoints folder and required files
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
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
        print("⬇️  Downloading SCNet config file...")
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
    else:
        print("✅ SCNet config file already present.")

    return scnet_dir


# Ensure SCNet is downloaded and ready
ensure_scnet_installed()

# ----------------------------------------
# Audio and YouTube utilities
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, COMM

# ----------------------------------------
# Common Utils
def spinner(msg, stop_event):
    spinner_chars = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\r{msg} {spinner_chars[idx % len(spinner_chars)]}", end="", flush=True)
        idx += 1
        time.sleep(0.1)
    print("\r" + " " * (len(msg) + 2) + "\r", end="", flush=True)

def mix_stems_without_drums(stems_path, output_path):
    # Mix all stems except drums
    components = ["vocals", "bass", "other"]
    mixed = None
    for comp in components:
        file = os.path.join(stems_path, f"{comp}.wav")
        if os.path.exists(file):
            seg = AudioSegment.from_wav(file)
            mixed = seg if mixed is None else mixed.overlay(seg)
        else:
            print(f"⚠️  Missing stem: {comp}. Skipping.")
    if mixed:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mixed.export(output_path, format="mp3", bitrate="320k")
        print(f"✅ Final track saved to {output_path}")
    else:
        print("❌ No valid stems found. Cannot create mix without drums.")

def set_mp3_metadata(mp3_path, info):
    """Set title, artist, and comment metadata on an MP3 file using mutagen."""
    if info is None:
        return
    try:
        audio = MP3(mp3_path, ID3=ID3)
        if audio.tags is None:
            audio.add_tags()
        # Try to extract artist and title from info dict
        title = info.get("track") or info.get("title") or ""
        artist = info.get("artist") or info.get("uploader") or info.get("channel") or ""
        # If title contains ' - ', split to get artist and title
        if not artist and ' - ' in title:
            artist, title = title.split(' - ', 1)
        # Set tags
        pretty_title = f"{title.strip()} (Drumless)" if title else "Drumless Version"
        audio.tags.add(TIT2(encoding=3, text=pretty_title))
        if artist:
            audio.tags.add(TPE1(encoding=3, text=artist.strip()))
        audio.tags.add(COMM(encoding=3, lang="eng", desc="", text="Drumless version generated by SCNet"))
        audio.save()
        print(f"✅ Metadata set: Title='{pretty_title}', Artist='{artist.strip() if artist else ''}'")
    except Exception as e:
        print(f"⚠️  Could not set metadata: {e}")

# ----------------------------------------
# Download audio from YouTube
def download_audio(link: str, output_folder: str = "temp") -> str:
    os.makedirs(output_folder, exist_ok=True)
    # Always download if the file for this URL is not present
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
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
        expected_file = os.path.join(output_folder, f"{title}.wav")
    if os.path.exists(expected_file):
        print(f"✅ Audio already downloaded: {expected_file}")
        return expected_file, info_dict
    # Otherwise, download
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("Downloading audio", stop_event))
    spinner_thread.start()
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=True)
            title = info.get("title", "downloaded_track")
            filename = os.path.join(output_folder, f"{title}.wav")
    finally:
        stop_event.set()
        spinner_thread.join()
    print(f"✅ Audio downloaded: {filename}")
    return filename, info

# ----------------------------------------
# Separate stems using SCnet
def separate_stems_scnet(input_wav, output_folder="separated_scnet"):
    import openvino as ov
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly

    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.abspath(os.path.join("ov_model", "model.xml"))
    if not os.path.exists(model_path):
        print(f"❌ OpenVINO model not found at {model_path}. Please export it first.")
        sys.exit(1)
    if os.path.isfile(input_wav):
        track = os.path.splitext(os.path.basename(input_wav))[0]
        output_dir = os.path.join(output_folder, track)
        expected_stem = os.path.join(output_dir, "vocals.wav")
        if os.path.exists(expected_stem):
            print(f"✅ Stems already separated for {track}.")
            return output_dir
    else:
        print(f"❌ Input wav not found: {input_wav}")
        sys.exit(1)

    # Load audio
    audio, sr = sf.read(input_wav)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)  # mono to stereo
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    input_sr = sr
    target_sr = 44100
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz for model inference...")
        audio = resample_poly(audio, target_sr, sr, axis=0)
        sr = target_sr

    # Pad or trim to multiple of segment length (11s)
    segment_len = 11 * target_sr
    n_channels = 2
    n_samples = audio.shape[0]
    if n_samples < segment_len:
        pad = segment_len - n_samples
        audio = np.pad(audio, ((0, pad), (0, 0)), mode='constant')
    elif n_samples % segment_len != 0:
        pad = segment_len - (n_samples % segment_len)
        audio = np.pad(audio, ((0, pad), (0, 0)), mode='constant')

    # Prepare input for model: (batch, channels, samples)
    audio = audio.T  # (channels, samples)
    audio = np.expand_dims(audio, 0).astype(np.float32)  # (1, 2, N)

    # Load OpenVINO model
    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device_name="CPU")
    input_name = compiled.input(0).get_any_name()

    # Run inference in segments
    outputs = {k: [] for k in ['drums', 'bass', 'other', 'vocals']}
    n_segments = audio.shape[2] // segment_len
    print(f"🎚️ Running SCNet inference with OpenVINO on {n_segments} segment(s)...")
    import threading, time
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=("SCNet is processing", stop_event))
    start_time = time.time()
    spinner_thread.start()
    try:
        for i in range(n_segments):
            seg = audio[:, :, i*segment_len:(i+1)*segment_len]
            result = compiled({input_name: seg})
            # Output is a dict: {output_name: np.ndarray}
            # Assume output order: [drums, bass, other, vocals]
            out = list(result.values())[0]  # (1, 4, 2, segment_len)
            for idx, k in enumerate(['drums', 'bass', 'other', 'vocals']):
                outputs[k].append(out[0, idx])
    finally:
        stop_event.set()
        spinner_thread.join()
    elapsed = time.time() - start_time
    print(f"⏱️  SCNet inference completed in {elapsed:.2f} seconds.")

    # Concatenate segments and save stems
    os.makedirs(output_dir, exist_ok=True)
    for k in outputs:
        stem = np.concatenate(outputs[k], axis=1)  # (2, N)
        stem = stem.T  # (N, 2)
        # Trim to original length (in model's sample rate)
        stem = stem[:n_samples, :]
        # Always save at model's sample rate (44100 Hz), skip resampling
        out_path = os.path.join(output_dir, f"{k}.wav")
        sf.write(out_path, stem, target_sr)
        print(f"✅ Saved {k} stem: {out_path}")
    return output_dir

# ----------------------------------------
# Main pipeline
if __name__ == "__main__":
    print("\n=== MakeItDrumless ===\n")
    parser = argparse.ArgumentParser(description="MakeItDrumless")
    parser.add_argument("url", help="YouTube URL")
    args = parser.parse_args()
    url = args.url
    downloaded, info = download_audio(url)
    stems = separate_stems_scnet(downloaded)
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
    output = os.path.join("output", out_filename)
    mix_stems_without_drums(stems, output)
    set_mp3_metadata(output, info)
    print(f"\n🎉 All done! Check the output folder for your no-drum track: {output}\n")