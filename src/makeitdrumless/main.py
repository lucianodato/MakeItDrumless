import os
import sys
import argparse
import time
import shutil
from pathlib import Path

from makeitdrumless.msst_integration.models import (
    list_available_models,
    download_model_preset,
    MODEL_REGISTRY,
)
from makeitdrumless.msst_integration.inference import separate_stems_msst
from makeitdrumless.audio.downloader import get_audio_input, get_default_output_base
from makeitdrumless.audio.processing import mix_stems_without_drums, set_mp3_metadata
from makeitdrumless.ffmpeg.manager import setup_ffmpeg_binary


def main():
    print("\n🥁 === MakeItDrumless === 🥁\n")

    parser = argparse.ArgumentParser(
        prog="makeitdrumless",
        description="Generate high-quality drumless backing tracks from YouTube videos or local audio files with Apple Silicon MPS acceleration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate drumless track from YouTube URL (saves to ~/Music/MakeItDrumless/<Song Title>/):
  makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

  # Process a local MP3 or WAV file:
  makeitdrumless "/path/to/my_song.mp3"

  # Use a specific model preset (e.g. BS-Conformer or fast SCNet Small):
  makeitdrumless "/path/to/my_song.mp3" --model bs_conformer

  # Save output to a custom directory:
  makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o ~/Desktop/MyTracks

  # List all available pretrained models:
  makeitdrumless --list-models

  # Force re-running separation even if stems already exist:
  makeitdrumless "/path/to/my_song.mp3" --force
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="YouTube URL or path to a local audio file (WAV, MP3, FLAC, M4A, etc.)"
    )
    parser.add_argument(
        "--model", "-m",
        default="scnet_large_starrytong",
        help="Model preset name (default: 'scnet_large_starrytong'). Run with --list-models to see all options."
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device for inference ('auto' selects Apple Silicon MPS on Mac, CUDA on NVIDIA, or CPU)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Custom base output directory (default: ~/Music/MakeItDrumless)."
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all available model presets, descriptions, and download status."
    )
    parser.add_argument(
        "--download-model",
        metavar="PRESET",
        help="Download weights and config for a specific model preset and exit."
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to a custom YAML model configuration file."
    )
    parser.add_argument(
        "--checkpoint", "-k",
        help="Path to a custom model checkpoint file (.ckpt)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size in samples for inference (e.g. 132300 for 3s, 264600 for 6s)."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Chunk overlap factor (e.g. 2 or 4). Default is defined in model config."
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-running separation and overwrite existing cached stems for this model."
    )

    args = parser.parse_args()

    # 1. Handle --list-models
    if args.list_models:
        list_available_models()
        return

    # 2. Handle --download-model
    if args.download_model:
        model_name = args.download_model.strip()
        print(f"📥 Downloading model preset: {model_name}...")
        try:
            m_type, cfg_p, ckpt_p = download_model_preset(model_name)
            print(f"\n🎉 Successfully downloaded and cached '{model_name}'!")
            print(f"  - Config:     {cfg_p}")
            print(f"  - Checkpoint: {ckpt_p}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            sys.exit(1)
        return

    # 3. Setup FFmpeg
    setup_ffmpeg_binary()

    # 4. Validate input source
    if not args.input:
        parser.print_help()
        print("\n❌ Error: Please provide a YouTube URL or local audio file path.\n")
        sys.exit(1)

    start_total_time = time.time()

    # 5. Determine Base Output Directory (~/Music/MakeItDrumless by default)
    base_output_dir = os.path.abspath(args.output_dir or get_default_output_base())
    os.makedirs(base_output_dir, exist_ok=True)

    # 6. Acquire Audio Input (Download or convert)
    initial_audio_wav, info = get_audio_input(args.input, output_folder=base_output_dir)

    # Extract clean track title
    out_title = None
    out_artist = None
    if info:
        out_title = info.get("track") or info.get("title")
        out_artist = info.get("artist") or info.get("uploader") or info.get("channel")
        if not out_artist and out_title and ' - ' in out_title:
            out_artist, out_title = out_title.split(' - ', 1)

    if out_title:
        safe_title = "".join(c for c in out_title if c not in r'\/:*?"<>|').strip()
    else:
        raw_name = os.path.splitext(os.path.basename(initial_audio_wav))[0].replace(" (Original)", "")
        safe_title = "".join(c for c in raw_name if c not in r'\/:*?"<>|').strip()

    # Create dedicated song output directory: ~/Music/MakeItDrumless/<Safe Title>/
    track_dir = os.path.join(base_output_dir, safe_title)
    os.makedirs(track_dir, exist_ok=True)

    # Move/place original WAV inside the song's folder
    final_original_wav = os.path.join(track_dir, f"{safe_title} (Original).wav")
    if os.path.abspath(initial_audio_wav) != os.path.abspath(final_original_wav):
        if not os.path.exists(final_original_wav) or args.force:
            shutil.copy2(initial_audio_wav, final_original_wav)
            if os.path.dirname(os.path.abspath(initial_audio_wav)) == base_output_dir:
                try:
                    os.remove(initial_audio_wav)
                except Exception:
                    pass

    # 7. Run Stem Separation (stems saved into track_dir/stems_<model>/)
    model_tag = args.model if not args.checkpoint else os.path.splitext(os.path.basename(args.checkpoint))[0]
    clean_model_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_tag)
    stems_dir = os.path.join(track_dir, f"stems_{clean_model_tag}")

    stems = separate_stems_msst(
        input_audio_path=final_original_wav,
        output_folder=stems_dir,
        model_preset=args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device_name=args.device,
        force=args.force,
    )

    # 8. Mix non-drum stems into drumless MP3
    out_mp3_path = os.path.join(track_dir, f"{safe_title} (Drumless).mp3")

    mix_stems_without_drums(stems, out_mp3_path)
    set_mp3_metadata(out_mp3_path, info, model_name=args.model)

    total_elapsed = time.time() - start_total_time
    print(f"\n🎉 All done in {total_elapsed:.1f}s!")
    print(f"📁 Track Folder: {track_dir}")
    print(f"  🎵 Drumless MP3:   {out_mp3_path}")
    print(f"  🎙️ Original Audio:  {final_original_wav}")
    print(f"  🎛️ Separated Stems: {stems_dir}\n")


if __name__ == "__main__":
    main()
