import os
import sys
import argparse
import time

from makeitdrumless.msst_integration.models import (
    list_available_models,
    download_model_preset,
    MODEL_REGISTRY,
)
from makeitdrumless.msst_integration.inference import separate_stems_msst
from makeitdrumless.audio.downloader import get_audio_input
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
  # Generate drumless track from YouTube URL with default model (SCNet XL):
  makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

  # Process a local MP3 or WAV file:
  makeitdrumless "/path/to/my_song.mp3"

  # Use a specific model preset (e.g. BS-Conformer or fast SCNet Small):
  makeitdrumless "/path/to/my_song.mp3" --model bs_conformer

  # Save the output MP3 into a specific folder:
  makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o ~/Music

  # List all available pretrained models:
  makeitdrumless --list-models

  # Pre-download a model checkpoint:
  makeitdrumless --download-model bs_conformer
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
        default=".",
        help="Output directory for generated drumless MP3 (default: current working directory)."
    )
    parser.add_argument(
        "--save-stems",
        metavar="DIR",
        help="Optional directory to save all individual separated stems (vocals, bass, other, drums)."
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

    # 5. Acquire Audio Input (Download or convert local file)
    audio_wav, info = get_audio_input(args.input)

    # 6. Run Stem Separation with MSST & Apple Silicon MPS
    stems = separate_stems_msst(
        input_audio_path=audio_wav,
        output_folder=args.save_stems,
        model_preset=args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device_name=args.device,
        force=args.force,
    )

    # 7. Mix non-drum stems into drumless backing track
    out_title = None
    out_artist = None
    if info:
        out_title = info.get("track") or info.get("title")
        out_artist = info.get("artist") or info.get("uploader") or info.get("channel")
        if not out_artist and out_title and ' - ' in out_title:
            out_artist, out_title = out_title.split(' - ', 1)

    if out_title:
        # Sanitize filename
        safe_title = "".join(c for c in out_title if c not in r'\/:*?"<>|').strip()
        out_filename = f"{safe_title} (Drumless).mp3"
    else:
        out_filename = os.path.splitext(os.path.basename(audio_wav))[0] + " (Drumless).mp3"

    output_dir_abs = os.path.abspath(args.output_dir)
    output_filepath = os.path.join(output_dir_abs, out_filename)

    mix_stems_without_drums(stems, output_filepath)
    set_mp3_metadata(output_filepath, info, model_name=args.model)

    total_elapsed = time.time() - start_total_time
    print(f"\n🎉 All done in {total_elapsed:.1f}s! Check your drumless track at:")
    print(f"👉 {output_filepath}\n")


if __name__ == "__main__":
    main()
