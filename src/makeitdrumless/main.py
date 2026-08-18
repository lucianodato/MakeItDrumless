import os
import sys
import argparse
import time
import shutil
from pathlib import Path

from makeitdrumless.msst_integration.models import (
    list_available_models as list_available_msst_models,
    download_model_preset,
    MODEL_REGISTRY as MSST_MODEL_REGISTRY,
)
from makeitdrumless.msst_integration.inference import separate_stems_msst
from makeitdrumless.mlx_integration.models import (
    MLX_MODEL_REGISTRY,
    list_available_mlx_models,
    is_mlx_model,
)
from makeitdrumless.mlx_integration.inference import separate_stems_mlx
from makeitdrumless.audio.downloader import get_audio_input, get_default_output_base
from makeitdrumless.audio.processing import (
    mix_stems_without_drums,
    set_mp3_metadata,
    ensemble_stems,
)
from makeitdrumless.ffmpeg.manager import setup_ffmpeg_binary
from makeitdrumless.ytmusic import upload_drumless_track, setup_ytmusic_auth


def list_all_models():
    """Prints all available models for both MLX and MSST PyTorch engines."""
    print("\n" + "=" * 90)
    print("🍏 APPLE MLX NATIVE MODELS (High Speed Metal Acceleration on Apple Silicon)")
    print("=" * 90)
    list_available_mlx_models()
    
    print("=" * 90)
    print("🔥 MSST PYTORCH MODELS (SCNet & DualPathRNN Architectures)")
    print("=" * 90)
    list_available_msst_models()


def main():
    print("\n🥁 === MakeItDrumless === 🥁\n")

    parser = argparse.ArgumentParser(
        prog="makeitdrumless",
        description="Generate high-quality drumless backing tracks from YouTube videos or local audio files with Apple MLX or PyTorch MPS acceleration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate drumless track using default high-quality model:
  makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

  # Fast MLX inference with Demucs v4 FT (SDR 10.02, matches SCNet XL):
  makeitdrumless "/path/to/song.mp3" --model mlx_demucs_ft

  # MLX BS-RoFormer (SDR 10.53, exceeds SCNet XL):
  makeitdrumless "/path/to/song.mp3" --model mlx_roformer

  # Classic PyTorch SCNet Large:
  makeitdrumless "/path/to/song.mp3" --model scnet_large_starrytong

  # List all available models (MLX and PyTorch):
  makeitdrumless --list-models
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
        help="Model preset name (e.g. 'scnet_large_starrytong', 'mlx_demucs_ft', 'mlx_roformer'). Run --list-models to see all."
    )
    parser.add_argument(
        "--engine", "-e",
        default="auto",
        choices=["auto", "mlx", "msst"],
        help="Inference engine ('auto' detects by model name, 'mlx' for Apple MLX, 'msst' for PyTorch MSST)."
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device for PyTorch MSST ('auto' selects Apple Silicon MPS on Mac, CUDA on NVIDIA, or CPU)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Custom base output directory (default: ~/Music/MakeItDrumless)."
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all available model presets (MLX & MSST), descriptions, and download status."
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
        help="Chunk overlap factor (e.g. 2, 4, 8). Default is defined in model config."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Batch size for chunk inference (e.g. 1, 2, 4, 8)."
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="Number of random time-shift passes for Demucs (e.g. 1 or 2 for smoother spectrograms). Default: 0."
    )
    parser.add_argument(
        "--ensemble",
        help="Comma-separated list of models to ensemble (e.g. 'scnet_large_starrytong,mlx_demucs_ft')."
    )
    parser.add_argument(
        "--ensemble-weights",
        help="Comma-separated weights for ensemble models (e.g. '0.6,0.4'). Defaults to equal weighting."
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-running separation and overwrite existing cached stems for this model."
    )
    parser.add_argument(
        "--upload-ytmusic", "-u",
        action="store_true",
        help="Automatically upload the generated drumless track to your YouTube Music library."
    )
    parser.add_argument(
        "--setup-ytmusic",
        action="store_true",
        help="Run interactive YouTube Music authentication setup wizard and exit."
    )
    parser.add_argument(
        "--ytmusic-auth",
        help="Custom path to YouTube Music authentication JSON file (default: ~/.config/makeitdrumless/ytmusic_auth.json)."
    )

    args = parser.parse_args()

    # 1. Handle --setup-ytmusic
    if args.setup_ytmusic:
        setup_ytmusic_auth(output_path=args.ytmusic_auth)
        return

    # 2. Handle --list-models
    if args.list_models:
        list_all_models()
        return

    # 3. Handle --download-model
    if args.download_model:
        model_name = args.download_model.strip()
        if is_mlx_model(model_name):
            print(f"📥 Preparing MLX model preset: {model_name}...")
            try:
                from demucs_mlx import Separator
                info = MLX_MODEL_REGISTRY[model_name]
                Separator(model=info["demucs_model"])
                print(f"\n🎉 Successfully downloaded and cached MLX model '{model_name}' ({info['demucs_model']})!")
            except Exception as e:
                print(f"❌ Failed to download MLX model: {e}")
                sys.exit(1)
        else:
            print(f"📥 Downloading MSST model preset: {model_name}...")
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

    # 7. Run separation (Ensemble or Single Model)
    if args.ensemble:
        ensemble_model_names = [m.strip() for m in args.ensemble.split(",") if m.strip()]
        if len(ensemble_model_names) < 2:
            print("⚠️  --ensemble requires at least 2 comma-separated models. Running in single model mode.")
            ensemble_model_names = [ensemble_model_names[0]]

        ensemble_weights = None
        if args.ensemble_weights:
            try:
                ensemble_weights = [float(w.strip()) for w in args.ensemble_weights.split(",") if w.strip()]
            except ValueError:
                print("⚠️  Invalid --ensemble-weights. Using equal weights.")
                ensemble_weights = None

        print(f"\n🔮 Multi-Model Ensemble Separation across: {', '.join(ensemble_model_names)}")
        stems_list = []

        for m_name in ensemble_model_names:
            m_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in m_name)
            m_stems_dir = os.path.join(track_dir, f"stems_{m_tag}")

            if is_mlx_model(m_name) or (args.engine == "mlx"):
                m_stems = separate_stems_mlx(
                    input_audio_path=final_original_wav,
                    output_folder=m_stems_dir,
                    model_preset=m_name,
                    overlap=args.overlap,
                    shifts=args.shifts,
                    batch_size=args.batch_size,
                    force=args.force,
                )
            else:
                m_stems = separate_stems_msst(
                    input_audio_path=final_original_wav,
                    output_folder=m_stems_dir,
                    model_preset=m_name,
                    config_path=args.config,
                    checkpoint_path=args.checkpoint,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    shifts=args.shifts,
                    device_name=args.device,
                    force=args.force,
                )
            stems_list.append(m_stems)

        # Blend ensemble
        ensemble_tag = "_".join("".join(c if c.isalnum() else "_" for c in m) for m in ensemble_model_names)
        stems_dir = os.path.join(track_dir, f"stems_ensemble_{ensemble_tag}")
        stems = ensemble_stems(stems_list, weights=ensemble_weights, output_dir=stems_dir)
        model_display_name = f"Ensemble ({'+'.join(ensemble_model_names)})"
    else:
        # Single model path
        use_mlx = False
        if args.engine == "mlx":
            use_mlx = True
        elif args.engine == "msst":
            use_mlx = False
        else: # auto
            use_mlx = is_mlx_model(args.model)

        model_tag = args.model if not args.checkpoint else os.path.splitext(os.path.basename(args.checkpoint))[0]
        clean_model_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_tag)
        stems_dir = os.path.join(track_dir, f"stems_{clean_model_tag}")

        if use_mlx:
            mlx_model_preset = args.model if is_mlx_model(args.model) else "mlx_demucs_ft"
            stems = separate_stems_mlx(
                input_audio_path=final_original_wav,
                output_folder=stems_dir,
                model_preset=mlx_model_preset,
                overlap=args.overlap,
                shifts=args.shifts,
                batch_size=args.batch_size,
                force=args.force,
            )
        else:
            stems = separate_stems_msst(
                input_audio_path=final_original_wav,
                output_folder=stems_dir,
                model_preset=args.model,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                shifts=args.shifts,
                device_name=args.device,
                force=args.force,
            )
        model_display_name = args.model

    # 8. Mix non-drum stems into drumless MP3
    out_mp3_path = os.path.join(track_dir, f"{safe_title} (Drumless).mp3")

    mix_stems_without_drums(stems, out_mp3_path)
    set_mp3_metadata(out_mp3_path, info, model_name=model_display_name)

    # 9. Upload to YouTube Music if requested
    if args.upload_ytmusic:
        upload_drumless_track(out_mp3_path, auth_file=args.ytmusic_auth)

    total_elapsed = time.time() - start_total_time
    print(f"\n🎉 All done in {total_elapsed:.1f}s!")
    print(f"📁 Track Folder: {track_dir}")
    print(f"  🎵 Drumless MP3:   {out_mp3_path}")
    print(f"  🎙️ Original Audio:  {final_original_wav}")
    print(f"  🎛️ Separated Stems: {stems_dir}\n")


if __name__ == "__main__":
    main()
