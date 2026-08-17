import os
import sys
import time
import tempfile
from typing import Optional, List, Dict
import numpy as np
import soundfile as sf
import librosa
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from makeitdrumless.msst_integration.device import get_optimal_device, print_device_info
from makeitdrumless.msst_integration.models import download_model_preset, MODEL_REGISTRY, get_base_cache_dir
from makeitdrumless.msst_integration.mps_patch import apply_all_patches


def separate_stems_msst(
    input_audio_path: str,
    output_folder: Optional[str] = None,
    model_preset: str = "scnet_large_starrytong",
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    device_name: str = "auto",
    force: bool = False,
) -> Dict[str, str]:
    """
    Separates an audio file into musical stems using MSST (Music-Source-Separation-Training).

    Args:
        input_audio_path: Path to input WAV/audio file.
        output_folder: Optional output directory for stems. If None, uses a temporary directory.
        model_preset: Name of model preset from MODEL_REGISTRY (e.g. 'scnet_xl', 'bs_conformer').
        config_path: Optional override for custom model YAML config.
        checkpoint_path: Optional override for custom model checkpoint (.ckpt).
        model_type: Architecture type (scnet, bs_roformer, mel_band_roformer, htdemucs, etc.).
        chunk_size: Custom chunk size in samples (e.g. 132300 for 3s, 264600 for 6s).
        overlap: Overlap factor for chunk blending (e.g. 2 or 4).
        device_name: 'auto', 'mps', 'cuda', or 'cpu'.
        force: If True, forces re-separation even if stems exist for this model.

    Returns:
        Dict mapping stem names (e.g. 'vocals', 'drums', 'bass', 'other') to their saved file paths.
    """
    # 1. Apply MPS & memory patches
    apply_all_patches()

    # 2. Try importing MSST utils
    try:
        from utils.settings import get_model_from_config
        from utils.model_utils import prefer_target_instrument, bigshifts_wrapper
        from utils.audio_utils import normalize_audio, denormalize_audio
    except ImportError as e:
        raise ImportError(
            "Music-Source-Separation-Training (MSST) package is not found. "
            "Please ensure music-source-separation-training is installed."
        ) from e

    # 3. Resolve Device (Apple Silicon MPS / CUDA / CPU)
    device = get_optimal_device(device_name)
    print_device_info(device)

    # 4. Resolve Model Checkpoint and Config
    if not config_path or not checkpoint_path:
        preset_type, dl_config, dl_ckpt = download_model_preset(model_preset)
        config_path = config_path or dl_config
        checkpoint_path = checkpoint_path or dl_ckpt
        model_type = model_type or preset_type

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")

    model_tag = model_preset if not checkpoint_path else os.path.splitext(os.path.basename(checkpoint_path))[0]
    clean_model_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_tag)

    track_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    
    if output_folder:
        track_output_dir = os.path.abspath(output_folder)
    else:
        track_output_dir = os.path.join(tempfile.gettempdir(), "makeitdrumless", "separated", f"{track_name}_{clean_model_tag}")
    os.makedirs(track_output_dir, exist_ok=True)

    # Check if stems are already separated with this model
    existing_stems = {}
    if not force and os.path.exists(track_output_dir):
        wav_files = [f for f in os.listdir(track_output_dir) if f.endswith(".wav")]
        if len(wav_files) >= 2:
            print(f"✅ Stems already separated with {model_preset} in {track_output_dir}")
            for wav in wav_files:
                stem_name = os.path.splitext(wav)[0]
                existing_stems[stem_name] = os.path.join(track_output_dir, wav)
            return existing_stems

    print(f"\n🎛️  Running MSST Separation using model: {os.path.basename(checkpoint_path)}")
    start_time = time.time()

    mpl_dir = os.path.join(tempfile.gettempdir(), "makeitdrumless", "mpl_config")
    os.environ["MPLCONFIGDIR"] = mpl_dir
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    os.makedirs(mpl_dir, exist_ok=True)

    # Instantiate model and load configuration
    if not model_type or model_type == "auto":
        model_type = "scnet"

    model, config = get_model_from_config(model_type, config_path)

    # Configure batch size, chunk size and overlap for memory efficiency on Apple Silicon
    if device.type == "mps":
        if hasattr(config, "inference"):
            config.inference.batch_size = 1

    if chunk_size is not None:
        if hasattr(config, "audio"):
            config.audio.chunk_size = chunk_size
        if hasattr(config, "inference"):
            config.inference.chunk_size = chunk_size
    elif device.type == "mps":
        # On Apple Silicon MPS, prevent excessive VRAM allocations for huge models
        current_chunk = getattr(getattr(config, "audio", None), "chunk_size", None) or getattr(getattr(config, "inference", {}), "chunk_size", None)
        if current_chunk and current_chunk > 132300:
            print(f"💡 Automatically using optimized chunk size for Apple Silicon: {current_chunk} -> 132300 (3s)")
            if hasattr(config, "audio"):
                config.audio.chunk_size = 132300
            if hasattr(config, "inference"):
                config.inference.chunk_size = 132300

    if overlap is not None:
        if hasattr(config, "inference"):
            config.inference.num_overlap = overlap

    # Check if config specifies a more specific model_type
    resolved_model_type = model_type
    if hasattr(config, "training") and hasattr(config.training, "model_type"):
        resolved_model_type = config.training.model_type

    # Load checkpoint weights
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_data, dict):
        if "state" in ckpt_data:
            ckpt_data = ckpt_data["state"]
        elif "state_dict" in ckpt_data:
            ckpt_data = ckpt_data["state_dict"]
        elif "model_state_dict" in ckpt_data:
            ckpt_data = ckpt_data["model_state_dict"]
    model.load_state_dict(ckpt_data)

    model = model.to(device)
    model.eval()

    sample_rate = getattr(config.audio, "sample_rate", 44100)
    instruments = prefer_target_instrument(config)[:]

    print(f"🎵 Loading audio '{os.path.basename(input_audio_path)}' (Sample rate: {sample_rate}Hz)...")
    mix, sr = librosa.load(input_audio_path, sr=sample_rate, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    print(f"⏳ Separating stems on {device.type.upper()}... (Instruments: {', '.join(instruments)})")

    # Normalize audio if requested in config
    norm_params = None
    if "normalize" in getattr(config, "inference", {}):
        if config.inference["normalize"] is True:
            mix, norm_params = normalize_audio(mix)

    # Perform separation using MSST bigshifts_wrapper
    with torch.no_grad():
        waveforms = bigshifts_wrapper(
            config,
            model,
            mix,
            device,
            model_type=resolved_model_type,
            pbar=True,
            bigshifts=1
        )

    # Save output stems
    saved_stems = {}
    for inst_name in instruments:
        if inst_name in waveforms:
            estimates = waveforms[inst_name]
            if norm_params is not None and "normalize" in getattr(config, "inference", {}):
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            out_file = os.path.join(track_output_dir, f"{inst_name}.wav")
            _save_waveform(estimates, sample_rate, out_file)
            saved_stems[inst_name] = out_file

    elapsed = time.time() - start_time
    print(f"⏱️  Separation finished in {elapsed:.2f} seconds.")
    if output_folder:
        print(f"📁 Separated stems saved to: {track_output_dir}")

    return saved_stems


def _save_waveform(wave, sample_rate: int, out_path: str):
    """Saves a 1D or 2D audio waveform numpy array / tensor to WAV file."""
    if isinstance(wave, torch.Tensor):
        wave = wave.detach().cpu().numpy()
    if len(wave.shape) == 2 and wave.shape[0] == 2:
        wave = wave.T
    elif len(wave.shape) == 3:
        wave = wave[0].T
    sf.write(out_path, wave, sample_rate, subtype="PCM_16")
