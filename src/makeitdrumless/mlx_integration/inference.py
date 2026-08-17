"""Native Apple MLX Demucs inference engine for MakeItDrumless."""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import soundfile as sf

from makeitdrumless.mlx_integration.models import MLX_MODEL_REGISTRY, get_mlx_model_info


def separate_stems_mlx(
    input_audio_path: str,
    output_folder: Optional[str] = None,
    model_preset: str = "mlx_demucs",
    overlap: Optional[float] = None,
    shifts: int = 0,
    batch_size: int = 8,
    force: bool = False,
) -> Dict[str, str]:
    """
    Separates an audio file into musical stems using dedicated Apple MLX Demucs.

    Args:
        input_audio_path: Path to input WAV/MP3 audio file.
        output_folder: Destination directory for stems.
        model_preset: Name of MLX model preset ('mlx_demucs', 'mlx_demucs_ft', 'mlx_demucs_6s', etc.)
        overlap: Chunk overlap ratio in [0, 1) (e.g. 0.25)
        shifts: Number of random shift passes (default 0 for fastest bit-exact inference)
        batch_size: Batch size for chunk inference on Metal GPU (default 8)
        force: Force re-separation even if files exist

    Returns:
        Dict mapping clean stem names (e.g. 'drums', 'vocals', 'bass', 'other') to their saved file paths.
    """
    try:
        from demucs_mlx import Separator
    except ImportError:
        raise ImportError(
            "demucs-mlx is not installed. Please install it with: pip install demucs-mlx"
        )

    info = get_mlx_model_info(model_preset)
    demucs_model_name = info["demucs_model"]

    if output_folder:
        track_output_dir = os.path.abspath(output_folder)
    else:
        track_name = Path(input_audio_path).stem
        track_output_dir = os.path.join(tempfile.gettempdir(), "makeitdrumless", f"stems_{model_preset}")
    os.makedirs(track_output_dir, exist_ok=True)

    # Check cache if not forcing
    if not force:
        existing_files = [f for f in os.listdir(track_output_dir) if f.endswith(".wav")]
        if len(existing_files) >= 2:
            print(f"✅ Stems already separated with {model_preset} in {track_output_dir}")
            stems_dict = {}
            for stem in info["stems"]:
                candidate = os.path.join(track_output_dir, f"{stem}.wav")
                if os.path.exists(candidate):
                    stems_dict[stem] = candidate
            if len(stems_dict) >= 2:
                return stems_dict

    print(f"\n🚀 Running Apple MLX Separation using model: {model_preset} ({demucs_model_name})")
    print(f"   Target Quality: {info['description']}")

    # Setup Demucs MLX separator
    overlap_val = float(overlap) if overlap is not None else 0.25
    if overlap_val >= 1.0:
        # Convert integer overlap factor like 2 to fraction (0.25 or 0.5)
        overlap_val = 0.5 if overlap_val == 2 else 0.25

    separator = Separator(
        model=demucs_model_name,
        shifts=shifts,
        overlap=overlap_val,
        split=True,
        batch_size=batch_size or 8,
        progress=True,
    )

    start_time = time.time()
    _, stems = separator.separate_audio_file(input_audio_path)
    elapsed = time.time() - start_time
    print(f"⏱️  MLX Separation finished in {elapsed:.2f} seconds.")

    stems_dict: Dict[str, str] = {}
    sr = separator.samplerate
    from demucs_mlx import save_audio
    for stem_name, stem_tensor in stems.items():
        stem_clean = stem_name.lower().strip()
        stem_path = os.path.join(track_output_dir, f"{stem_clean}.wav")
        os.makedirs(os.path.dirname(os.path.abspath(stem_path)), exist_ok=True)
        save_audio(stem_tensor, stem_path, samplerate=sr)
        stems_dict[stem_clean] = stem_path

    return stems_dict
