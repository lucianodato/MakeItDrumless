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


def patch_mlx_memory_management():
    """
    Patches demucs_mlx.apply_mlx.apply_model to force eager evaluation (mx.eval)
    and cache clearing (mx.clear_cache) between BagOfModels sub-models and shift passes.
    This prevents MLX's lazy computation graph from retaining all passes simultaneously,
    reducing memory usage from 60GB swap down to under 1GB RAM.
    """
    try:
        import demucs_mlx.apply_mlx as am
        import mlx.core as mx

        # Set strict cache limit so Metal allocator returns memory to macOS immediately
        try:
            mx.set_cache_limit(512 * 1024 * 1024)  # 512 MB cache limit
        except Exception:
            try:
                mx.metal.set_cache_limit(512 * 1024 * 1024)
            except Exception:
                pass

        if getattr(am, "_makeitdrumless_memory_patched", False):
            return

        orig_apply = am.apply_model

        def memory_efficient_apply_model(
            model,
            mix,
            shifts=1,
            split=True,
            overlap=0.25,
            transition_power=1.0,
            progress=False,
            num_workers=0,
            segment=None,
            batch_size=2,
            seed=None,
            _rng=None,
        ):
            from demucs_mlx.mlx_convert import BagOfModelsMLX
            import random
            from demucs_mlx.apply_mlx import TensorChunk, tensor_chunk

            if _rng is None:
                rng = random if seed is None else random.Random(int(seed))
            else:
                rng = _rng

            # --- BagOfModels Handling with immediate per-model evaluation ---
            if isinstance(model, BagOfModelsMLX):
                totals = [0.0] * len(model.sources)
                estimates = None
                min_length = None

                for sub_model, model_weights in zip(model.models, model.weights):
                    res = memory_efficient_apply_model(
                        sub_model, mix, shifts, split, overlap, transition_power,
                        progress, num_workers, segment, batch_size, seed=seed, _rng=rng
                    )
                    out = mx.array(res)
                    w = mx.array(model_weights, dtype=out.dtype).reshape(1, -1, 1, 1)
                    out = out * w
                    mx.eval(out)

                    for k, inst_weight in enumerate(model_weights):
                        totals[k] += float(inst_weight)

                    if min_length is None:
                        min_length = out.shape[-1]
                        estimates = out
                    else:
                        if out.shape[-1] < min_length:
                            min_length = out.shape[-1]
                            estimates = estimates[..., :min_length]
                        elif out.shape[-1] > min_length:
                            out = out[..., :min_length]
                        estimates = estimates + out
                    
                    # Eagerly evaluate accumulated estimate and clear allocator cache
                    mx.eval(estimates)
                    mx.clear_cache()

                denom = mx.array(totals, dtype=estimates.dtype).reshape(1, -1, 1, 1)
                estimates = estimates / denom
                mx.eval(estimates)
                mx.clear_cache()
                return estimates

            # --- Single Model shifts loop with per-shift evaluation ---
            mix_chunk = tensor_chunk(mix)
            batch, channels, length = mix_chunk.shape
            mix_dtype = mix_chunk.tensor.dtype

            if shifts:
                max_shift = int(0.5 * model.samplerate)
                padded_mix = mix_chunk.padded(length + 2 * max_shift)
                padded_chunk = TensorChunk(padded_mix)
                out = 0.0
                for _ in range(shifts):
                    offset = rng.randint(0, max_shift)
                    shifted = TensorChunk(padded_chunk, offset, length + max_shift - offset)
                    shifted_out = memory_efficient_apply_model(
                        model, shifted, 0, split, overlap, transition_power,
                        progress, num_workers, segment, batch_size, seed=seed, _rng=rng
                    )
                    mx.eval(shifted_out)
                    out = out + shifted_out[..., max_shift - offset:]
                    mx.eval(out)
                    mx.clear_cache()
                out = out / shifts
                mx.eval(out)
                mx.clear_cache()
                return out

            return orig_apply(
                model, mix, shifts=0, split=split, overlap=overlap,
                transition_power=transition_power, progress=progress,
                num_workers=num_workers, segment=segment, batch_size=batch_size,
                seed=seed, _rng=_rng
            )

        am.apply_model = memory_efficient_apply_model
        am._makeitdrumless_memory_patched = True
    except Exception as e:
        print(f"⚠️  Could not apply MLX memory patch: {e}")


def separate_stems_mlx(
    input_audio_path: str,
    output_folder: Optional[str] = None,
    model_preset: str = "mlx_demucs",
    overlap: Optional[float] = None,
    shifts: int = 0,
    batch_size: int = 2,
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
    patch_mlx_memory_management()

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
        batch_size=batch_size or 2,
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
