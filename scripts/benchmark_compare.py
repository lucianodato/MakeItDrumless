"""
Benchmark script comparing MLX separation (via mlx-audio-separator) vs PyTorch SCNet (via MSST).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

def generate_synthetic_audio(path: Path, duration_sec: int = 15, sample_rate: int = 44100):
    """Generates a synthetic stereo audio file with kick, snare, and synth harmonics for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Sine sweeps simulating kick + bass
    kick_pulse = np.sin(2 * np.pi * 55 * t) * np.exp(-((t % 0.5) * 10))
    # High frequency bursts simulating hi-hat / snare
    noise = np.random.normal(0, 0.1, size=t.shape) * ((t % 0.25) < 0.05)
    # Melodic chords
    synth = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 554.37 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)
    
    audio_l = kick_pulse + noise + synth
    audio_r = kick_pulse + noise * 0.8 + synth
    audio = np.stack([audio_l, audio_r], axis=1).astype(np.float32)
    # Normalize
    audio /= np.max(np.abs(audio) + 1e-6)
    sf.write(str(path), audio, sample_rate)
    print(f"Generated {duration_sec}s test audio at: {path}")

def run_mlx_benchmark(input_file: Path, output_dir: Path, model_name: str = "htdemucs.yaml"):
    from mlx_audio_separator import Separator
    
    print("\n" + "="*50)
    print(f"🚀 Running MLX Separation ({model_name})")
    print("="*50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        sample_rate=44100,
        model_file_dir="/tmp/audio-separator-models/"
    )
    
    start_load = time.perf_counter()
    separator.load_model(model_name)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    start_infer = time.perf_counter()
    outputs = separator.separate(str(input_file))
    infer_time = time.perf_counter() - start_infer
    print(f"✅ MLX Inference completed in: {infer_time:.2f}s")
    return infer_time, outputs

def run_scnet_benchmark(input_file: Path, output_dir: Path, preset: str = "scnet_large_starrytong", overlap: int = 2):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from makeitdrumless.msst_integration.inference import separate_stems_msst
    
    print("\n" + "="*50)
    print(f"🔥 Running PyTorch MPS SCNet ({preset}, overlap={overlap})")
    print("="*50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    start_infer = time.perf_counter()
    stems = separate_stems_msst(
        input_audio_path=str(input_file),
        output_folder=str(output_dir),
        model_preset=preset,
        overlap=overlap,
        chunk_size=264600, # 6s chunk
        device_name="mps",
        force=True,
    )
    infer_time = time.perf_counter() - start_infer
    print(f"✅ SCNet Inference completed in: {infer_time:.2f}s")
    return infer_time, stems

def main():
    test_dir = Path("/tmp/makeitdrumless_benchmark")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_audio = test_dir / "test_benchmark_15s.wav"
    
    if not test_audio.exists():
        generate_synthetic_audio(test_audio, duration_sec=15)
        
    results = {}
    
    # 1. MLX htdemucs (single model)
    try:
        t_mlx, _ = run_mlx_benchmark(test_audio, test_dir / "mlx_htdemucs", "htdemucs.yaml")
        results["MLX (htdemucs single)"] = t_mlx
    except Exception as e:
        print(f"MLX htdemucs failed: {e}")
        
    # 2. MLX htdemucs_ft (4-model ensemble)
    try:
        t_mlx_ft, _ = run_mlx_benchmark(test_audio, test_dir / "mlx_htdemucs_ft", "htdemucs_ft.yaml")
        results["MLX (htdemucs_ft ensemble)"] = t_mlx_ft
    except Exception as e:
        print(f"MLX htdemucs_ft failed: {e}")
        
    # 3. PyTorch SCNet Small
    try:
        t_scnet_small, _ = run_scnet_benchmark(test_audio, test_dir / "scnet_small", preset="scnet_small", overlap=2)
        results["PyTorch MPS (scnet_small)"] = t_scnet_small
    except Exception as e:
        print(f"SCNet small failed: {e}")

    # 4. PyTorch SCNet Large Starrytong (overlap 2)
    try:
        t_scnet_large, _ = run_scnet_benchmark(test_audio, test_dir / "scnet_large", preset="scnet_large_starrytong", overlap=2)
        results["PyTorch MPS (scnet_large, overlap=2)"] = t_scnet_large
    except Exception as e:
        print(f"SCNet large failed: {e}")
        
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS (15s Audio on M4):")
    print("="*60)
    for name, elapsed in results.items():
        speedup = 15.0 / elapsed
        print(f"• {name:<35}: {elapsed:>6.2f}s ({speedup:>4.1f}x real-time)")
    print("="*60)

if __name__ == "__main__":
    main()
