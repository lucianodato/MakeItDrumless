import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Built-in curated registry of high quality multi-stem models for drum isolation
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- SCNet Architectures ---
    "scnet_large_starrytong": {
        "description": "SCNet Large by starrytong (4 stems) - High quality SDR 9.70",
        "model_type": "scnet",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt",
        "default": True,
    },
    "scnet_xl": {
        "description": "SCNet XL IHF (4 stems) - State-of-the-Art quality, SDR 10.08",
        "model_type": "scnet",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
    },
    "scnet_masked_xl": {
        "description": "SCNet Masked XL IHF (4 stems) - Noise reduction mask, SDR 9.82",
        "model_type": "scnet_masked",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/config_musdb18_scnet_xl_ihf.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt",
    },
    "scnet_large": {
        "description": "SCNet Large (4 stems) - SDR 9.32",
        "model_type": "scnet",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/config_musdb18_scnet_large.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.8/model_scnet_sdr_9.3244.ckpt",
    },
    "scnet_small_starrytong": {
        "description": "SCNet Small by starrytong (4 stems) - SDR 9.03",
        "model_type": "scnet",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt",
    },
    "scnet_tran_small": {
        "description": "SCNet Transformer Small (4 stems) - SDR 8.92",
        "model_type": "scnet_tran",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.14/config_musdb18_scnet_tran.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.14/model_scnet_tran_sdr_8.9272.ckpt",
    },
    "scnet_small": {
        "description": "SCNet Masked Small (4 stems) - Fast & lightweight checkpoint (~42MB), SDR 8.81",
        "model_type": "scnet_masked",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.16/config_musdb18_scnet_small.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.16/model_scnet_masked_ep_156_sdr_8.8149.ckpt",
    },

    # --- RoFormer & Conformer Architectures ---
    "bs_mega_53stem_drums": {
        "description": "BS-RoFormer Mega 53-stem Drums (2 stems: drums, other) - MVSep SOTA drum extraction",
        "model_type": "bs_roformer",
        "stems": ["drums", "other"],
        "config_url": "https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems/resolve/main/v1/bs_mega_53stem_drums_mvsep_config.yaml",
        "checkpoint_url": "https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems/resolve/main/v1/bs_mega_53stem_drums_mvsep.ckpt",
    },
    "bs_drums2_xlancer": {
        "description": "BS-RoFormer Drums v2 by Xlance (2 stems: drums, other) - Punchy transient isolation",
        "model_type": "bs_roformer",
        "stems": ["drums", "other"],
        "config_url": "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums2_xlancer_config.yaml",
        "checkpoint_url": "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums2_xlancer.ckpt",
    },
    "bs_drums_gilliaaan": {
        "description": "BS-RoFormer Drums Duality by Gilliaaan (2 stems: drums, other) - High cymbal/hihat precision",
        "model_type": "bs_roformer",
        "stems": ["drums", "other"],
        "config_url": "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_gilliaaan_config.yaml",
        "checkpoint_url": "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_gilliaaan.ckpt",
    },
    "bs_roformer": {
        "description": "Band-Split RoFormer (4 stems: vocals, bass, drums, other) - SDR 9.65",
        "model_type": "bs_roformer",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
    },
    "bs_conformer": {
        "description": "BS Conformer Medium (4 stems: vocals, bass, drums, other) - SDR 9.18",
        "model_type": "bs_conformer",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.18/config_musdb18_bs_conformer_infer.yaml",
        "checkpoint_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.18/fused_model_bs_conformer_sdr_9.18.ckpt",
    },

    # --- Demucs & HTDemucs Architectures ---
    "htdemucs": {
        "description": "HTDemucs4 Hybrid Transformer (4 stems: vocals, bass, drums, other) - SDR 9.16",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    },
    "htdemucs_ft": {
        "description": "HTDemucs4 Hybrid Transformer (4 stems: vocals, bass, drums, other) - SDR 9.16",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    },
    "htdemucs4": {
        "description": "HTDemucs4 Hybrid Transformer (4 stems: vocals, bass, drums, other) - SDR 9.16",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_htdemucs.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    },
    "htdemucs4_6s": {
        "description": "HTDemucs4 (6 stems: vocals, bass, drums, other, piano, guitar)",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other", "piano", "guitar"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    },
    "htdemucs_6s": {
        "description": "HTDemucs4 (6 stems: vocals, bass, drums, other, piano, guitar)",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other", "piano", "guitar"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_htdemucs_6stems.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    },
    "demucs3_mmi": {
        "description": "Demucs3 MMI (4 stems: vocals, bass, drums, other) - SDR 8.88",
        "model_type": "htdemucs",
        "stems": ["vocals", "bass", "drums", "other"],
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_musdb18_demucs3_mmi.yaml",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
    },
}


def get_base_cache_dir() -> Path:
    """Returns the base cache directory for MakeItDrumless."""
    env_cache = os.environ.get("MAKEITDRUMLESS_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path.home() / ".cache" / "makeitdrumless"


def get_model_cache_dir(model_name: str) -> str:
    """Returns local storage path for a model's checkpoints and configuration."""
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
    cache_path = get_base_cache_dir() / "checkpoints" / clean_name
    return str(cache_path)


def download_file(url: str, dest_path: str, description: str = "Downloading"):
    """Downloads a file with streaming progress bar."""
    import requests
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_dest = dest_path + ".tmp"

    headers = {"User-Agent": "MakeItDrumless/1.0"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    if tqdm:
        with open(temp_dest, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)
    else:
        downloaded = 0
        with open(temp_dest, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                downloaded += size
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  {description}: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
        print()

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_dest, dest_path)


def normalize_preset_name(model_name: str) -> str:
    """Normalizes preset name for robust matching."""
    name = model_name.strip().lower().replace("-", "_")
    if name in MODEL_REGISTRY:
        return name
    # Check case-insensitive match
    for k in MODEL_REGISTRY:
        if k.lower() == name:
            return k
    return model_name.strip()


def is_model_downloaded(model_name: str) -> bool:
    """Checks whether both config and checkpoint files are present locally."""
    norm_name = normalize_preset_name(model_name)
    if norm_name not in MODEL_REGISTRY:
        return False
    entry = MODEL_REGISTRY[norm_name]
    cache_dir = get_model_cache_dir(norm_name)
    config_name = os.path.basename(entry["config_url"])
    ckpt_name = os.path.basename(entry["checkpoint_url"])
    config_path = os.path.join(cache_dir, config_name)
    ckpt_path = os.path.join(cache_dir, ckpt_name)
    return os.path.exists(config_path) and os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0


def fetch_github_release_models() -> Dict[str, Dict[str, Any]]:
    """Fetches all releases dynamically from ZFTurbo/Music-Source-Separation-Training GitHub API."""
    import requests
    url = "https://api.github.com/repos/ZFTurbo/Music-Source-Separation-Training/releases"
    headers = {"User-Agent": "MakeItDrumless/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}
        releases = r.json()
        dynamic_models = {}
        for rel in releases:
            tag = rel.get("tag_name", "")
            assets = rel.get("assets", [])
            configs = [a for a in assets if a["name"].endswith((".yaml", ".yml"))]
            ckpts = [a for a in assets if a["name"].endswith((".ckpt", ".th", ".pt", ".pth"))]

            for ckpt in ckpts:
                base_stem = ckpt["name"].rsplit(".", 1)[0]
                matched_config = None
                for cfg in configs:
                    if cfg["name"].startswith(base_stem) or base_stem.startswith(cfg["name"].rsplit(".", 1)[0]):
                        matched_config = cfg
                        break
                if not matched_config and configs:
                    matched_config = configs[0]

                if matched_config:
                    model_key = f"{tag}_{ckpt['name'].rsplit('.', 1)[0]}"
                    dynamic_models[model_key] = {
                        "description": f"{rel.get('name', tag)} - {ckpt['name']}",
                        "model_type": "auto",
                        "stems": ["stems"],
                        "config_url": matched_config["browser_download_url"],
                        "checkpoint_url": ckpt["browser_download_url"],
                    }
        return dynamic_models
    except Exception:
        return {}


def download_model_preset(model_name: str) -> Tuple[str, str, str]:
    """
    Downloads and prepares config and weights for the specified model preset.

    Returns:
        (model_type, config_path, checkpoint_path)
    """
    norm_name = normalize_preset_name(model_name)
    preset = MODEL_REGISTRY.get(norm_name)
    if not preset:
        dyn = fetch_github_release_models()
        if model_name in dyn:
            preset = dyn[model_name]
            norm_name = model_name
        elif norm_name in dyn:
            preset = dyn[norm_name]
        else:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model preset '{model_name}'. Available presets: {available}")

    cache_dir = get_model_cache_dir(norm_name)
    os.makedirs(cache_dir, exist_ok=True)

    config_filename = os.path.basename(preset["config_url"])
    ckpt_filename = os.path.basename(preset["checkpoint_url"])

    config_path = os.path.abspath(os.path.join(cache_dir, config_filename))
    checkpoint_path = os.path.abspath(os.path.join(cache_dir, ckpt_filename))

    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        print(f"📥 Downloading config for {norm_name}...")
        download_file(preset["config_url"], config_path, description=f"{norm_name} (Config)")
    else:
        print(f"✅ Config found: {config_filename}")

    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"📥 Downloading model weights for {norm_name} (this may take a few minutes)...")
        download_file(preset["checkpoint_url"], checkpoint_path, description=f"{norm_name} (Weights)")
    else:
        print(f"✅ Checkpoint found: {ckpt_filename}")

    return preset["model_type"], config_path, checkpoint_path


def list_available_models():
    """Prints a formatted summary table of available model presets and their status."""
    print("\n" + "=" * 80)
    print(f"{'MODEL PRESET':<25} | {'STATUS':<12} | {'TYPE':<12} | {'DESCRIPTION'}")
    print("=" * 80)
    for name, info in MODEL_REGISTRY.items():
        is_dl = is_model_downloaded(name)
        status_str = "🟢 Downloaded" if is_dl else "⚪ Available"
        default_tag = " (Default)" if info.get("default") else ""
        print(f"{name + default_tag:<25} | {status_str:<12} | {info['model_type']:<12} | {info['description']}")
    print("=" * 80 + "\n")
