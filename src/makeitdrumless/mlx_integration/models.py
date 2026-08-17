"""Curated model registry and metadata for Apple MLX-native Demucs separation."""

from typing import Dict, Any

MLX_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlx_demucs": {
        "description": "Demucs v4 Hybrid Transformer - Blazing Fast (~23s full track), SDR 9.40",
        "demucs_model": "htdemucs",
        "stems": ["drums", "vocals", "bass", "other"],
        "default": True,
    },
    "mlx_demucs_ft": {
        "description": "Demucs v4 Fine-Tuned (4 models ensemble) - SDR 10.02 (Matches SCNet XL)",
        "demucs_model": "htdemucs_ft",
        "stems": ["drums", "vocals", "bass", "other"],
    },
    "mlx_demucs_6s": {
        "description": "Demucs v4 6-Stem - Drums, Bass, Vocals, Guitar, Piano, Other",
        "demucs_model": "htdemucs_6s",
        "stems": ["drums", "vocals", "bass", "guitar", "piano", "other"],
    },
    "mlx_demucs_mmi": {
        "description": "Demucs v3 MMI Model - Fast 4-stem separation, SDR 8.88",
        "demucs_model": "hdemucs_mmi",
        "stems": ["drums", "vocals", "bass", "other"],
    },
}


def is_mlx_model(model_name: str) -> bool:
    """Returns True if the given model name is registered as an MLX model."""
    return model_name in MLX_MODEL_REGISTRY


def get_mlx_model_info(model_name: str) -> Dict[str, Any]:
    """Retrieves metadata for an MLX model preset."""
    if model_name not in MLX_MODEL_REGISTRY:
        raise ValueError(f"Unknown MLX model '{model_name}'. Available MLX models: {', '.join(MLX_MODEL_REGISTRY.keys())}")
    return MLX_MODEL_REGISTRY[model_name]


def list_available_mlx_models():
    """Prints a formatted summary table of available MLX models."""
    print("\n" + "=" * 85)
    print(f"{'MLX MODEL PRESET':<25} | {'UNDERLYING MODEL':<18} | {'DESCRIPTION'}")
    print("=" * 85)
    for name, info in MLX_MODEL_REGISTRY.items():
        default_tag = " (MLX Default)" if info.get("default") else ""
        print(f"{name + default_tag:<25} | {info['demucs_model']:<18} | {info['description']}")
    print("=" * 85 + "\n")
