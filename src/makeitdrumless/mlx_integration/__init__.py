"""Apple MLX native acceleration integration for MakeItDrumless."""

from makeitdrumless.mlx_integration.models import (
    MLX_MODEL_REGISTRY,
    list_available_mlx_models,
    is_mlx_model,
)
from makeitdrumless.mlx_integration.inference import separate_stems_mlx

__all__ = [
    "MLX_MODEL_REGISTRY",
    "list_available_mlx_models",
    "is_mlx_model",
    "separate_stems_mlx",
]
