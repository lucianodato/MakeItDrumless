import platform

try:
    import torch
except ImportError:
    torch = None

try:
    import mlx.core as mx
    _MLX_BASE_AVAILABLE = platform.system() == "Darwin" and platform.machine() == "arm64"
except ImportError:
    mx = None
    _MLX_BASE_AVAILABLE = False


def is_mlx_supported() -> bool:
    """Returns True if Apple MLX is available and running on Apple Silicon."""
    return _MLX_BASE_AVAILABLE and mx is not None


def get_optimal_device(requested_device: str = "auto"):
    """
    Resolves the optimal device (or 'mlx' backend tag) for inference based on hardware
    availability and user preferences.

    Args:
        requested_device: One of 'auto', 'mlx', 'mps', 'cuda', or 'cpu'.

    Returns:
        'mlx' string tag or torch.device configured for the optimal target.
    """
    req = (requested_device or "auto").strip().lower()

    if req == "mlx":
        if is_mlx_supported():
            return "mlx"
        else:
            print("⚠️  MLX requested but 'mlx' is not installed in the active environment.")
            print("💡 To enable MLX acceleration, run: pip install mlx")
            print("➡️  Falling back to Apple Silicon GPU via PyTorch MPS...\n")
            if torch and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu") if torch else "cpu"

    if torch is None:
        if req in ["auto", "mlx"] and is_mlx_supported():
            return "mlx"
        raise ImportError(
            "PyTorch is not installed. Please install PyTorch to run neural network source separation."
        )

    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("⚠️  MPS requested but torch.backends.mps is not available. Falling back to CPU.")
            return torch.device("cpu")

    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("⚠️  CUDA requested but torch.cuda is not available. Falling back to CPU.")
            return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    # 'auto' mode: check MLX (Apple Silicon native Metal) -> MPS -> CUDA -> CPU
    if is_mlx_supported():
        return "mlx"

    if torch.backends.mps.is_available():
        try:
            # Quick allocation test to verify MPS is functional
            _ = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info(device):
    """Print user-friendly information about the active execution device."""
    if device is None:
        return
    dev_type = getattr(device, "type", str(device)).strip().lower()
    if dev_type == "mlx":
        print("⚡ Accelerated by Apple Silicon GPU via Apple MLX (Metal)")
    elif dev_type == "mps":
        print("⚡ Accelerated by Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif dev_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "CUDA"
        print(f"⚡ Accelerated by NVIDIA GPU: {gpu_name}")
    else:
        print("💻 Running inference on CPU (Hardware acceleration not detected)")

