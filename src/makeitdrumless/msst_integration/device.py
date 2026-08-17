import torch


def get_optimal_device(requested_device: str = "auto") -> torch.device:
    """
    Resolves the optimal torch.device for inference based on hardware availability
    and user preferences.

    Args:
        requested_device: One of 'auto', 'mps', 'cuda', or 'cpu'.

    Returns:
        torch.device configured for the optimal or requested target.
    """
    req = (requested_device or "auto").strip().lower()

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

    # 'auto' mode: check MPS (macOS Apple Silicon) -> CUDA -> CPU
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


def print_device_info(device: torch.device):
    """Print user-friendly information about the active execution device."""
    if device.type == "mps":
        print("⚡ Accelerated by Apple Silicon GPU via Metal Performance Shaders (MPS)")
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        print(f"⚡ Accelerated by NVIDIA GPU: {gpu_name}")
    else:
        print("💻 Running inference on CPU (Hardware acceleration not detected)")
