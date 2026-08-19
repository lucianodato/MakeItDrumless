import os
import inspect
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def apply_all_patches():
    """Applies all Apple Silicon MPS and stability optimizations to MSST modules in memory."""
    import sys
    local_msst = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "msst"))
    if os.path.exists(local_msst) and local_msst not in sys.path:
        sys.path.insert(0, local_msst)

    # Ensure optimal MPS memory allocator limits and PyTorch fallback
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch is None:
        return
    patch_dualpath_rnn()
    patch_demix_mps()
    patch_settings_kwargs()


def patch_dualpath_rnn():
    """Patches SCNet DualPathRNN to process sequences in smaller sub-batches to prevent MPS VRAM spikes and handle dtype casting."""
    try:
        from models.scnet.separation import DualPathRNN

        # Only patch if not already patched
        if getattr(DualPathRNN, "_makeitdrumless_patched", False):
            return

        def patched_forward(self, x):
            B, C, F, T = x.shape

            # Process dual-path rnn
            original_x = x
            # Frequency-path
            x = self.norm_layers[0](x)
            x = x.transpose(1, 3).contiguous().view(B * T, F, C)
            
            # Sub-batch LSTM processing to prevent massive workspace allocations on MPS/GPU
            batch_limit = 64
            lstm0_dtype = self.lstm_layers[0].weight_ih_l0.dtype
            if x.dtype != lstm0_dtype:
                x = x.to(lstm0_dtype)

            if x.shape[0] > batch_limit:
                x_chunks = []
                for b_idx in range(0, x.shape[0], batch_limit):
                    sub_out, _ = self.lstm_layers[0](x[b_idx:b_idx + batch_limit])
                    x_chunks.append(sub_out)
                x = torch.cat(x_chunks, dim=0)
            else:
                x, _ = self.lstm_layers[0](x)

            lin0_dtype = self.linear_layers[0].weight.dtype
            if x.dtype != lin0_dtype:
                x = x.to(lin0_dtype)
            x = self.linear_layers[0](x)
            x = x.view(B, T, F, C).transpose(1, 3)
            x = x.to(original_x.dtype) + original_x

            original_x = x
            # Time-path
            x = self.norm_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
            
            lstm1_dtype = self.lstm_layers[1].weight_ih_l0.dtype
            if x.dtype != lst1_dtype if (lst1_dtype := lstm1_dtype) else False:
                x = x.to(lstm1_dtype)

            if x.shape[0] > batch_limit:
                x_chunks = []
                for b_idx in range(0, x.shape[0], batch_limit):
                    sub_out, _ = self.lstm_layers[1](x[b_idx:b_idx + batch_limit])
                    x_chunks.append(sub_out)
                x = torch.cat(x_chunks, dim=0)
            else:
                x, _ = self.lstm_layers[1](x)

            lin1_dtype = self.linear_layers[1].weight.dtype
            if x.dtype != lin1_dtype:
                x = x.to(lin1_dtype)
            x = self.linear_layers[1](x)
            x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
            x = x.to(original_x.dtype) + original_x

            return x

        DualPathRNN.forward = patched_forward
        DualPathRNN._makeitdrumless_patched = True
    except Exception:
        pass


def patch_demix_mps():
    """Patches MSST demix function with standalone CPU accumulation, adaptive precision, and throttled cache clearing."""
    try:
        import utils.model_utils as mu
        import numpy as np
        from tqdm.auto import tqdm

        if getattr(mu, "_makeitdrumless_patched", False):
            return

        def patched_demix(config, model, mix, device, model_type='scnet', pbar=False):
            should_print = True
            try:
                import torch.distributed as dist
                should_print = not dist.is_initialized() or dist.get_rank() == 0
            except Exception:
                pass

            mix_tensor = torch.tensor(mix, dtype=torch.float32)

            if model_type == 'htdemucs':
                mode = 'demucs'
            else:
                mode = 'generic'

            if mode == 'demucs':
                chunk_size = config.training.samplerate * config.training.segment
                num_instruments = len(config.training.instruments)
                num_overlap = getattr(config.inference, "num_overlap", 2)
                step = chunk_size // num_overlap
            else:
                if hasattr(config, "inference") and 'chunk_size' in config.inference:
                    chunk_size = config.inference.chunk_size
                else:
                    chunk_size = getattr(getattr(config, "audio", None), "chunk_size", 132300)
                num_instruments = len(mu.prefer_target_instrument(config))
                num_overlap = getattr(config.inference, "num_overlap", 2)

                fade_size = chunk_size // 10
                step = chunk_size // num_overlap
                border = chunk_size - step
                length_init = mix_tensor.shape[-1]
                windowing_array = mu._getWindowingArray(chunk_size, fade_size)
                if length_init > 2 * border and border > 0:
                    mix_tensor = nn.functional.pad(mix_tensor, (border, border), mode="reflect")

            dev_type = getattr(device, "type", str(device)).split(":")[0]
            use_amp = getattr(getattr(config, "training", {}), "use_amp", True) if hasattr(config, "training") else True

            # For sequential LSTMs (SCNet), batch_size=1 minimizes Python loop overhead.
            # For parallel Transformers (RoFormer), batch_size=2 allows GPU saturation.
            is_rnn = any(k in str(model_type).lower() for k in ["scnet", "bandit", "demucs"])
            if dev_type == "mps":
                batch_size = getattr(getattr(config, "inference", None), "batch_size", 1)
                if is_rnn:
                    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
                elif use_amp:
                    autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=True)
                else:
                    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
            elif dev_type in ["cuda", "cpu"]:
                batch_size = getattr(getattr(config, "inference", None), "batch_size", 1)
                amp_dtype = torch.float16 if dev_type == "cuda" else torch.bfloat16
                autocast_ctx = torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp)
            else:
                batch_size = 1
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

            with autocast_ctx:
                with torch.inference_mode():
                    req_shape = (num_instruments,) + mix_tensor.shape
                    result = torch.zeros(req_shape, dtype=torch.float32, device="cpu")
                    counter = torch.zeros(req_shape, dtype=torch.float32, device="cpu")

                    i = 0
                    batch_count = 0
                    batch_data = []
                    batch_locations = []
                    if pbar and should_print:
                        progress_bar = tqdm(
                            total=mix_tensor.shape[1], desc="Processing audio chunks", leave=False
                        )
                    else:
                        progress_bar = None

                    while i < mix_tensor.shape[1]:
                        part = mix_tensor[:, i:i + chunk_size]
                        chunk_len = part.shape[-1]
                        if mode == "generic" and chunk_len > chunk_size // 2:
                            pad_mode = "reflect"
                        else:
                            pad_mode = "constant"
                        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                        batch_data.append(part)
                        batch_locations.append((i, chunk_len))
                        i += step

                        if len(batch_data) >= batch_size or i >= mix_tensor.shape[1]:
                            arr = torch.stack(batch_data, dim=0).to(device, non_blocking=True)
                            x = model(arr)
                            out_cpu = x.detach().cpu()

                            if mode == "generic":
                                window = windowing_array.clone()
                                if i - step == 0:
                                    window[:fade_size] = 1
                                elif i >= mix_tensor.shape[1]:
                                    window[-fade_size:] = 1

                            for j, (start, seg_len) in enumerate(batch_locations):
                                if mode == "generic":
                                    result[..., start:start + seg_len] += out_cpu[j, ..., :seg_len] * window[..., :seg_len]
                                    counter[..., start:start + seg_len] += window[..., :seg_len]
                                else:
                                    result[..., start:start + seg_len] += out_cpu[j, ..., :seg_len]
                                    counter[..., start:start + seg_len] += 1.0

                            batch_data.clear()
                            batch_locations.clear()
                            del arr
                            del x
                            del out_cpu
                            batch_count += 1
                            if batch_count % 16 == 0:
                                if dev_type == "mps" and hasattr(torch.mps, "empty_cache"):
                                    torch.mps.empty_cache()
                                elif dev_type == "cuda" and hasattr(torch.cuda, "empty_cache"):
                                    torch.cuda.empty_cache()

                        if progress_bar:
                            progress_bar.update(step)

                    if progress_bar:
                        progress_bar.close()

                    if dev_type == "mps" and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                    elif dev_type == "cuda" and hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()

                    estimated_sources = result / counter
                    estimated_sources = estimated_sources.cpu().numpy()
                    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                    if mode == "generic":
                        if length_init > 2 * border and border > 0:
                            estimated_sources = estimated_sources[..., border:-border]

            if mode == "demucs":
                instruments = config.training.instruments
            else:
                instruments = mu.prefer_target_instrument(config)

            ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

            if mode == "demucs" and num_instruments <= 1:
                return estimated_sources
            else:
                return ret_data

        mu.demix = patched_demix
        mu._makeitdrumless_patched = True
    except Exception:
        pass


def patch_settings_kwargs():
    """Patches MSST get_model_from_config to filter kwargs safely before instantiating architectures."""
    try:
        import utils.settings as us

        if getattr(us, "_makeitdrumless_patched", False):
            return

        original_get_model = getattr(us, "get_model_from_config", None)
        if original_get_model is None:
            return

        def _filter_model_kwargs(cls, kwargs):
            try:
                sig = inspect.signature(cls.__init__)
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if has_varkw:
                    return kwargs
                valid_keys = set(sig.parameters.keys())
                return {k: v for k, v in kwargs.items() if k in valid_keys}
            except Exception:
                return kwargs

        us._filter_model_kwargs = _filter_model_kwargs
        us._makeitdrumless_patched = True
    except Exception:
        pass
