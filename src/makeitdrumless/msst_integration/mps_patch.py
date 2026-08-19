"""Runtime patches for MSST to enable Apple Silicon MPS acceleration, memory-safe LSTM batching, and robust kwargs filtering."""

import inspect
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def apply_all_patches():
    """Applies all Apple Silicon MPS and stability optimizations to MSST modules in memory."""
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
            batch_limit = 32
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
    """Patches MSST demix function to support torch.autocast cleanly and trigger cache clearing."""
    try:
        import utils.model_utils as mu

        if getattr(mu, "_makeitdrumless_patched", False):
            return

        original_demix = getattr(mu, "demix", None)
        if original_demix is None:
            return

        def patched_demix(config, model, mix, device, model_type='scnet', pbar=False):
            dev_type = getattr(device, "type", str(device)).split(":")[0]
            use_amp = getattr(getattr(config, "inference", {}), "amp", True) if hasattr(config, "inference") else True

            # Ensure batch_size is 1 on MPS/CPU to prevent large contiguous allocations
            if dev_type in ["mps", "cpu"] and hasattr(config, "inference"):
                config.inference.batch_size = 1

            # CUDA benefits greatly from float16 AMP; on MPS/CPU, native float32 is fast and prevents RNN dtype mismatches
            if dev_type == "cuda" and use_amp:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            elif dev_type == "cpu" and use_amp:
                autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
            else:
                # MPS runs natively in float32 for maximum stability and speed
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

            with autocast_ctx:
                with torch.inference_mode():
                    result = original_demix(config, model, mix, device, model_type=model_type, pbar=pbar)

            if dev_type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif dev_type == "cuda" and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

            return result

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
