import os
import sys
import unittest
import numpy as np
import soundfile as sf
import tempfile
import io
from contextlib import redirect_stdout

# Ensure source and local msst are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(repo_root, "src")
msst_path = os.path.join(repo_root, "msst")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if msst_path not in sys.path:
    sys.path.insert(0, msst_path)

from makeitdrumless.msst_integration.device import get_optimal_device, is_mlx_supported, print_device_info
from msst.utils.mlx_engine import is_mlx_available, can_run_on_mlx


@unittest.skipUnless(is_mlx_supported(), "Apple MLX is not installed in active environment")
class TestMLXInference(unittest.TestCase):

    def test_mlx_device_detection(self):
        """Test device resolver detects MLX on Apple Silicon."""
        self.assertTrue(is_mlx_supported())
        opt_device = get_optimal_device("auto")
        self.assertEqual(opt_device, "mlx")

        # Test print device info
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_device_info("mlx")
        output = buf.getvalue()
        self.assertIn("Apple MLX", output)

    def test_can_run_on_mlx_architectures(self):
        """Test supported vs unsupported architecture classification for clean-sheet native MLX."""
        # Supported natively
        can_roformer, _ = can_run_on_mlx("bs_roformer")
        self.assertTrue(can_roformer)
        can_scnet, _ = can_run_on_mlx("scnet")
        self.assertTrue(can_scnet)
        can_htdemucs, _ = can_run_on_mlx("htdemucs")
        self.assertTrue(can_htdemucs)
        can_demucs, _ = can_run_on_mlx("demucs")
        self.assertTrue(can_demucs)

        # Unsupported in clean-sheet MLX (should fallback to PyTorch MPS)
        can_melband, reason_melband = can_run_on_mlx("mel_band_roformer")
        self.assertFalse(can_melband)
        self.assertIn("mel", reason_melband.lower())

        can_bandit, reason_bandit = can_run_on_mlx("bandit")
        self.assertFalse(can_bandit)
        self.assertIn("bandit", reason_bandit.lower())

        can_mamba, _ = can_run_on_mlx("mamba")
        self.assertFalse(can_mamba)

    def test_unsupported_model_mlx_fallback_advertisement(self):
        """Verify that attempting to run an unsupported model on MLX explicitly logs fallback to MPS."""
        from makeitdrumless.msst_integration.inference import separate_stems_msst

        # Create temporary stereo WAV file
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_wav = os.path.join(tmpdir, "test_mix.wav")
            sample_rate = 44100
            t = np.linspace(0, 1.0, sample_rate, endpoint=False)
            tone = 0.1 * np.sin(2 * np.pi * 440 * t)
            stereo_audio = np.stack([tone, tone], axis=-1)
            sf.write(dummy_wav, stereo_audio, sample_rate)

            # Create a mock config for an unsupported model (e.g. bandit)
            cfg_path = os.path.join(tmpdir, "config_bandit.yaml")
            ckpt_path = os.path.join(tmpdir, "bandit.ckpt")
            with open(cfg_path, "w") as f:
                f.write("training:\n  model_type: bandit\n  instruments:\n    - drums\n    - other\n")
            with open(ckpt_path, "w") as f:
                f.write("dummy")

            buf = io.StringIO()
            with redirect_stdout(buf):
                # Request unsupported architecture with device="mlx" - should advertise fallback
                try:
                    _ = separate_stems_msst(
                        input_audio_path=dummy_wav,
                        output_folder=os.path.join(tmpdir, "out"),
                        config_path=cfg_path,
                        checkpoint_path=ckpt_path,
                        device_name="mlx",
                    )
                except Exception:
                    pass

            out_log = buf.getvalue()
            # Verify explicit user-facing fallback notices were printed
            self.assertIn("[MLX Notice]", out_log)
            self.assertIn("Falling back to Apple Silicon GPU via PyTorch MPS", out_log)

    def test_native_htdemucs_mlx_initialization(self):
        """Verify native HTDemucsMLX can be instantiated and run with mlx arrays."""
        import mlx.core as mx
        from msst.models.htdemucs_mlx import HTDemucsMLX

        model = HTDemucsMLX(
            sources=["drums", "bass", "other", "vocals"],
            audio_channels=2,
            channels=16,
            depth=2,
            nfft=512,
            t_layers=1,
            t_heads=2,
            kernel_size=8,
            stride=4,
        )
        dummy_audio = mx.zeros((1, 2, 2048))
        out = model(dummy_audio)
        self.assertEqual(out.shape, (1, 4, 2, 2048))

    def test_native_scnet_mlx_initialization(self):
        """Verify native SCNetMLX can be instantiated and run with mlx arrays."""
        import mlx.core as mx
        from msst.models.scnet.scnet_mlx import SCNetMLX

        model = SCNetMLX(
            sources=["drums", "bass", "other", "vocals"],
            audio_channels=2,
            dims=[4, 8, 16],
            nfft=512,
            hop_size=128,
            win_size=512,
            num_dplayer=2,
        )
        dummy_audio = mx.zeros((1, 2, 2048))
        out = model(dummy_audio)
        self.assertEqual(out.shape, (1, 4, 2, 2048))

    def test_native_bs_roformer_mlx_initialization(self):
        """Verify native BSRoformerMLX can be instantiated and run with mlx arrays."""
        import mlx.core as mx
        from msst.models.bs_roformer.bs_roformer_mlx import BSRoformerMLX

        model = BSRoformerMLX(
            dim=64,
            depth=1,
            stereo=True,
            num_stems=2,
            time_transformer_depth=1,
            freq_transformer_depth=1,
            dim_head=32,
            heads=2,
            stft_n_fft=512,
            stft_hop_length=128,
            stft_win_length=512,
            freqs_per_bands=(257,),
        )
        dummy_audio = mx.zeros((1, 2, 4096))
        out = model(dummy_audio)
        self.assertEqual(out.shape, (1, 2, 2, 4096))


if __name__ == "__main__":
    unittest.main()
