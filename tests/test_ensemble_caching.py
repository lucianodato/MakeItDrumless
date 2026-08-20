import os
import sys
import tempfile
import unittest
import wave
import struct
from pathlib import Path

# Ensure src/ is in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def write_dummy_wav(path, duration_samples=1000, num_channels=2, sample_rate=44100):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, 'w') as f:
        f.setnchannels(num_channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        data = struct.pack('<' + ('h' * (duration_samples * num_channels)), *([0] * (duration_samples * num_channels)))
        f.writeframes(data)

from makeitdrumless.msst_integration.models import normalize_preset_name, MODEL_REGISTRY
from makeitdrumless.msst_integration.inference import separate_stems_msst
from makeitdrumless.audio.processing import ensemble_stems
from makeitdrumless.audio.downloader import (
    clean_audio_title,
    parse_artist_title,
    get_audio_input,
)


class TestEnsembleCaching(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_preset_name(self):
        self.assertEqual(normalize_preset_name("bs_roformer"), "bs_roformer")
        self.assertEqual(normalize_preset_name("bs-roformer"), "bs_roformer")
        self.assertEqual(normalize_preset_name("BS_ROFORMER"), "bs_roformer")
        self.assertEqual(normalize_preset_name("  scnet-large-starrytong  "), "scnet_large_starrytong")
        self.assertEqual(normalize_preset_name("custom_preset"), "custom_preset")

    def test_clean_audio_title_and_parse(self):
        artist, title = parse_artist_title("Queen - Bohemian Rhapsody")
        self.assertEqual(artist, "Queen")
        self.assertEqual(title, "Bohemian Rhapsody")

        self.assertEqual(clean_audio_title("Song (Original).wav"), "Song")
        self.assertEqual(clean_audio_title("Song (Drumless).mp3"), "Song")
        self.assertEqual(clean_audio_title("/path/to/Song (Original).wav"), "Song")

    def test_separate_stems_msst_early_cache_hit(self):
        # Create fake track dir with fake stems
        stems_dir = os.path.join(self.base_dir, "stems_bs_roformer")
        drums_path = os.path.join(stems_dir, "drums.wav")
        other_path = os.path.join(stems_dir, "other.wav")
        write_dummy_wav(drums_path)
        write_dummy_wav(other_path)

        fake_input_audio = os.path.join(self.base_dir, "song.wav")
        write_dummy_wav(fake_input_audio)

        # Call separate_stems_msst - should hit cache immediately without model download/inference
        result = separate_stems_msst(
            input_audio_path=fake_input_audio,
            output_folder=stems_dir,
            model_preset="bs_roformer",
            force=False,
        )

        self.assertIn("drums", result)
        self.assertIn("other", result)
        self.assertEqual(os.path.abspath(result["drums"]), os.path.abspath(drums_path))
        self.assertEqual(os.path.abspath(result["other"]), os.path.abspath(other_path))

    def test_ensemble_stems_caching(self):
        ens_dir = os.path.join(self.base_dir, "stems_ensemble_bs_roformer_scnet_large_starrytong")
        ens_drums_path = os.path.join(ens_dir, "drums.wav")
        ens_vocals_path = os.path.join(ens_dir, "vocals.wav")
        write_dummy_wav(ens_drums_path)
        write_dummy_wav(ens_vocals_path)

        # When force=False, it should return existing stems
        res = ensemble_stems(
            stems_list=[{"drums": "dummy.wav"}],
            output_dir=ens_dir,
            force=False,
        )
        self.assertIn("drums", res)
        self.assertIn("vocals", res)
        self.assertEqual(os.path.abspath(res["drums"]), os.path.abspath(ens_drums_path))

    def test_get_audio_input_with_directory(self):
        song_dir = os.path.join(self.base_dir, "MySong")
        orig_wav = os.path.join(song_dir, "MySong (Original).wav")
        write_dummy_wav(orig_wav)

        # Pass directory directly
        wav_path, info = get_audio_input(song_dir, output_folder=self.base_dir)
        self.assertEqual(os.path.abspath(wav_path), os.path.abspath(orig_wav))
        self.assertEqual(info["title"], "MySong")

    def test_get_audio_input_with_original_wav(self):
        song_dir = os.path.join(self.base_dir, "MySong")
        orig_wav = os.path.join(song_dir, "MySong (Original).wav")
        write_dummy_wav(orig_wav)

        # Pass the (Original).wav file directly
        wav_path, info = get_audio_input(orig_wav, output_folder=self.base_dir)
        self.assertEqual(os.path.abspath(wav_path), os.path.abspath(orig_wav))
        self.assertEqual(info["title"], "MySong")

    def test_multi_model_ensemble_flow(self):
        # Setup track dir with 2 models previously separated
        track_dir = os.path.join(self.base_dir, "RockSong")
        song_wav = os.path.join(track_dir, "RockSong (Original).wav")
        write_dummy_wav(song_wav)

        m1_dir = os.path.join(track_dir, "stems_bs_roformer")
        write_dummy_wav(os.path.join(m1_dir, "drums.wav"))
        write_dummy_wav(os.path.join(m1_dir, "other.wav"))

        m2_dir = os.path.join(track_dir, "stems_scnet_large_starrytong")
        write_dummy_wav(os.path.join(m2_dir, "drums.wav"))
        write_dummy_wav(os.path.join(m2_dir, "vocals.wav"))
        write_dummy_wav(os.path.join(m2_dir, "bass.wav"))
        write_dummy_wav(os.path.join(m2_dir, "other.wav"))

        # In ensemble loop, both models should hit cache without error
        ensemble_models = ["bs_roformer", "scnet_large_starrytong"]
        stems_list = []
        for m_name in ensemble_models:
            norm_name = normalize_preset_name(m_name)
            m_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in norm_name)
            m_stems_dir = os.path.join(track_dir, f"stems_{m_tag}")

            m_stems = separate_stems_msst(
                input_audio_path=song_wav,
                output_folder=m_stems_dir,
                model_preset=norm_name,
                force=False,
            )
            stems_list.append(m_stems)

        self.assertEqual(len(stems_list), 2)
        self.assertIn("drums", stems_list[0])
        self.assertIn("vocals", stems_list[1])

        # Pre-create blended ensemble stems
        ens_tag = "_".join(ensemble_models)
        ens_dir = os.path.join(track_dir, f"stems_ensemble_{ens_tag}")
        write_dummy_wav(os.path.join(ens_dir, "drums.wav"))
        write_dummy_wav(os.path.join(ens_dir, "other.wav"))

        blended = ensemble_stems(stems_list, output_dir=ens_dir, force=False)
        self.assertIn("drums", blended)
        self.assertEqual(blended["drums"], os.path.join(ens_dir, "drums.wav"))

    def test_mel_band_roformer_crowd_registry_and_normalization(self):
        self.assertEqual(normalize_preset_name("mel_band_roformer_crowd"), "mel_band_roformer_crowd")
        self.assertEqual(normalize_preset_name("mel-band-roformer-crowd"), "mel_band_roformer_crowd")
        self.assertIn("mel_band_roformer_crowd", MODEL_REGISTRY)
        crowd_entry = MODEL_REGISTRY["mel_band_roformer_crowd"]
        self.assertEqual(crowd_entry["model_type"], "mel_band_roformer")
        self.assertIn("crowd", crowd_entry["stems"])
        self.assertIn("other", crowd_entry["stems"])

    def test_audience_removal_preprocessing_flow(self):
        track_dir = os.path.join(self.base_dir, "LiveConcert")
        song_wav = os.path.join(track_dir, "LiveConcert (Original).wav")
        write_dummy_wav(song_wav)

        # Pre-create crowd separation stems
        crowd_dir = os.path.join(track_dir, "stems_audience_mel_band_roformer_crowd")
        crowd_path = os.path.join(crowd_dir, "crowd.wav")
        other_path = os.path.join(crowd_dir, "other.wav")
        write_dummy_wav(crowd_path)
        write_dummy_wav(other_path)

        # 1. Test crowd separation cache hit
        aud_stems = separate_stems_msst(
            input_audio_path=song_wav,
            output_folder=crowd_dir,
            model_preset="mel_band_roformer_crowd",
            force=False,
        )
        self.assertIn("crowd", aud_stems)
        self.assertIn("other", aud_stems)

        # 2. Test drum separation cache hit on decrowded audio
        drum_dir = os.path.join(track_dir, "stems_scnet_large_starrytong")
        write_dummy_wav(os.path.join(drum_dir, "drums.wav"))
        write_dummy_wav(os.path.join(drum_dir, "vocals.wav"))
        write_dummy_wav(os.path.join(drum_dir, "bass.wav"))
        write_dummy_wav(os.path.join(drum_dir, "other.wav"))

        drum_stems = separate_stems_msst(
            input_audio_path=other_path,
            output_folder=drum_dir,
            model_preset="scnet_large_starrytong",
            force=False,
        )
        self.assertIn("drums", drum_stems)

        # 3. Retain crowd in final stems dict
        drum_stems["crowd"] = aud_stems["crowd"]
        self.assertIn("crowd", drum_stems)
        self.assertIn("vocals", drum_stems)
        self.assertIn("drums", drum_stems)


if __name__ == "__main__":
    unittest.main()

