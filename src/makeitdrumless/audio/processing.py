import os
from typing import Union, Dict, Optional
from pydub import AudioSegment
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, COMM

# Stem names to exclude when creating a drumless mix
DRUM_STEM_NAMES = {"drums", "drum", "kick", "snare", "hh", "toms", "cymbals", "percussion"}


def is_drum_stem(name: str) -> bool:
    clean = name.lower().replace("-", "_").replace(" ", "_")
    if clean.startswith("no_") or "no_drum" in clean or "nodrum" in clean:
        return False
    for drum_word in ["drum", "kick", "snare", "hh", "toms", "cymbals", "percussion", "hihat"]:
        if drum_word in clean:
            return True
    return False


def mix_stems_without_drums(
    stems_input: Union[str, Dict[str, str]],
    output_path: str,
) -> str:
    """
    Mixes all separated stems together EXCEPT drum-related stems to produce a drumless backing track.

    Args:
        stems_input: Either a directory containing separated stem WAV files,
                     or a dictionary mapping stem names to WAV file paths.
        output_path: Path where the resulting MP3 file will be saved.

    Returns:
        The output path of the generated drumless MP3.
    """
    stem_files = {}

    if isinstance(stems_input, dict):
        stem_files = stems_input
    elif isinstance(stems_input, str):
        if os.path.isdir(stems_input):
            for file in os.listdir(stems_input):
                if file.endswith(".wav"):
                    stem_name = os.path.splitext(file)[0].lower()
                    stem_files[stem_name] = os.path.join(stems_input, file)
        elif os.path.isfile(stems_input):
            stem_name = os.path.splitext(os.path.basename(stems_input))[0].lower()
            stem_files[stem_name] = stems_input

    # Filter out drum stems
    non_drum_stems = {
        name: path for name, path in stem_files.items()
        if not is_drum_stem(name) and os.path.exists(path)
    }

    if not non_drum_stems:
        print("⚠️  No non-drum stems found. Checking for fallback mix...")
        for name, path in stem_files.items():
            clean = name.lower()
            if not ("drum" in clean and not ("no" in clean)) and os.path.exists(path):
                non_drum_stems[name] = path

    if not non_drum_stems:
        print("❌ No valid stems found to create drumless track.")
        return ""

    # If exactly one non-drum stem exists (e.g. 2-stem model with 'other' or 'no_drums'),
    # bypass overlay mixing and directly export the pristine backing track
    if len(non_drum_stems) == 1:
        stem_name, file_path = next(iter(non_drum_stems.items()))
        print(f"⚡ Single backing stem detected ('{stem_name}'). Direct conversion to MP3 (bypassing stem overlay)...")
        try:
            seg = AudioSegment.from_wav(file_path)
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            seg = seg.normalize(headroom=0.1)
            seg.export(output_path, format="mp3", bitrate="320k")
            print(f"✅ Final drumless track saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"⚠️ Error exporting direct stem {stem_name}: {e}")

    print(f"🎚️  Mixing non-drum stems: {', '.join(non_drum_stems.keys())}...")
    mixed = None
    for stem_name, file_path in non_drum_stems.items():
        try:
            seg = AudioSegment.from_wav(file_path)
            mixed = seg if mixed is None else mixed.overlay(seg)
            print(f"  + Included stem: {stem_name}")
        except Exception as e:
            print(f"⚠️  Error loading stem {stem_name} ({file_path}): {e}")

    if mixed is not None:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        # Normalize to standard commercial listening volume with 0.1dB headroom
        mixed = mixed.normalize(headroom=0.1)
        mixed.export(output_path, format="mp3", bitrate="320k")
        print(f"✅ Final drumless track saved to: {output_path}")
        return output_path
    else:
        print("❌ Failed to overlay stems.")
        return ""


def set_mp3_metadata(mp3_path: str, info: Optional[dict], model_name: str = "MSST"):
    """Set title, artist, and comment metadata on an MP3 file using mutagen."""
    if not mp3_path or not os.path.exists(mp3_path) or info is None:
        return
    try:
        audio = MP3(mp3_path, ID3=ID3)
        if audio.tags is None:
            audio.add_tags()
        # Extract title and artist
        title = info.get("track") or info.get("title") or ""
        artist = info.get("artist") or info.get("uploader") or info.get("channel") or ""

        if not artist and ' - ' in title:
            parts = title.split(' - ', 1)
            artist, title = parts[0], parts[1]

        pretty_title = f"{title.strip()} (Drumless)" if title else "Drumless Version"
        audio.tags.add(TIT2(encoding=3, text=pretty_title))
        if artist:
            audio.tags.add(TPE1(encoding=3, text=artist.strip()))

        comment_text = f"Drumless backing track generated by MakeItDrumless ({model_name})"
        audio.tags.add(COMM(encoding=3, lang="eng", desc="", text=comment_text))
        audio.save()
        print(f"✅ Metadata set: Title='{pretty_title}', Artist='{artist.strip() if artist else 'Unknown'}'")
    except Exception as e:
        print(f"⚠️  Could not set metadata: {e}")


def ensemble_stems(
    stems_list: list,
    weights: Optional[list] = None,
    output_dir: str = "",
) -> Dict[str, str]:
    """
    Combines stem outputs from multiple models via weighted linear averaging.

    Args:
        stems_list: List of dictionaries mapping stem_name -> wav_file_path.
        weights: List of float weights for each model. Defaults to equal weights.
        output_dir: Output directory where the ensembled stems will be saved.

    Returns:
        Dict mapping stem names to the ensembled stem file paths.
    """
    import soundfile as sf
    import numpy as np

    if not stems_list:
        raise ValueError("stems_list cannot be empty.")

    n_models = len(stems_list)
    if weights is None:
        norm_weights = [1.0 / n_models] * n_models
    else:
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

    os.makedirs(output_dir, exist_ok=True)
    all_stem_names = sorted(list(set().union(*(s.keys() for s in stems_list))))

    print(f"\n🎛️  Blending ensemble of {n_models} models with weights: {[round(w, 2) for w in norm_weights]}...")
    ensembled_dict: Dict[str, str] = {}

    for stem_name in all_stem_names:
        stem_audios = []
        stem_weights = []
        sample_rate = 44100

        for model_idx, stems in enumerate(stems_list):
            if stem_name in stems and os.path.exists(stems[stem_name]):
                try:
                    data, sr = sf.read(stems[stem_name], dtype="float32")
                    sample_rate = sr
                    stem_audios.append(data)
                    stem_weights.append(norm_weights[model_idx])
                except Exception as e:
                    print(f"⚠️  Could not read stem {stem_name} from model {model_idx}: {e}")

        if not stem_audios:
            continue

        # Re-normalize weights if not all models produced this stem
        w_sum = sum(stem_weights)
        curr_weights = [w / w_sum for w in stem_weights]

        # Align lengths across models to the minimum length
        min_len = min(len(a) for a in stem_audios)
        blended = np.zeros_like(stem_audios[0][:min_len], dtype=np.float32)

        for w, audio in zip(curr_weights, stem_audios):
            blended += w * audio[:min_len]

        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(out_path, blended, sample_rate)
        ensembled_dict[stem_name] = out_path
        print(f"  + Ensembled stem: {stem_name} -> {out_path}")

    return ensembled_dict

