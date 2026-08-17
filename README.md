# MakeItDrumless 🥁

MakeItDrumless is a fast, automated CLI tool to generate high-quality **drumless backing tracks** from YouTube videos or local audio files to play along with your electronic drums (e-drums) or acoustic kit.

Powered by **[Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training)** with native **Apple Silicon GPU (`mps`)** acceleration on macOS.

---

## ✨ Features

- ⚡ **Apple Silicon Accelerated (`mps`)**: Native Metal Performance Shaders support on Mac (M1/M2/M3/M4) for blazing fast ~15-30s inference.
- 📦 **Zero Venv Friction**: Install globally via `uv tool` and run `makeitdrumless` from any terminal directory.
- 🎛️ **Multi-Architecture Support**: Pre-configured with top-performing source separation models (**SCNet XL**, **BS-Conformer**, **Mel-Band-RoFormer**, **MDX23C**, **HTDemucs**).
- 📥 **Automated Model Downloads**: Checkpoints (`.ckpt`) and configs (`.yaml`) are automatically fetched and cached in `~/.cache/makeitdrumless/`.
- 🎵 **Flexible Inputs**: Accepts YouTube URLs or local audio files (`.mp3`, `.wav`, `.flac`, `.m4a`).
- 🏷️ **ID3 Metadata & Tagging**: Automatically embeds track title, artist, and model information into the generated MP3.

---

## 🚀 Installation

### 1. Prerequisites

- **FFmpeg**: On macOS: `brew install ffmpeg`
- **uv** (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`)

### 2. Install as a Global CLI Tool

Clone the repository and install it globally with `uv tool`:

```bash
git clone https://github.com/lucianodato/MakeItDrumless.git
cd MakeItDrumless

# Install globally into your PATH:
uv tool install .

# Or install in editable mode for active development:
uv tool install --editable .
```

Now `makeitdrumless` is available everywhere in your terminal!

---

## 📖 Usage

### Basic Usage (YouTube URL)

Generates a drumless track using the default model (**SCNet Large by starrytong**) and Apple Silicon MPS acceleration. All outputs are saved to `~/Music/MakeItDrumless/<Song Title>/`:

```bash
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Process Local Audio Files

```bash
makeitdrumless "/path/to/my_song.mp3"
```

### Output Folder Structure

Every processed song automatically generates a dedicated folder in `~/Music/MakeItDrumless/`:

```text
~/Music/MakeItDrumless/
└── Bohemian Rhapsody/
    ├── Bohemian Rhapsody (Drumless).mp3       # Final drumless backing track
    ├── Bohemian Rhapsody (Original).wav       # Downloaded / converted original audio
    └── stems_scnet_large_starrytong/          # Extracted individual stems
        ├── vocals.wav
        ├── bass.wav
        ├── drums.wav
        └── other.wav
```

### Specify a Custom Output Directory

```bash
makeitdrumless "/path/to/my_song.mp3" -o ~/Desktop/MyTracks
```

### Force Re-Separation

If you want to re-run separation and overwrite existing stems:

```bash
makeitdrumless "/path/to/my_song.mp3" --force
```

### List Available Models

View all supported models, their descriptions, SDR metrics, and whether they are cached locally:

```bash
makeitdrumless --list-models
```

### Choose a Specific Model

```bash
# Using BS-Conformer
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model bs_conformer

# Using lightweight SCNet Small (~42MB)
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model scnet_small
```

---

## 🏆 Supported Model Presets

### 🍏 Apple MLX Models (Ultra-Fast Metal Acceleration on Apple Silicon via `demucs-mlx`)

| Preset Name | Underlying Model | Target Stems | SDR Quality | Notes |
|---|---|---|---|---|
| `mlx_demucs` *(MLX Default)* | HTDemucs v4 | 4 stems (`drums`, `vocals`, `bass`, `other`) | **9.40** | Blazing Fast (~50s full song), bit-exact |
| `mlx_demucs_ft` | HTDemucs v4 FT | 4 stems (`drums`, `vocals`, `bass`, `other`) | **10.02** | 4-model fine-tuned ensemble (matches SCNet XL) |
| `mlx_demucs_6s` | HTDemucs v4 6s | 6 stems | **8.50** | Drums, Bass, Vocals, Guitar, Piano, Other |
| `mlx_demucs_mmi` | HDemucs v3 MMI | 4 stems | **8.88** | Demucs v3 MMI model |

### 🔥 MSST PyTorch Models (SCNet & DualPathRNN Architectures)

| Preset Name | Architecture | Stems | Notes |
|---|---|---|---|
| `scnet_large_starrytong` *(MSST Default)* | SCNet | 4 stems | High SDR (9.70) by starrytong |
| `scnet_xl` | SCNet | 4 stems | State-of-the-Art quality (SDR 10.08) |
| `scnet_masked_xl` | SCNet | 4 stems | Noise reduction mask (SDR 9.82) |
| `scnet_large` | SCNet | 4 stems | Large SCNet architecture (SDR 9.32) |
| `scnet_small_starrytong` | SCNet | 4 stems | Compact model by starrytong (SDR 9.03) |
| `scnet_tran_small` | SCNet Tran | 4 stems | SCNet Transformer hybrid (SDR 8.92) |
| `scnet_small` | SCNet | 4 stems | Fast & ultra-lightweight (~42MB, SDR 8.81) |
| `bs_roformer` | BS-RoFormer | 4 stems | Band-Split RoFormer (SDR 9.65) |
| `bs_conformer` | BS-Conformer | 4 stems | Conformer-based 4-stem demixing (SDR 9.18) |
| `htdemucs4` | HTDemucs | 4 stems | Demucs v4 Hybrid Transformer (SDR 9.16) |
| `htdemucs4_6s` | HTDemucs | 6 stems | 6-stem separation (includes piano & guitar) |
| `demucs3_mmi` | Demucs | 4 stems | Demucs v3 MMI model (SDR 8.88) |

---

## 🔄 Keeping MSST Fork in Sync with Upstream

To sync your MSST fork with the latest upstream releases from ZFTurbo:

```bash
# Inside your MSST fork directory:
git remote add upstream https://github.com/ZFTurbo/Music-Source-Separation-Training.git
git fetch upstream
git merge upstream/main
git push origin main
```