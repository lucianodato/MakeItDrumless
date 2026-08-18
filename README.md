# MakeItDrumless 🥁

MakeItDrumless is a fast, automated CLI tool to generate high-quality **drumless backing tracks** from YouTube videos or local audio files to play along with your electronic drums (e-drums) or acoustic kit.

Powered by **[Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training)** with native **Apple Silicon GPU (`mps`)** acceleration on macOS.

---

## ✨ Features

- ⚡ **Apple Silicon Accelerated (`mps`)**: Native Metal Performance Shaders support on Mac (M1/M2/M3/M4) for blazing fast inference.
- 📦 **Zero Venv Friction**: Install globally via `uv tool` and run `makeitdrumless` from any terminal directory.
- 🎛️ **Multi-Architecture Support**: Pre-configured with top-performing source separation models (**SCNet XL**, **BS-Conformer**, **Mel-Band-RoFormer**, **BS-RoFormer**, **HTDemucs**).
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

### Basic Usage

```bash
# 1. Generate drumless track with default model (SCNet Large):
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 2. Highest quality SCNet XL (SDR 10.08):
makeitdrumless "/path/to/song.mp3" --model scnet_xl

# 3. Band-Split RoFormer (SDR 9.65):
makeitdrumless "/path/to/song.mp3" --model bs_roformer

# 4. Multi-Model Ensemble (Blends SCNet Large + BS-RoFormer):
makeitdrumless "/path/to/song.mp3" --ensemble "scnet_large_starrytong,bs_roformer" --ensemble-weights "0.5,0.5"

# 5. Generate and automatically upload to YouTube Music library:
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --upload-ytmusic

# 6. List all available models and their download status:
makeitdrumless --list-models
```

---

## ☁️ YouTube Music Auto-Upload

You can automatically push your generated drumless backing tracks directly to your personal **YouTube Music library** (`Library > Uploads` tab).

### One-Time Setup:
Configure your YouTube Music account credentials:
```bash
makeitdrumless --setup-ytmusic
```
Follow the on-screen instructions to paste your browser session headers. Credentials will be securely saved to `~/.config/makeitdrumless/ytmusic_auth.json`.

### Uploading Tracks:
Add the `--upload-ytmusic` (or `-u`) flag when running `makeitdrumless`:
```bash
makeitdrumless "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --upload-ytmusic
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

### 🔥 MSST PyTorch Models (SCNet, RoFormer, Conformer & Demucs Architectures)

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