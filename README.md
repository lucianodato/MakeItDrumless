# MakeItDrumless

This project allows you to generate drumless tracks from YouTube videos using the SCNet model.

## Project Structure

```
MakeItDrumless/
├── .git/
├── .gitignore
├── README.md
├── requirements.txt
├── checkpoints/ (SCNet model checkpoints)
├── ffmpeg/ (Portable FFmpeg binaries)
├── SCNet/ (SCNet repository - external, for inference)
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── downloader.py (Handles YouTube audio download)
│   │   └── processing.py (Handles audio mixing and metadata)
│   ├── ffmpeg/
│   │   ├── __init__.py
│   │   └── manager.py (Manages FFmpeg installation)
│   ├── scnet_integration/
│   │   ├── __init__.py
│   │   ├── inference.py (Integrates SCNet inference logic)
│   │   └── repo_manager.py (Manages SCNet repository and checkpoints)
│   └── utils/
│       ├── __init__.py
│       └── spinner.py (Utility for console spinner)
└── temp/ (Temporary files)
└── output/ (Generated drumless tracks)
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/MakeItDrumless.git
    cd MakeItDrumless
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Initial Setup (FFmpeg and SCNet Model):**

    The first time you run `main.py`, it will automatically:
    *   Download and set up a portable FFmpeg (Windows only, otherwise requires manual installation).
    *   Clone the SCNet repository into the `SCNet/` directory.
    *   Check for the SCNet model checkpoint (`SCNet-large.th`) and configuration file (`config.yaml`) in the `checkpoints/` directory. You will be prompted to manually download the checkpoint if it's missing.

## Usage

To generate a drumless track from a YouTube video, run the `main.py` script with the YouTube URL as an argument:

```bash
.venv\Scripts\python.exe src/main.py "<YouTube_URL>"
```

**Example:**

```bash
.venv\Scripts\python.exe src/main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Output

The generated drumless track (MP3 format) will be saved in the `output/` directory. Separated stems (vocals, bass, other) will be saved in `separated_scnet/<video_title>/`.

## Troubleshooting

*   **FFmpeg not found:** If you are not on Windows, ensure FFmpeg is installed and added to your system's PATH.
*   **SCNet Checkpoint Missing:** Follow the instructions in the console to manually download the `SCNet-large.th` file.
*   **Dependency Issues:** Ensure all packages in `requirements.txt` are installed.