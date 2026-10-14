# AGENTS.md

## Project
Python CLI package `makeitdrumless`: generates drumless backing tracks from YouTube/local audio using Music-Source-Separation-Training (MSST). Source package lives in `src/makeitdrumless`; editable entry point `makeitdrumless = makeitdrumless.main:main`.

## Commands
- Dev install: `uv tool install --editable .`
- Run app: `makeitdrumless <url-or-file> [flags]` (`--list-models`, `--model`, `--ensemble`, `--remove-audience`, `--force`, `-o`, `--upload-ytmusic`, `--setup-ytmusic`).
- Tests (unittest; tests insert `src/` and sometimes `msst/` into `sys.path` themselves): `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- Single file: `.venv/bin/python -m unittest tests.test_ensemble_caching`
- Single test: `.venv/bin/python -m unittest tests.test_ensemble_caching.TestEnsembleCaching.test_normalize_preset_name`
- MLX tests (`tests.test_mlx_inference`) are auto-skipped unless running on Apple Silicon with `mlx` installed.

## Structure
- `src/makeitdrumless/`: package source. Public modules by concern: `audio/`, `ffmpeg/`, `msst_integration/`, `ytmusic/`, `cli_utils/`, `main.py`.
- `msst/`: gitignored local MSST fork. Used by `inference.py`/`mps_patch.py` (preferred over installed dependency) and by MLX tests. Not part of this repo; do not commit edits there.
- `tests/`: only unittest files; no pytest config.
- `build/`, `dist/`, `.venv/`: generated; avoid editing.

## Important quirks
- MSST resolution: inference inserts `repo_root/msst` into `sys.path` and imports `utils.settings` etc. If the local clone is absent, the installed `music-source-separation-training` dependency must provide them.
- ffmpeg is required on macOS/Linux (README prerequisite). `setup_ffmpeg_binary()` only auto-downloads a portable binary on Windows; otherwise it exits if missing.
- Device selection: `get_optimal_device("auto")` prefers MLX, then MPS, then CUDA, then CPU. Mel-band RoFormer/Bandit/Mamba are not MLX-native and fall back to MPS.
- Signal handling: `main.py` registers `_emergency_cleanup`, which kills the whole process group on Ctrl+C/SIGTERM. Long-running inference is normally terminated abruptly.
- Caching: existing non-empty stem WAVs in `stems_*` dirs are reused unless `--force`; tests rely on this cache-hit behavior.
- User data/cache: config auth `~/.config/makeitdrumless/`, model cache `~/.cache/makeitdrumless/`, outputs default `~/Music/MakeItDrumless/`.
- No lint/typecheck configuration exists; run the unittest suite as the verification step.