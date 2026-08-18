import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "makeitdrumless"
DEFAULT_AUTH_FILE = DEFAULT_CONFIG_DIR / "ytmusic_auth.json"


def get_ytmusic_auth_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Returns the resolved path to the YouTube Music authentication file if it exists, else None."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists():
            return p
        return None

    # Check default config location
    if DEFAULT_AUTH_FILE.exists():
        return DEFAULT_AUTH_FILE

    # Check local directory as fallback
    local_p = Path("browser.json")
    if local_p.exists():
        return local_p.resolve()

    local_oauth = Path("oauth.json")
    if local_oauth.exists():
        return local_oauth.resolve()

    return None


def get_clipboard_text() -> Optional[str]:
    """Tries to get text from the macOS/system clipboard."""
    try:
        import subprocess
        if sys.platform == "darwin":
            res = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
            return res.stdout
    except Exception:
        pass
    return None


def setup_ytmusic_auth(output_path: Optional[str] = None) -> bool:
    """
    Interactive setup wizard to configure YouTube Music authentication.
    Supports reading directly from clipboard (on macOS), reading from a text file, or typing.
    """
    try:
        from ytmusicapi import YTMusic
        from ytmusicapi.setup import setup_browser
    except ImportError:
        print("❌ 'ytmusicapi' is not installed. Please run: pip install ytmusicapi")
        return False

    auth_path = Path(output_path).expanduser().resolve() if output_path else DEFAULT_AUTH_FILE
    auth_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n🎧 === YouTube Music Authentication Setup === 🎧\n")
    print("To upload songs to your personal YouTube Music library, ytmusicapi requires your")
    print("YouTube Music session headers from your browser.\n")
    print("Quick Steps:")
    print("  1. Open https://music.youtube.com in your browser (logged in).")
    print("  2. Open Developer Tools (Cmd+Option+I on Mac) -> Network tab.")
    print("  3. Filter by `music.youtube.com` or click on any song / Library.")
    print("  4. Right click a `browse` or `next` request -> Copy -> 'Copy Request Headers'.\n")

    raw_headers = None

    # Method 1: Check clipboard automatically (especially helpful on macOS)
    clip_text = get_clipboard_text()
    if clip_text and ("cookie:" in clip_text.lower() or "authorization:" in clip_text.lower() or "user-agent:" in clip_text.lower()):
        print("📋 Detected request headers in your macOS clipboard!")
        try:
            choice = input("👉 Would you like to use the headers currently in your clipboard? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "y"

        if choice in ("", "y", "yes"):
            raw_headers = clip_text

    if not raw_headers:
        print("How would you like to provide the headers?")
        print("  [1] Read automatically from system clipboard (Copy headers to clipboard first!)")
        print("  [2] Read from a text file (e.g. headers.txt)")
        print("  [3] Paste manually into terminal")
        try:
            method = input("\nSelect option [1/2/3] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            method = "1"

        if method in ("", "1"):
            clip_text = get_clipboard_text()
            if not clip_text:
                print("❌ Clipboard is empty or could not be read. Please copy your headers and try again.")
                return False
            raw_headers = clip_text

        elif method == "2":
            try:
                file_input = input("Enter path to file containing headers: ").strip().strip("'\"")
                fpath = Path(file_input).expanduser().resolve()
                if not fpath.exists():
                    print(f"❌ File not found: {fpath}")
                    return False
                raw_headers = fpath.read_text(encoding="utf-8")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return False

        else:
            print("\nPaste your request headers below (press Ctrl+D or type EOF on a new line when done):")
            print("-" * 60)
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == "EOF":
                        break
                    lines.append(line)
                except EOFError:
                    break
            raw_headers = "\n".join(lines).strip()

    if not raw_headers or not raw_headers.strip():
        print("\n⚠️ No headers received. Setup cancelled.")
        return False

    try:
        setup_browser(filepath=str(auth_path), headers_raw=raw_headers.strip())
        
        # Test the auth
        yt = YTMusic(str(auth_path))
        print("\n✅ Authentication credentials generated and verified successfully!")
        print(f"📁 Saved to: {auth_path}")
        print("🎉 YouTube Music upload is now ready to use with `--upload-ytmusic`!\n")
        return True

    except Exception as e:
        print(f"\n❌ Error setting up YouTube Music authentication: {e}")
        print(f"💡 Tip: You can also manually create `{auth_path}` using ytmusicapi.")
        return False

