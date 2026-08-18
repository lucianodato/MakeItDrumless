import os
import sys
from pathlib import Path
from typing import Optional
from makeitdrumless.ytmusic.auth import get_ytmusic_auth_path, DEFAULT_AUTH_FILE


def upload_drumless_track(audio_path: str, auth_file: Optional[str] = None) -> bool:
    """
    Uploads a generated drumless audio track to YouTube Music personal library.

    :param audio_path: Path to the MP3/M4A/FLAC track to upload.
    :param auth_file: Optional custom path to YouTube Music authentication file.
    :return: True if upload succeeded or already present, False otherwise.
    """
    resolved_auth = get_ytmusic_auth_path(auth_file)
    if not resolved_auth:
        print("\n⚠️  YouTube Music credentials not found!")
        print(f"   Expected location: {DEFAULT_AUTH_FILE}")
        print("   To configure your account for automated uploads, run:")
        print("     makeitdrumless --setup-ytmusic\n")
        return False

    audio_file = Path(audio_path).resolve()
    if not audio_file.exists():
        print(f"❌ Cannot upload to YouTube Music: Audio file '{audio_path}' not found.")
        return False

    try:
        from ytmusicapi import YTMusic
    except ImportError:
        print("❌ 'ytmusicapi' is required for uploading to YouTube Music.")
        print("   Install it using: pip install ytmusicapi")
        return False

    print(f"\n☁️  Uploading to YouTube Music ({audio_file.name})...")
    try:
        yt = YTMusic(str(resolved_auth))
        response = yt.upload_song(str(audio_file))
        
        # ytmusicapi upload_song returns 'STATUS_SUCCEEDED' or response object / string
        if isinstance(response, str) and response.upper() == "STATUS_SUCCEEDED":
            print(f"✅ Successfully uploaded '{audio_file.name}' to YouTube Music Library!")
            return True
        elif response is None or (hasattr(response, "status_code") and response.status_code in (200, 201)):
            print(f"✅ Upload completed for '{audio_file.name}'!")
            return True
        else:
            print(f"ℹ️  YouTube Music upload response: {response}")
            return True

    except Exception as e:
        err_msg = str(e)
        if "409" in err_msg or "already exists" in err_msg.lower():
            print(f"ℹ️  Track already exists in your YouTube Music library.")
            return True
        print(f"❌ YouTube Music upload failed: {err_msg}")
        return False
