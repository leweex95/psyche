from pathlib import Path
import os
import sys


class EncryptionError(RuntimeError):
    pass


def load_or_create_key(key_path: Path) -> bytes:
    """Return a SQLCipher key protected by the current Windows user's DPAPI."""
    if sys.platform != "win32":
        raise EncryptionError("Encrypted storage currently requires Windows DPAPI.")
    try:
        import win32crypt
        if key_path.exists():
            return win32crypt.CryptUnprotectData(key_path.read_bytes(), None, None, None, 0)[1]
        key = os.urandom(32)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        protected = win32crypt.CryptProtectData(key, "Psyche database key", None, None, None, 0)
        with key_path.open("xb") as handle:
            handle.write(protected)
        return key
    except Exception as exc:
        raise EncryptionError("Could not initialize DPAPI-protected database encryption.") from exc


def sqlcipher_module():
    try:
        from sqlcipher3 import dbapi2
        return dbapi2
    except ImportError as exc:
        raise EncryptionError("SQLCipher is unavailable; install Psyche on Windows with its declared dependencies.") from exc
