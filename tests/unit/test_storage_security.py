import pytest

from psyche.storage import database
from psyche.storage.encryption import EncryptionError


def test_database_initialization_fails_closed_when_key_setup_fails(tmp_path, monkeypatch):
    class UnexpectedDBAPI:
        @staticmethod
        def connect(_path):
            pytest.fail("database connection must not start without an encryption key")

    def fail_key_setup(_key_path):
        raise EncryptionError("key initialization failed")

    monkeypatch.setattr(database, "sqlcipher_module", lambda: UnexpectedDBAPI)
    monkeypatch.setattr(database, "load_or_create_key", fail_key_setup)

    with pytest.raises(EncryptionError, match="key initialization failed"):
        database.Database(tmp_path)

    assert not (tmp_path / "psyche.db").exists()
