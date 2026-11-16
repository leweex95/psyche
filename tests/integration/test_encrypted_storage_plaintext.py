import sqlite3
import sys

import pytest

from psyche.storage.database import Database


@pytest.mark.skipif(sys.platform != "win32", reason="Encrypted storage requires Windows DPAPI")
def test_encrypted_database_content_is_not_readable_as_plaintext_sqlite(tmp_path):
    marker = "private-storage-security-marker-7f1c9a"
    db = Database(tmp_path)
    session_id = db.create_session()
    db.add_message(session_id, "user", marker)
    db.close()

    database_path = tmp_path / "psyche.db"
    raw_bytes = database_path.read_bytes()
    assert marker.encode("utf-8") not in raw_bytes
    assert not raw_bytes.startswith(b"SQLite format 3\x00")

    plaintext_connection = sqlite3.connect(database_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            plaintext_connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
    finally:
        plaintext_connection.close()
