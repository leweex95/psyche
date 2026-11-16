import sys
import json

import pytest

from psyche.storage.database import Database


@pytest.mark.skipif(sys.platform != "win32", reason="Encrypted storage requires Windows DPAPI")
def test_memory_review_correction_retrieval_and_deletion(tmp_path):
    db = Database(tmp_path)
    source = db.import_document("notes.md", "A private source", "unique-hash")
    memory_id = db.add_memory("I value creative work and close friendships.", "preference", source,
                              source_type="imported_document", source_span="char:0-44")
    assert len(db.list_memories("pending")) == 1
    stored = next(row for row in db.list_memories() if row[0] == memory_id)
    assert stored[8:] == ("imported_document", "char:0-44")
    assert db.set_memory_status(memory_id, "approved")
    assert db.search_memories("creative work")
    assert db.correct_memory(memory_id, "I value art and trusted friendships.")
    assert db.search_memories("trusted friendships")[0][2] == "I value art and trusted friendships."
    assert db.set_memory_status(memory_id, "deleted")
    assert db.search_memories("trusted friendships") == []
    db.close()
    db = Database(tmp_path)
    assert db.search_memories("trusted friendships") == []
    assert next(row for row in db.list_memories() if row[0] == memory_id)[6] == "deleted"
    db.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Encrypted storage requires Windows DPAPI")
def test_source_deletion_preserves_independently_supported_memory_then_purges_derivatives(tmp_path):
    db = Database(tmp_path)
    statement = "Taking a quiet walk helps me recover after difficult workdays."
    first_source = db.import_document("first.md", "First independent note.", "first-source-hash")
    second_source = db.import_document("second.md", "Second independent note.", "second-source-hash")
    memory_id = db.add_memory(
        statement, "coping_behavior", first_source, status="approved",
        source_type="imported_document", source_span="char:0-24",
    )
    linked_id, added, linked = db.add_memory_with_provenance(
        statement, "coping_behavior", second_source, "imported_document", "char:0-25",
        status="approved",
    )
    assert (linked_id, added, linked) == (memory_id, False, True)

    profile = db.generate_memory_profile()
    assert profile["memory_ids"] == [memory_id]
    assert db.delete_document(first_source)
    assert db.get_document(first_source) is None
    assert {source[0] for source in db.get_memory_sources(memory_id)} == {second_source}
    active = next(row for row in db.list_memories() if row[0] == memory_id)
    assert active[6] == "approved"
    assert db.search_memories("quiet walk difficult workdays")[0][0] == memory_id

    assert db.delete_document(second_source)
    assert db.get_document(second_source) is None
    assert db.get_memory_sources(memory_id) == []
    assert next(row for row in db.list_memories() if row[0] == memory_id)[6] == "deleted"
    assert db.search_memories("quiet walk difficult workdays") == []
    assert db.get_current_memory_profile() is None
    stored_profiles = db.conn.execute("SELECT memory_ids FROM memory_profiles").fetchall()
    assert all(memory_id not in json.loads(row[0]) for row in stored_profiles)
    db.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Encrypted storage requires Windows DPAPI")
def test_memory_contradiction_review_and_user_confirmed_evolution(tmp_path):
    db = Database(tmp_path)
    old_source = db.import_document("old.md", "A previously stored preference", "old-hash")
    new_source = db.import_document("new.md", "A newer preference", "new-hash")
    old_id = db.add_memory(
        "I prefer working alone in the evenings.", "preference", old_source,
        status="approved", source_type="imported_document", source_span="char:0-31",
    )
    new_id = db.add_memory(
        "I prefer working with a team in the evenings.", "preference", new_source,
        source_type="imported_document", source_span="char:0-18",
    )

    relation_id = db.propose_memory_relation(new_id, old_id)
    assert relation_id
    assert db.propose_memory_relation(new_id, old_id) == relation_id
    assert db.list_memory_relations()[0][5:7] == ("possible_contradiction", "pending")
    assert {row[6] for row in db.list_memories() if row[0] in {old_id, new_id}} == {"approved", "pending"}
    assert db.review_memory_relation(relation_id, accept=True)
    assert db.list_memory_relations("accepted")[0][0] == relation_id
    # Accepting a relationship is only an annotation; it never hides either claim.
    assert {row[6] for row in db.list_memories() if row[0] in {old_id, new_id}} == {"approved", "pending"}
    profile = db.generate_memory_profile()
    assert profile["memory_ids"] == [old_id]
    assert db.get_current_memory_profile()["version"] == profile["version"]

    assert db.supersede_memory(new_id, old_id)
    assert db.get_current_memory_profile() is None
    profile = db.generate_memory_profile()
    statuses = {row[0]: row[6] for row in db.list_memories()}
    assert statuses[new_id] == "approved"
    assert statuses[old_id] == "superseded"
    assert profile["version"] == 2
    assert new_id in profile["memory_ids"] and old_id not in profile["memory_ids"]
    assert "I prefer working with a team" in profile["content"]
    assert "I prefer working alone" not in profile["content"]
    assert db.search_memories("working team evenings")[0][0] == new_id
    assert db.get_memory_sources(old_id) and db.get_memory_sources(new_id)
    operations = db.conn.execute(
        "SELECT operation FROM memory_events WHERE memory_id IN (?, ?)", (new_id, old_id)
    ).fetchall()
    assert {"RELATION_PROPOSED", "RELATION_ACCEPTED", "EVOLUTION_APPROVED", "SUPERSEDED"} <= {
        operation[0] for operation in operations
    }
    assert db.set_memory_status(new_id, "deleted")
    assert db.get_current_memory_profile() is None
    stored_profiles = db.conn.execute("SELECT memory_ids FROM memory_profiles").fetchall()
    assert all(new_id not in json.loads(row[0]) for row in stored_profiles)
    db.close()
