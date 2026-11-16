from pathlib import Path
from datetime import datetime, timezone
import json
import uuid

from .encryption import load_or_create_key, sqlcipher_module
from ..memory.retriever import rank_memories


class Database:
    PROFILE_TEXT_BUDGET = 6000  # Approx. 1,500 tokens for English prose; conservative character bound.

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "psyche.db"
        self._db = sqlcipher_module()
        key = load_or_create_key(data_dir / "database.dpapi")
        self.conn = self._db.connect(str(self.path))
        self.conn.execute(f"PRAGMA key = \"x'{key.hex()}'\"")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        self._migrate()

    def _migrate(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, methodology TEXT NOT NULL, modality TEXT NOT NULL,
              started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, summary TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
              role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
              content TEXT NOT NULL, created_at TEXT NOT NULL,
              delivery_status TEXT NOT NULL DEFAULT 'delivered'
            );
            CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, created_at);
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, filename TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE,
              imported_at TEXT NOT NULL, content TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY, category TEXT NOT NULL, statement TEXT NOT NULL,
              epistemic_status TEXT NOT NULL, importance INTEGER NOT NULL,
              confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
              source_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              source_type TEXT NOT NULL DEFAULT 'document', source_span TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_events (
              id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, operation TEXT NOT NULL,
              previous_state TEXT, proposed_state TEXT, review_status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_sources (
              memory_id TEXT NOT NULL REFERENCES memories(id), source_id TEXT NOT NULL,
              source_type TEXT NOT NULL, source_span TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
              PRIMARY KEY(memory_id, source_id, source_span)
            );
            CREATE TABLE IF NOT EXISTS memory_relations (
              id TEXT PRIMARY KEY,
              memory_id TEXT NOT NULL REFERENCES memories(id),
              related_memory_id TEXT NOT NULL REFERENCES memories(id),
              relation_type TEXT NOT NULL CHECK(relation_type IN ('possible_contradiction','related')),
              review_status TEXT NOT NULL DEFAULT 'pending'
                CHECK(review_status IN ('pending','accepted','dismissed')),
              created_at TEXT NOT NULL, reviewed_at TEXT,
              UNIQUE(memory_id, related_memory_id, relation_type),
              CHECK(memory_id <> related_memory_id)
            );
            CREATE TABLE IF NOT EXISTS memory_profiles (
              version INTEGER PRIMARY KEY, content TEXT NOT NULL,
              memory_ids TEXT NOT NULL, created_at TEXT NOT NULL,
              is_current INTEGER NOT NULL DEFAULT 1
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(memory_id UNINDEXED, statement, category);
        """)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "source_type" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN source_type TEXT NOT NULL DEFAULT 'document'")
        if "source_span" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN source_span TEXT")
        self.conn.execute("""INSERT OR IGNORE INTO memory_sources
                          (memory_id, source_id, source_type, source_span, created_at)
                          SELECT id, source_id, source_type, COALESCE(source_span, ''), created_at FROM memories""")
        self.conn.commit()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def _invalidate_profiles(self) -> None:
        self.conn.execute("UPDATE memory_profiles SET is_current=0 WHERE is_current=1")

    def _purge_profiles_for_memory(self, memory_id: str) -> None:
        """Remove derived snapshots containing a memory that was explicitly deleted."""
        profiles = self.conn.execute("SELECT version, memory_ids FROM memory_profiles").fetchall()
        for version, encoded_ids in profiles:
            try:
                contains_memory = memory_id in json.loads(encoded_ids)
            except (TypeError, json.JSONDecodeError):
                contains_memory = True
            if contains_memory:
                self.conn.execute("DELETE FROM memory_profiles WHERE version=?", (version,))

    def create_session(self, methodology="auto", modality="text") -> str:
        sid = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, NULL, 'active', NULL)",
                          (sid, methodology, modality, self.now()))
        self.conn.commit()
        return sid

    def add_message(self, session_id: str, role: str, content: str, delivery_status="delivered") -> str:
        mid = str(uuid.uuid4())
        self.conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                          (mid, session_id, role, content, self.now(), delivery_status))
        self.conn.commit()
        return mid

    def get_messages(self, session_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at, rowid", (session_id,)).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def get_message_records(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, role, content, created_at, delivery_status FROM messages "
            "WHERE session_id=? ORDER BY created_at, rowid", (session_id,),
        ).fetchall()
        return [{"id": row[0], "role": row[1], "content": row[2],
                 "created_at": row[3], "delivery_status": row[4]} for row in rows]

    def list_sessions(self) -> list[tuple]:
        return self.conn.execute("SELECT id, methodology, modality, started_at, status FROM sessions ORDER BY started_at DESC").fetchall()

    def session_exists(self, session_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None

    def end_session(self, session_id: str):
        self.conn.execute("UPDATE sessions SET status='completed', ended_at=? WHERE id=?", (self.now(), session_id))
        self.conn.commit()

    def import_document(self, filename: str, content: str, content_hash: str) -> str | None:
        if self.conn.execute("SELECT 1 FROM documents WHERE content_hash=?", (content_hash,)).fetchone():
            return None
        document_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                          (document_id, filename, content_hash, self.now(), content))
        self.conn.commit()
        return document_id

    def has_document_hash(self, content_hash: str) -> bool:
        return self.conn.execute("SELECT 1 FROM documents WHERE content_hash=?", (content_hash,)).fetchone() is not None

    def documents(self):
        return self.conn.execute("SELECT id, filename, content FROM documents ORDER BY imported_at").fetchall()

    def add_memory(self, statement: str, category: str, source_id: str,
                   epistemic_status="user_reported", importance=3, confidence=1.0,
                   status="pending", source_type="document", source_span: str | None = None) -> str:
        memory_id, _, _ = self.add_memory_with_provenance(
            statement, category, source_id, source_type, source_span,
            epistemic_status, importance, confidence, status,
        )
        return memory_id

    def add_memory_with_provenance(self, statement: str, category: str, source_id: str,
                                   source_type: str, source_span: str | None,
                                   epistemic_status="user_reported", importance=3,
                                   confidence=1.0, status="pending") -> tuple[str, bool, bool]:
        def normalized(value: str) -> str:
            return " ".join("".join(ch if ch.isalnum() else " " for ch in value.casefold()).split())

        candidate_key = normalized(statement)
        existing = next((row for row in self.conn.execute(
            "SELECT id, statement, status FROM memories WHERE category=? AND status IN ('pending','approved')",
            (category,),
        ) if normalized(row[1]) == candidate_key), None)
        if existing:
            memory_id = existing[0]
            span = source_span or ""
            linked_already = self.conn.execute(
                "SELECT 1 FROM memory_sources WHERE memory_id=? AND source_id=? AND source_span=?",
                (memory_id, source_id, span),
            ).fetchone()
            linked = linked_already is None
            if linked:
                now = self.now()
                self.conn.execute("""INSERT INTO memory_sources
                                  (memory_id, source_id, source_type, source_span, created_at)
                                  VALUES (?, ?, ?, ?, ?)""",
                                  (memory_id, source_id, source_type, span, now))
                self.conn.execute("INSERT INTO memory_events VALUES (?, ?, 'LINK', ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), memory_id, existing[2],
                                   f"{source_type}:{source_id}:{span}", existing[2], now))
            self.conn.commit()
            return memory_id, False, linked
        memory_id, now = str(uuid.uuid4()), self.now()
        self.conn.execute("""INSERT INTO memories (
                          id, category, statement, epistemic_status, importance,
                          confidence, status, source_id, created_at, updated_at,
                          source_type, source_span
                          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (memory_id, category, statement, epistemic_status, importance,
                           confidence, status, source_id, now, now, source_type, source_span))
        self.conn.execute("INSERT INTO memories_fts VALUES (?, ?, ?)",
                          (memory_id, statement, category))
        self.conn.execute("""INSERT INTO memory_sources
                          (memory_id, source_id, source_type, source_span, created_at)
                          VALUES (?, ?, ?, ?, ?)""",
                          (memory_id, source_id, source_type, source_span or "", now))
        self.conn.execute("INSERT INTO memory_events VALUES (?, ?, 'ADD', NULL, ?, ?, ?)",
                          (str(uuid.uuid4()), memory_id, statement,
                           "pending" if status == "pending" else "approved", now))
        self._invalidate_profiles()
        self.conn.commit()
        return memory_id, True, True

    def get_memory_sources(self, memory_id: str) -> list[tuple]:
        return self.conn.execute("""SELECT source_id, source_type, source_span, created_at
                                 FROM memory_sources WHERE memory_id=? ORDER BY created_at""",
                                  (memory_id,)).fetchall()

    def propose_memory_relation(self, memory_id: str, related_memory_id: str,
                                relation_type: str = "possible_contradiction") -> str | None:
        """Record a user-suggested relationship; it never changes retrieval status."""
        if relation_type not in {"possible_contradiction", "related"} or memory_id == related_memory_id:
            return None
        rows = self.conn.execute(
            "SELECT id, status FROM memories WHERE id IN (?, ?)",
            (memory_id, related_memory_id),
        ).fetchall()
        if len(rows) != 2 or any(row[1] in {"deleted", "rejected", "superseded"} for row in rows):
            return None
        existing = self.conn.execute(
            "SELECT id, review_status FROM memory_relations WHERE memory_id=? AND related_memory_id=? AND relation_type=?",
            (memory_id, related_memory_id, relation_type),
        ).fetchone()
        if existing:
            if existing[1] != "pending":
                now = self.now()
                self.conn.execute(
                    "UPDATE memory_relations SET review_status='pending', reviewed_at=NULL WHERE id=?",
                    (existing[0],),
                )
                self.conn.execute(
                    "INSERT INTO memory_events VALUES (?, ?, 'RELATION_PROPOSED', ?, ?, 'pending', ?)",
                    (str(uuid.uuid4()), memory_id, related_memory_id,
                     f"{relation_type}:{existing[0]}", now),
                )
                self._invalidate_profiles()
                self.conn.commit()
            return existing[0]
        relation_id, now = str(uuid.uuid4()), self.now()
        self.conn.execute(
            "INSERT INTO memory_relations VALUES (?, ?, ?, ?, 'pending', ?, NULL)",
            (relation_id, memory_id, related_memory_id, relation_type, now),
        )
        for item_id, other_id in ((memory_id, related_memory_id), (related_memory_id, memory_id)):
            self.conn.execute(
                "INSERT INTO memory_events VALUES (?, ?, 'RELATION_PROPOSED', ?, ?, 'pending', ?)",
                (str(uuid.uuid4()), item_id, other_id, f"{relation_type}:{relation_id}", now),
            )
        self.conn.commit()
        return relation_id

    def list_memory_relations(self, review_status: str | None = "pending") -> list[tuple]:
        sql = """SELECT r.id, r.memory_id, a.statement, r.related_memory_id, b.statement,
                         r.relation_type, r.review_status, r.created_at
                  FROM memory_relations r JOIN memories a ON a.id=r.memory_id
                  JOIN memories b ON b.id=r.related_memory_id"""
        args: tuple = ()
        if review_status is not None:
            sql += " WHERE r.review_status=?"
            args = (review_status,)
        return self.conn.execute(sql + " ORDER BY r.created_at", args).fetchall()

    def review_memory_relation(self, relation_id: str, accept: bool) -> bool:
        row = self.conn.execute(
            "SELECT memory_id, related_memory_id, relation_type, review_status FROM memory_relations WHERE id=?",
            (relation_id,),
        ).fetchone()
        if not row or row[3] != "pending":
            return False
        status, now = ("accepted" if accept else "dismissed"), self.now()
        self.conn.execute(
            "UPDATE memory_relations SET review_status=?, reviewed_at=? WHERE id=? AND review_status='pending'",
            (status, now, relation_id),
        )
        operation = "RELATION_ACCEPTED" if accept else "RELATION_DISMISSED"
        for memory_id, other_id in ((row[0], row[1]), (row[1], row[0])):
            self.conn.execute(
                "INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, 'approved', ?)",
                (str(uuid.uuid4()), memory_id, operation, other_id,
                 f"{row[2]}:{relation_id}", now),
            )
        self._invalidate_profiles()
        self.conn.commit()
        return True

    def supersede_memory(self, new_memory_id: str, old_memory_id: str) -> bool:
        """User-confirmed evolution: approve the sourced replacement and retire the old fact."""
        if new_memory_id == old_memory_id:
            return False
        rows = self.conn.execute(
            "SELECT id, category, statement, status FROM memories WHERE id IN (?, ?)",
            (new_memory_id, old_memory_id),
        ).fetchall()
        by_id = {row[0]: row for row in rows}
        new, old = by_id.get(new_memory_id), by_id.get(old_memory_id)
        if (new is None or old is None or new[1] != old[1]
                or new[3] not in {"pending", "approved"} or old[3] != "approved"
                or not self.get_memory_sources(new_memory_id)
                or not self.get_memory_sources(old_memory_id)):
            return False
        now = self.now()
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "UPDATE memories SET status='approved', updated_at=? WHERE id=?",
                (now, new_memory_id),
            )
            self.conn.execute(
                "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                (now, old_memory_id),
            )
            self.conn.execute("UPDATE memories_fts SET statement=? WHERE memory_id=?", (new[2], new_memory_id))
            self.conn.execute("UPDATE memories_fts SET statement=? WHERE memory_id=?", (old[2], old_memory_id))
            self.conn.execute(
                "UPDATE memory_events SET review_status='superseded' WHERE memory_id=? AND review_status='pending'",
                (new_memory_id,),
            )
            self.conn.execute("INSERT INTO memory_events VALUES (?, ?, 'EVOLUTION_APPROVED', ?, ?, 'approved', ?)",
                              (str(uuid.uuid4()), new_memory_id, old[2], new[2], now))
            self.conn.execute("INSERT INTO memory_events VALUES (?, ?, 'SUPERSEDED', ?, ?, 'approved', ?)",
                              (str(uuid.uuid4()), old_memory_id, old[2], new[2], now))
            self._invalidate_profiles()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return True

    def get_document(self, document_id: str):
        return self.conn.execute("SELECT id, filename, content FROM documents WHERE id=?",
                                 (document_id,)).fetchone()

    def delete_document(self, document_id: str) -> bool:
        return self.delete_source(document_id, {"document", "imported_document"})

    def delete_source(self, source_id: str, source_types: set[str] | None = None) -> bool:
        """Delete an imported document or transcript message and its dependent evidence links."""
        source_types = source_types or {"document", "imported_document", "session_message"}
        if not source_types:
            return False
        links = self.conn.execute(
            "SELECT memory_id FROM memory_sources WHERE source_id=? AND source_type IN ("
            + ",".join("?" for _ in source_types) + ")",
            (source_id, *sorted(source_types)),
        ).fetchall() if source_types else []
        document = self.conn.execute("SELECT 1 FROM documents WHERE id=?", (source_id,)).fetchone()
        message = self.conn.execute("SELECT 1 FROM messages WHERE id=?", (source_id,)).fetchone()
        if document and source_types.intersection({"document", "imported_document"}):
            self.conn.execute("DELETE FROM documents WHERE id=?", (source_id,))
        elif message and "session_message" in source_types:
            self.conn.execute("DELETE FROM messages WHERE id=?", (source_id,))
        elif not (document or message):
            return False
        self.conn.execute(
            "DELETE FROM memory_sources WHERE source_id=? AND source_type IN ("
            + ",".join("?" for _ in source_types) + ")",
            (source_id, *sorted(source_types)),
        )
        affected = {row[0] for row in links}
        self.conn.commit()
        for memory_id in affected:
            remaining = self.conn.execute(
                "SELECT source_id, source_type, source_span FROM memory_sources "
                "WHERE memory_id=? ORDER BY created_at LIMIT 1", (memory_id,)
            ).fetchone()
            current = self.conn.execute("SELECT status FROM memories WHERE id=?", (memory_id,)).fetchone()
            if remaining:
                self.conn.execute(
                    "UPDATE memories SET source_id=?, source_type=?, source_span=?, updated_at=? WHERE id=?",
                    (*remaining, self.now(), memory_id),
                )
                self.conn.commit()
            else:
                self.conn.execute(
                    "UPDATE memories SET source_id='', source_type='deleted', source_span=NULL, updated_at=? WHERE id=?",
                    (self.now(), memory_id),
                )
                self.conn.commit()
            if not remaining and current and current[0] != "deleted":
                self.set_memory_status(memory_id, "deleted")
        return True

    def get_user_messages(self, session_id: str) -> list[tuple]:
        return self.conn.execute("""SELECT id, content FROM messages
                                 WHERE session_id=? AND role='user' ORDER BY created_at, rowid""",
                                 (session_id,)).fetchall()

    def list_memories(self, status=None):
        sql = "SELECT id, category, statement, epistemic_status, importance, confidence, status, source_id, source_type, source_span FROM memories"
        args = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        return self.conn.execute(sql + " ORDER BY created_at", args).fetchall()

    def search_memories(self, query: str, limit=12, methodology="auto", weights=None):
        terms = [term for term in query.split() if term.isalnum()]
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '') + '"' for term in terms[:12])
        try:
            rows = self.conn.execute("""SELECT m.id,m.category,m.statement,m.epistemic_status,
                     m.importance,m.confidence,m.status,m.source_id,m.created_at,
                     (SELECT COUNT(*) FROM memory_sources ms WHERE ms.memory_id=m.id),
                     bm25(memories_fts)
              FROM memories_fts f JOIN memories m ON m.id=f.memory_id
              WHERE memories_fts MATCH ? AND m.status='approved'""",
              (expression,)).fetchall()
            return rank_memories(rows, query, methodology, weights)[:limit]
        except Exception:
            return []

    def set_memory_status(self, memory_id: str, status: str):
        row = self.conn.execute("SELECT statement FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return False
        self.conn.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?", (status, self.now(), memory_id))
        self.conn.execute("UPDATE memories_fts SET statement=? WHERE memory_id=?", (row[0] if status != 'deleted' else '', memory_id))
        self.conn.execute("UPDATE memory_events SET review_status='superseded' WHERE memory_id=? AND review_status='pending'", (memory_id,))
        self.conn.execute("INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, 'approved', ?)",
                          (str(uuid.uuid4()), memory_id, status.upper(), row[0], row[0], self.now()))
        if status == "deleted":
            self._purge_profiles_for_memory(memory_id)
        self._invalidate_profiles()
        self.conn.commit()
        return True

    def correct_memory(self, memory_id: str, statement: str):
        row = self.conn.execute("SELECT statement FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return False
        now = self.now()
        self.conn.execute("UPDATE memories SET statement=?, epistemic_status='user_confirmed', status='approved', updated_at=? WHERE id=?",
                          (statement, now, memory_id))
        self.conn.execute("UPDATE memories_fts SET statement=? WHERE memory_id=?", (statement, memory_id))
        self.conn.execute("INSERT INTO memory_events VALUES (?, ?, 'CORRECT', ?, ?, 'approved', ?)",
                          (str(uuid.uuid4()), memory_id, row[0], statement, now))
        self._invalidate_profiles()
        self.conn.commit()
        return True

    def generate_memory_profile(self) -> dict:
        """Persist a transparent, deterministic profile derived only from approved memories."""
        memories = self.conn.execute(
            "SELECT id, category, statement, epistemic_status FROM memories "
            "WHERE status='approved' ORDER BY created_at, id"
        ).fetchall()
        by_id = {row[0]: row for row in memories}
        sections: list[tuple[str, list[tuple]]] = [
            ("CURRENT GOALS", [row for row in memories if row[1] == "goal"]),
            ("RECURRING SELF-REPORTED PATTERNS", [row for row in memories if row[1] in {
                "emotional_pattern", "coping_behavior", "relationship_pattern",
            }]),
            ("IMPORTANT PERSONAL HISTORY", [row for row in memories if row[1] in {
                "personal_history", "important_event",
            }]),
            ("CURRENT CONCERNS", []),
            ("PREFERRED CONVERSATIONAL APPROACH", [row for row in memories if row[1] == "preference"]),
            ("ACTIVE THERAPEUTIC EXERCISES", []),
        ]
        assigned = {row[0] for _, items in sections for row in items}
        other = [row for row in memories if row[0] not in assigned and row[3] != "disputed"]
        if other:
            sections.append(("OTHER APPROVED MEMORIES (categorized, not interpreted)", other))

        disputed = [row for row in memories if row[3] == "disputed"]
        flagged_ids: set[str] = set()
        for first, second in self.conn.execute(
            "SELECT memory_id, related_memory_id FROM memory_relations "
            "WHERE relation_type='possible_contradiction' AND review_status='accepted'"
        ):
            if first in by_id and second in by_id:
                flagged_ids.update((first, second))
        disputed_ids = {row[0] for row in disputed}
        disputed.extend(by_id[item_id] for item_id in sorted(flagged_ids - disputed_ids))
        sections.append(("UNRESOLVED OR DISPUTED INTERPRETATIONS", disputed))

        lines = ["# Derived memory profile", "",
                 "Built only from currently approved atomic memories. IDs refer to source memory records.", ""]
        included_ids: set[str] = set()
        omitted = 0
        used_chars = 0
        for title, items in sections:
            lines.extend((f"## {title}", ""))
            included = []
            for row in items:
                entry = f"- [{row[0]}] {row[2]}"
                if used_chars + len(entry) <= self.PROFILE_TEXT_BUDGET:
                    included.append(entry)
                    included_ids.add(row[0])
                    used_chars += len(entry)
                else:
                    omitted += 1
            lines.extend(included or (["- None recorded."] if not items else []))
            lines.append("")
        if omitted:
            lines.extend((f"_{omitted} approved memory item(s) omitted by the {self.PROFILE_TEXT_BUDGET}-character profile limit._", ""))
        used_ids = sorted(included_ids)
        version = self.conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM memory_profiles").fetchone()[0]
        created_at = self.now()
        content = "\n".join(lines).rstrip()
        self.conn.execute("UPDATE memory_profiles SET is_current=0 WHERE is_current=1")
        self.conn.execute("INSERT INTO memory_profiles VALUES (?, ?, ?, ?, 1)",
                          (version, content, json.dumps(used_ids), created_at))
        self.conn.commit()
        return {"version": version, "content": content, "memory_ids": used_ids, "created_at": created_at}

    def get_current_memory_profile(self) -> dict | None:
        row = self.conn.execute(
            "SELECT version, content, memory_ids, created_at FROM memory_profiles "
            "WHERE is_current=1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {"version": row[0], "content": row[1],
                "memory_ids": json.loads(row[2]), "created_at": row[3]}

    def close(self):
        self.conn.close()
