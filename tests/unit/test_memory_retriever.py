from datetime import datetime, timedelta, timezone

from psyche.memory.retriever import DEFAULT_WEIGHTS, rank_memories


def memory(memory_id, *, category="goal", statement="work goals", epistemic="user_reported",
           importance=3, confidence=1.0, created_at=None, sources=1, bm25=-0.1):
    return (
        memory_id, category, statement, epistemic, importance, confidence,
        "approved", "source", (created_at or datetime.now(timezone.utc)).isoformat(),
        sources, bm25,
    )


def test_weighted_rank_prefers_relevant_supported_memories_and_preserves_critical_override():
    now = datetime.now(timezone.utc)
    rows = [
        memory("unrelated-critical", category="relationship_pattern", statement="team conflict",
               importance=5, created_at=now, bm25=-0.3),
        memory("relevant", statement="work goals and next steps", importance=3,
               epistemic="user_confirmed", created_at=now - timedelta(days=30), bm25=-0.1),
        memory("critical-relevant", statement="work goals and deadlines", importance=5,
               created_at=now, bm25=-0.2),
    ]

    ranked = rank_memories(rows, "work goals", methodology="cbt", now=now)

    assert [row[0] for row in ranked] == ["critical-relevant", "relevant", "unrelated-critical"]
    assert len(ranked[0]) == 8


def test_ranking_weights_are_configurable():
    rows = [
        memory("relevant-low", statement="goal work", importance=1),
        memory("relevant-high", statement="goal work", importance=5),
    ]
    weights = {**DEFAULT_WEIGHTS, "importance": 1.0, "relevance": 0.0,
               "evidence": 0.0, "recency": 0.0, "context": 0.0}

    ranked = rank_memories(rows, "goal work", weights=weights)

    assert [row[0] for row in ranked] == ["relevant-high", "relevant-low"]
