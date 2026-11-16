from __future__ import annotations

import re
from datetime import datetime, timezone


DEFAULT_WEIGHTS = {"relevance": 0.40, "importance": 0.25, "evidence": 0.15,
                   "recency": 0.10, "context": 0.10}
_EPISTEMIC_QUALITY = {
    "user_confirmed": 1.0,
    "user_reported": 0.9,
    "model_hypothesis": 0.25,
    "disputed": 0.2,
    "superseded": 0.0,
}
_METHOD_CATEGORIES = {
    "cbt": {"emotional_pattern", "coping_behavior", "goal", "therapeutic_insight"},
    "schema": {"personal_history", "belief", "emotional_pattern", "coping_behavior",
               "relationship_pattern", "therapeutic_insight"},
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", text.casefold()))


def rank_memories(rows: list[tuple], query: str, methodology: str = "auto",
                  weights: dict[str, float] | None = None,
                  now: datetime | None = None) -> list[tuple]:
    """Rank approved FTS candidates with an explicit, uncalibrated heuristic.

    Rows are (id, category, statement, epistemic_status, importance, confidence,
    status, source_id, created_at, source_count, bm25). The last three fields are
    internal ranking data and are removed from returned tuples.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_weight = sum(max(0.0, float(value)) for value in w.values())
    if total_weight <= 0:
        w = DEFAULT_WEIGHTS.copy()
    else:
        w = {key: max(0.0, float(value)) / total_weight for key, value in w.items()}
    terms = _tokens(query)
    current = now or datetime.now(timezone.utc)
    method_categories = _METHOD_CATEGORIES.get(methodology, set())
    ranked: list[tuple[float, float, float, tuple]] = []
    for row in rows:
        age_days = 365.0
        memory_terms = _tokens(row[2])
        relevance = len(terms & memory_terms) / max(1, len(terms))
        importance = max(0.0, min(1.0, (float(row[4]) - 1) / 4))
        source_count = max(1, int(row[9]))
        corroboration = min(1.0, 0.8 + 0.1 * (source_count - 1))
        evidence = max(0.0, min(1.0, float(row[5]))) * _EPISTEMIC_QUALITY.get(row[3], 0.5) * corroboration
        try:
            created = datetime.fromisoformat(row[8])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (current - created.astimezone(timezone.utc)).total_seconds() / 86400)
            recency = 1 / (1 + age_days / 365)
        except (TypeError, ValueError):
            recency = 0.5
        context = (1.0 if row[1] in method_categories else 0.4) if method_categories else 0.7
        score = (w["relevance"] * relevance + w["importance"] * importance
                 + w["evidence"] * evidence + w["recency"] * recency
                 + w["context"] * context)
        # Importance alone never elevates an unrelated record. This narrow
        # override is limited to high-importance, directly relevant evidence.
        if int(row[4]) == 5 and relevance >= 0.8:
            score = max(score, 0.85)
        ranked.append((score, -float(row[10]), -age_days, row[:8]))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked]
