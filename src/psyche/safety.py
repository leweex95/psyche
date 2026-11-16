"""Small deterministic safety baseline; this is not a complete risk detector."""

import re


_ACUTE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\bkill myself\s+(?:tonight|today|now|right now)\b",
    r"\b(?:going to|about to|planning to|plan to|will)\s+kill myself\b",
    r"\b(?:about to|going to|planning to|plan to|will)\s+hurt\s+(?:myself|someone|somebody)\b",
    r"\bhurt\s+(?:myself|someone|somebody)\s+(?:tonight|today|now|right now)\b",
    r"\b(?:end my life|take my own life)\s+(?:tonight|today|now|right now)\b",
    r"\b(?:going to|about to|planning to|plan to|will)\s+(?:end my life|take my own life)\b",
))

_NEGATED_INTENT = re.compile(
    r"\b(?:i\s+)?(?:am not|i'm not|not|never|don't|do not|won't|will not)\s+"
    r"(?:(?:going|about|planning|plan|intend|want)\s+to\s+)?$"
)

_DIAGNOSIS_REQUESTS = tuple(re.compile(pattern) for pattern in (
    r"\b(?:can|could|would)\s+you\s+(?:please\s+)?diagnose\b",
    r"\bdiagnose\s+me\b",
    r"\bwhat\s+(?:mental\s+)?disorder\s+do\s+i\s+have\b",
    r"\bdo\s+i\s+have\s+(?:depression|anxiety|bipolar disorder|adhd|ptsd)\b",
))


def acute_concern(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    for pattern in _ACUTE_PATTERNS:
        for match in pattern.finditer(normalized):
            prefix = normalized[max(0, match.start() - 48):match.start()]
            if not _NEGATED_INTENT.search(prefix):
                return True
    return False


def safety_response(text: str) -> str | None:
    if acute_concern(text):
        return ("I’m really glad you told me. If you may act soon, contact your local emergency number now or go to an emergency department. "
                "If you can, move near someone you trust and tell them plainly what’s happening. I can stay with you while you reach out.")
    normalized = " ".join(text.casefold().split())
    if any(pattern.search(normalized) for pattern in _DIAGNOSIS_REQUESTS):
        return "I can’t diagnose or determine a disorder. I can help you describe what you’re experiencing and think through questions to discuss with a qualified clinician."
    return None
