"""Small deterministic safety baseline; this is not a complete risk detector."""

ACUTE_PHRASES = ("kill myself tonight", "going to kill myself", "about to hurt myself", "hurt someone right now")


def acute_concern(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(phrase in normalized for phrase in ACUTE_PHRASES)


def safety_response(text: str) -> str | None:
    if acute_concern(text):
        return ("I’m really glad you told me. If you may act soon, contact your local emergency number now or go to an emergency department. "
                "If you can, move near someone you trust and tell them plainly what’s happening. I can stay with you while you reach out.")
    if "diagnos" in text.casefold() or "what disorder" in text.casefold():
        return "I can’t diagnose or determine a disorder. I can help you describe what you’re experiencing and think through questions to discuss with a qualified clinician."
    return None
