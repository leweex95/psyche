import pytest

from psyche.conversation.prompts import SYSTEM_PROMPT
from psyche.safety import acute_concern, safety_response


@pytest.mark.parametrize("text", [
    "I am going to kill myself tonight.",
    "I'm about to hurt myself.",
    "I will hurt someone right now.",
    "I plan to end my life tonight.",
    "I intend to take my own life now.",
])
def test_explicit_immediate_danger_uses_local_support_response(text):
    assert acute_concern(text)
    response = safety_response(text)
    assert response is not None
    assert "emergency" in response.lower()


@pytest.mark.parametrize("text", [
    "I am not going to kill myself tonight.",
    "I don't want to hurt myself right now.",
    "I will not end my life tonight.",
])
def test_negated_intent_does_not_trigger_the_acute_script(text):
    assert not acute_concern(text)
    assert safety_response(text) is None


@pytest.mark.parametrize("text", [
    "Can you diagnose me with depression?",
    "What disorder do I have?",
    "Do I have PTSD?",
])
def test_diagnosis_requests_are_redirected_to_qualified_care(text):
    response = safety_response(text)
    assert response is not None
    assert "diagnose or determine" in response.lower()
    assert "qualified clinician" in response.lower()


def test_reporting_an_existing_diagnosis_is_not_misread_as_a_diagnosis_request():
    assert safety_response("My doctor diagnosed me with depression last year.") is None


def test_routine_distress_is_not_mistaken_for_an_acute_crisis():
    assert safety_response("I felt overwhelmed after a difficult meeting.") is None


def test_system_policy_covers_plan_critical_safety_behaviors():
    prompt = SYSTEM_PROMPT.casefold()
    assert "without affirming unverified claims as facts" in prompt
    assert "do not label or diagnose it" in prompt
    assert "without demanding details" in prompt
    assert "only or best source of support" in prompt
    assert "accept the correction" in prompt
    assert "do not treat silence or disconnection as evidence" in prompt
    assert "avoid canned praise and stock reassurance" in prompt
    assert "do not repeat a greeting or introduction" in prompt
    assert "ask directly but gently whether they are having thoughts of suicide" in prompt
