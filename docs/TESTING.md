# Testing

Install and run deterministic tests on Windows:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pip check
psyche --help
psyche init
psyche doctor
```

Run just the deterministic text workflow rehearsal with:

```powershell
python -m pytest tests/integration/test_text_e2e_workflow.py -q
```

For voice adapter development, install optional dependencies with `python -m pip install -e ".[voice]"`. Voice unit tests use generated PCM-like frames and fake VAD; they do not require audio hardware and do not establish device or live-voice acceptance.

Unit tests use fakes and need no API key. Windows integration tests use SQLCipher and DPAPI. A memory lifecycle test covers pending review, approval, FTS retrieval, correction, deletion, and reopen. A CLI integration test covers start, exit, resume, transcript persistence, and end with a fake provider; cloud-disabled mode is tested to block provider initialization. The text E2E workflow uses a synthetic 3,000-word source and mocked providers to exercise import, evidence-quoted extraction, pending review, correction, profile generation, retrieval in an Auto-to-CBT session, process-exit persistence, resumed-context retrieval (including the correction and prior turns), transcript persistence, and source/memory/profile deletion. Context assembly tests cover configurable profile/memory/history budgets, source attribution, and current-turn placement. These deterministic checks do not establish live-provider, clinical-quality, personal-essay recall, or audio acceptance.

Manual acceptance still required: install on target Windows; import and review a 3,000-word essay; test actual cloud-bound payloads; conduct a 30-minute live voice session and barge-in; switch method in-session; validate memory recall/correction/deletion and prompt-injection resistance; review safety scenarios with qualified independent clinical reviewers before public deployment. Performance and conversational-quality thresholds in `DEVELOPMENT_PLAN.md` remain unmeasured.
