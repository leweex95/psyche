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

For local voice, install optional dependencies with `python -m pip install -e ".[voice]"`, then run `psyche voice setup` to download the configured faster-whisper and Piper weights. `psyche audio check` provides a local-only input diagnostic: say a normal sentence during capture and compare per-channel levels with Psyche's downmixed mono level. It reports aggregate measurements, not audio. Voice unit tests use generated PCM frames and fake model adapters; they do not require model downloads or audio hardware and do not establish intelligibility, device, or live-voice acceptance. Audio is not uploaded; transcripts and bounded conversation context are sent to the configured LLM for replies.

Unit tests use fakes and need no API key. Windows integration tests use SQLCipher and DPAPI. A memory lifecycle test covers pending review, approval, FTS retrieval, correction, deletion, and reopen. A CLI integration test covers start, exit, resume, transcript persistence, and end with a fake provider; cloud-disabled mode is tested to block provider initialization. The text E2E workflow uses a synthetic 3,000-word source and mocked providers to exercise import, evidence-quoted extraction, pending review, correction, profile generation, retrieval in an Auto-to-CBT session, process-exit persistence, resumed-context retrieval (including the correction and prior turns), transcript persistence, and source/memory/profile deletion. Context assembly tests cover configurable profile/memory/history budgets, source attribution, and current-turn placement. Safety unit tests cover a small set of explicit and negated imminent-harm phrases, diagnosis-request routing, ordinary distress, provider bypass for streamed acute responses, and prompt-level boundaries for paranoia, possible mania, trauma disclosures, dependence, and corrections; they are not comprehensive or clinically validated. Voice tests verify local PCM conversion, local model adapter behavior, microphone negotiation/downmixing, aggregate level diagnostics, and playback resampling. They do not establish live-provider, clinical-quality, personal-essay recall, or audio acceptance.

Manual acceptance still required: set up local speech models; import and review a 3,000-word essay; confirm only text is sent to DeepSeek; conduct a 30-minute live voice session and barge-in; switch method in-session; validate memory recall/correction/deletion and prompt-injection resistance; review safety scenarios with qualified independent clinical reviewers before public deployment. Performance and conversational-quality thresholds in `DEVELOPMENT_PLAN.md` remain unmeasured.
