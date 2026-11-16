# Psyche

Psyche is a local-first CLI psychological-support companion. It is not a clinician, diagnostic tool, or substitute for professional care. It supports encrypted text sessions, UTF-8 text/Markdown imports, user-confirmed audio transcription, reviewable memory candidates, methodology prompts, experimental voice sessions, and memory management. Live voice and full end-to-end acceptance are not yet verified.

## Install (Windows 10/11, Python 3.12+)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
# Optional microphone, VAD, and streaming-transcription dependencies:
pip install -e ".[voice]"
Copy-Item .env.example .env
```

Set `PSYCHE_LLM_API_KEY` in `.env` for conversational and explicitly requested memory extraction. `PSYCHE_LLM_MODEL` optionally selects the model. Data is stored in `%USERPROFILE%\.psyche` by default; set `PSYCHE_DATA_DIR` to change it. The database encryption key is protected using Windows DPAPI for the current Windows account. Losing the Windows account/profile or moving the database to another computer can make it unrecoverable. Do not copy the key file or database as a plaintext backup; SQLCipher database files remain encrypted. The `.[voice]` extra installs microphone capture, local VAD, and WebSocket support for the experimental voice session.

## Use

```powershell
psyche init
psyche doctor
psyche session start --mode text --method auto
psyche sessions list
psyche session resume SESSION_ID
psyche sessions show SESSION_ID
psyche config show
psyche import .\notes\personal_essay.md
psyche import-audio .\notes\voice-note.m4a
psyche memory extract --document-id DOCUMENT_ID
psyche memory extract --session-id SESSION_ID
psyche memory review
psyche memory show
psyche memory search "work"
psyche export .\psyche-export.json
```

In a session, use `/help`, `/method` to switch, `/memory`, and `/end`. A session interrupted by closing the terminal remains stored and can be resumed. Methodology prompts are guardrails, not evidence of clinical efficacy. Import creates conservative verbatim sentence candidates that remain pending until you approve them. `psyche memory extract` explicitly sends the selected document or session's user messages to the configured LLM provider and creates additional source-quoted candidates for review; extracted candidates are never auto-approved. `psyche memory forget ID --delete-source` also deletes linked source messages/documents. The provider may retain or process submitted data under its own policies. Telemetry is not added.

## Privacy and safety limitations

Live voice audio is streamed to the configured transcription provider and not persisted locally. `psyche import-audio` uploads a selected file only after interactive confirmation, stores its transcript locally, and does not retain the raw file in the app database. Experimental voice capture/VAD continue during response generation and playback, allowing speech onset to cancel active work. Incremental STT emits partial captions during capture; LLM text streams into sentence-sized TTS requests, not token-to-audio synthesis. On interruption, only complete sentences confirmed played are saved, explicitly marked interrupted; echo cancellation is absent. Conversation requests include bounded recent session context, a current profile when available, relevant approved memories, and system/method prompts; generated speech text is sent to TTS. Set `privacy.cloud_processing_allowed: false` to block cloud-dependent conversations, audio import, and memory extraction; no local provider is implemented. Provider retention is outside app control. API credentials are read from environment variables and excluded from displayed configuration. Safety phrase matching is intentionally narrow and is not a reliable crisis detector; in immediate danger, contact local emergency services or a trusted person. Do not rely on this tool for diagnosis or emergency treatment. `psyche export` writes unencrypted JSON. `psyche memory forget ID --delete-source` removes linked imported documents or session messages; deleting a memory without this option leaves source history intact. `psyche delete-local-data --confirm` removes local data files, but cannot promise physical erasure or deletion of provider-side copies.

## Tests

```powershell
pip install -e ".[dev]"
pytest
```

The automated suite requires neither an API credential nor a live provider. SQLCipher initialization and DPAPI encryption require Windows and should additionally be verified on the target machine.
