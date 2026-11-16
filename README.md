# Psyche

Psyche is a local-first CLI psychological-support companion. It is not a clinician, diagnostic tool, or substitute for professional care. It supports encrypted text sessions, UTF-8 text/Markdown imports, user-confirmed audio transcription, reviewable memory candidates, methodology prompts, experimental voice sessions, and memory management. Live voice and full end-to-end acceptance are not yet verified.

## Install (Windows 10/11, Python 3.13; tested with 3.13.9)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
# Optional local microphone, VAD, transcription, and speech dependencies:
pip install -e ".[voice]"
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` for the text conversation model and explicitly requested memory extraction. `PSYCHE_LLM_MODEL` optionally selects the DeepSeek model. Psyche sends transcripts and bounded text context to DeepSeek for responses; it does **not** send microphone audio. Transcription (faster-whisper) and speech synthesis (Piper) run locally. Run `psyche voice setup` to download the selected speech-model weights; those model files are downloaded to `%USERPROFILE%\.psyche\models`, but captured audio is not uploaded. Data is stored in `%USERPROFILE%\.psyche` by default; set `PSYCHE_DATA_DIR` to change it. The database encryption key is protected using Windows DPAPI for the current Windows account. Losing the Windows account/profile or moving the database to another computer can make it unrecoverable. Do not copy the key file or database as a plaintext backup; SQLCipher database files remain encrypted. The `.[voice]` extra installs microphone capture, local VAD, faster-whisper, and Piper for the experimental voice session.

## Use

```powershell
psyche init
psyche doctor
psyche voice setup
psyche session start --mode text --method auto
psyche session start --mode voice
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

Voice audio capture, VAD, utterance transcription (faster-whisper), and speech synthesis (Piper) run locally. `psyche import-audio` also transcribes locally after confirmation; raw audio is not copied into Psyche's database or sent to a provider. First-time model setup downloads model weights only. The recognized transcript, bounded recent context, current profile, relevant approved memories, and prompts are sent to DeepSeek to generate a reply; provider retention is outside app control. Assistant text is synthesized locally. Local Whisper currently transcribes after the VAD utterance ends (no partial captions); speech capture/VAD remain active during response generation and playback, allowing new speech to cancel active work. On interruption, only complete sentences confirmed played are saved as interrupted/incomplete; echo cancellation is absent. Set `privacy.cloud_processing_allowed: false` to block conversations and memory extraction; local audio import remains available. API credentials are read from environment variables and excluded from displayed configuration. Safety phrase matching is intentionally narrow and is not a reliable crisis detector; in immediate danger, contact local emergency services or a trusted person. Do not rely on this tool for diagnosis or emergency treatment. `psyche export` writes unencrypted JSON. `psyche memory forget ID --delete-source` removes linked imported documents or session messages; deleting a memory without this option leaves source history intact. `psyche delete-local-data --confirm` removes local data files, but cannot promise physical erasure or deletion of provider-side copies.

## Tests

```powershell
pip install -e ".[dev]"
pytest
```

The automated suite requires neither an API credential nor a live provider. SQLCipher initialization and DPAPI encryption require Windows and should additionally be verified on the target machine.
