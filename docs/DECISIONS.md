# Architecture decisions

- SQLCipher protects the whole local database; Windows DPAPI protects its random database key. Fail closed when SQLCipher/DPAPI is unavailable. Backups are not portable across Windows accounts without a separate migration mechanism.
- The currently available Python 3.13.9 was used; no Python 3.12 installation was performed. `sqlcipher3-wheels` replaces unavailable `sqlcipher3-binary` for Windows CPython 3.13.
- Text/Markdown import creates conservative verbatim candidates; optional provider extraction is separately invoked, source-quote checked, and review-only. This preserves provenance and prevents unreviewed model output becoming active memory.
- Methodology policies are prompt-level guidance and do not diagnose or claim therapeutic efficacy.
- Voice audio I/O is local-only: faster-whisper transcribes VAD-completed utterances and Piper synthesizes replies. Model weights may be downloaded during explicit setup, but user audio is not uploaded. Local Whisper is utterance-batched (no partial captions); live device, transcription-quality, latency, and long-run acceptance remain unperformed.
- DeepSeek is the POC text LLM. Bounded transcripts/context (recent messages, current profile, and ranked relevant approved memories) are sent to DeepSeek for responses and explicit extraction; provider retention is outside local control. Disabling cloud processing blocks conversation and extraction, while local audio processing/import remains available.
