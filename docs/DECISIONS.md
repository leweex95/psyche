# Architecture decisions

- SQLCipher protects the whole local database; Windows DPAPI protects its random database key. Fail closed when SQLCipher/DPAPI is unavailable. Backups are not portable across Windows accounts without a separate migration mechanism.
- The currently available Python 3.13.9 was used; no Python 3.12 installation was performed. `sqlcipher3-wheels` replaces unavailable `sqlcipher3-binary` for Windows CPython 3.13.
- Text/Markdown import creates conservative verbatim candidates; optional provider extraction is separately invoked, source-quote checked, and review-only. This preserves provenance and prevents unreviewed model output becoming active memory.
- Methodology policies are prompt-level guidance and do not diagnose or claim therapeutic efficacy.
- Voice is not simulated. Incremental transcription and streamed LLM-to-sentence-TTS playback exist, but live provider, physical device, latency, and long-run acceptance remain unperformed.
- Cloud context is bounded using configurable, conservative character-based token estimates and contains recent messages, current profile, and ranked relevant approved memories; provider retention is outside local control. Disabling cloud processing blocks cloud-dependent conversation and extraction commands because no local model adapter exists.
