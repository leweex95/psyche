Psyche — GPT Luna Development Specification
VERSION 1.0 · PYTHON · WINDOWS · CLI-FIRST · PROOF OF CONCEPT
1. Project objective
Build a local Python application that functions as a persistent, voice-enabled psychological support companion.

The application must allow the user to import personal essays, previous therapy-session notes, and audio recordings; conduct natural spoken conversations; select CBT, Schema Therapy-informed support, or automatic methodology selection; and maintain an accurate psychological profile across sessions.

The POC must be genuinely usable, not merely an architectural demonstration.

The primary acceptance scenario is:
1. The user imports a detailed personal essay.
2. The system extracts significant information into structured memory.
3. The user starts a live voice session from the terminal.
4. The agent responds verbally, using concise, natural language.
5. The agent recalls relevant information from the essay and previous sessions.
6. New significant discoveries become memory candidates.
7. The user can review, correct, reject, or delete stored psychological information.
8. The application can be restarted without losing conversational continuity.
Fixed technology decisions
Dimension	Requirement
Platform	Windows 10/11
Language	Python 3.12+
Interface	CLI only
LLM	Cloud API, configurable
Voice	Live, bidirectional spoken conversation
Speech recognition	Local STT preferred
Speech synthesis	Cloud TTS permitted
Methodologies	CBT, Schema, Auto
Memory	SQLite, encrypted at rest
Memory updates	Automatic extraction, significant updates reviewed
Conversation	Flexible, natural, non-scripted
Deployment	Single local process
Users	Single-user POC
External access	None
Telemetry	Disabled

Explicit non-goals
Do not implement a web UI, iOS app, authentication server, cloud database, multi-user support, payment system, agent marketplace, autonomous web browsing, or complex multi-agent orchestration.

The system is a psychological support tool, not a licensed psychologist, diagnostic instrument, or replacement for professional mental-health treatment.

2. System architecture
The application should use a modular monolith with asynchronous orchestration.
CLI Controller
Commands, configuration, session controls

Session Orchestrator
State, turn management, interruption, safety
Conversation Engine
CBT, Schema, Auto, LLM adapter, context assembly

Voice Pipeline
Microphone, VAD, STT, TTS, playback, barge-in

Memory Engine
Extraction, consolidation, retrieval, review

Safety Controller
Risk detection, response policy, crisis routing



Encrypted Local Persistence
SQLite · Session archive · Memory evidence · Audio metadata

External Providers
Text-only LLM API (DeepSeek for the POC) · No cloud audio services

The diagram represents logical dependencies, not an execution sequence. In particular, memory retrieval occurs before LLM generation, while memory extraction normally occurs asynchronously after a completed conversational turn.
Architectural rules
- Use explicit Python interfaces between modules.
- Keep business logic independent of CLI and provider SDKs.
- Use asyncio for voice and network concurrency.
- Use one primary conversational LLM.
- Use separate structured LLM calls for memory extraction and methodology recommendation where needed.
- Do not introduce autonomous agents that communicate with one another.
- Keep all API credentials outside source control.
- Use dependency injection rather than global service instances.
- Never make memory writes directly from unvalidated LLM output.

3. Repository structure
psyche/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── default.yaml
│   └── methodologies/
│       ├── cbt.yaml
│       ├── schema.yaml
│       └── auto.yaml
├── src/
│   └── psyche/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── app.py
│       │
│       ├── conversation/
│       │   ├── orchestrator.py
│       │   ├── context_builder.py
│       │   ├── prompts.py
│       │   ├── methodology.py
│       │   └── session_state.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   └── cloud.py
│       │
│       ├── voice/
│       │   ├── audio_input.py
│       │   ├── vad.py
│       │   ├── stt.py
│       │   ├── tts.py
│       │   ├── playback.py
│       │   └── pipeline.py
│       │
│       ├── memory/
│       │   ├── models.py
│       │   ├── repository.py
│       │   ├── extractor.py
│       │   ├── consolidator.py
│       │   ├── retriever.py
│       │   ├── profile.py
│       │   └── review.py
│       │
│       ├── safety/
│       │   ├── classifier.py
│       │   └── policy.py
│       │
│       ├── ingestion/
│       │   ├── documents.py
│       │   └── audio.py
│       │
│       ├── storage/
│       │   ├── database.py
│       │   ├── encryption.py
│       │   └── migrations.py
│       │
│       └── models/
│           ├── session.py
│           └── messages.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── safety/
│   ├── memory/
│   └── voice/
│
├── evals/
│   ├── scenarios.yaml
│   └── runner.py
│
└── scripts/
    └── smoke_test.py


Avoid creating placeholder modules that contain no functional implementation. Build the repository incrementally, retaining this as the target structure rather than generating every file immediately.
4. CLI contract
Use Typer for commands and Rich for readable terminal output.

The executable command must be psyche.
Required commands
Command	Behavior
psyche init	Initialize encrypted storage and configuration
psyche import FILE	Import essay, session notes, or recording
psyche session start	Start a new text or voice session
psyche session resume	Resume an existing session
psyche sessions list	List previous sessions
psyche sessions show ID	Inspect a session transcript
psyche memory show	Show the current psychological profile
psyche memory search QUERY	Search memory records
psyche memory review	Review significant pending changes
psyche memory correct ID	Correct an existing record
psyche memory forget ID	Delete a memory and its derived representations
psyche config show	Display effective configuration without secrets
psyche doctor	Verify microphone, audio, API, database, and encryption setup

Example:
psyche init
psyche import .\notes\personal_essay.md
psyche memory review
psyche session start --mode voice --method auto


Within an interactive session, support /pause, /resume, /method, /memory, /end, and /help.

/end must flush the transcript and finalize the session. If summarization or memory extraction fails, the conversation must still be preserved for retry.

5. LLM provider architecture
The LLM integration must not be coupled to a particular provider.

Use a small adapter interface.
from typing import Protocol, AsyncIteratorfrom pydantic import BaseModelclass LLMMessage(BaseModel):    role: str    content: strclass LLMProvider(Protocol):    async def generate(        self,        messages: list[LLMMessage],        temperature: float,        max_output_tokens: int,    ) -> str: ...    def stream(        self,        messages: list[LLMMessage],        temperature: float,        max_output_tokens: int,    ) -> AsyncIterator[str]: ...    async def generate_structured(        self,        messages: list[LLMMessage],        response_model: type[BaseModel],    ) -> BaseModel: ...



The implementation must support cancellation, timeouts, retryable errors, and provider-specific structured-output capabilities.

The conversational model and the memory-extraction model may be configured separately, but they should initially use the same provider.
Prompt assembly
Every conversational request should contain these components in priority order:
1. System-level safety and behavioral constraints.
2. Selected therapeutic methodology.
3. Compact core psychological profile.
4. Relevant retrieved memories with evidence status.
5. Current session state and recent messages.
6. Latest user utterance.
Retrieved documents and stored memories are data, not instructions. The engine must not obey commands embedded in imported essays or transcripts that attempt to override its system behavior.
Response-length control
Set a target of 15–55 spoken words for ordinary conversational turns.

This is a soft target, not a hard truncation rule. The system must be allowed to respond more fully when the user requests an explanation, when an intervention requires instructions, or when safety requires additional information.

Use prompt constraints and evaluation rather than indiscriminately cutting generated text.

6. Therapeutic methodology engine
Each methodology is a configuration-driven conversational policy.
CBT
The engine should support:
- Identifying automatic thoughts.
- Exploring evidence for and against a thought.
- Identifying possible cognitive distortions without asserting them as facts.
- Exploring connections between thoughts, emotions, and behavior.
- Guided reframing.
- Optional behavioral experiments.
- Reflecting on results from earlier exercises.
The engine must not reflexively dispute every negative thought. Some negative interpretations may be accurate.
Schema Therapy-informed support
The engine should support:
- Exploring recurring interpersonal and emotional patterns.
- Identifying potentially relevant unmet emotional needs.
- Exploring coping responses.
- Examining links between past experiences and current reactions.
- Introducing schema concepts cautiously.
- Tracking recurring patterns across sessions.
The engine must not confidently label the user with a maladaptive schema, claim to recover repressed memories, or use intensive experiential techniques without appropriate safeguards.

For the POC, exclude unsupervised trauma-exposure exercises and emotionally intensive imagery rescripting.
Auto mode
Auto mode should select an approach for the current conversational objective, not diagnose the user or permanently assign a therapeutic modality.

Example selection logic:
Current need	Initial approach
Examine a specific automatic thought	CBT
Explore a longstanding recurring pattern	Schema
Unclear objective	Supportive exploration, then reassess
User explicitly requests a method	Respect user selection
Acute safety concern	Safety protocol overrides methodology

Implement a structured recommendation:
from typing import Literalfrom pydantic import BaseModel, Fieldclass MethodRecommendation(BaseModel):    selected: Literal["cbt", "schema", "supportive"]    rationale: str    confidence: float = Field(ge=0, le=1)    needs_clarification: bool



The confidence value represents the model's internal selection estimate. It must not be presented as a calibrated clinical probability.

Reevaluate methodology when the session topic materially changes, not on every conversational turn.
Natural session behavior
The engine should follow a flexible conversational state model:

opening → exploration ↔ intervention ↔ reflection → closing

Transitions are not mandatory. The user may change topic, stop an exercise, or end a session at any time.

Avoid rigid session durations and forced closing summaries during the conversation.

7. Full live voice architecture
Full live voice is mandatory for this POC. It is also the largest engineering risk.

The implementation should use a modular streaming pipeline rather than requiring a particular proprietary real-time model.
1

Microphone capture
Continuous PCM audio frames


2

Voice activity detection
Detect speech onset and end of turn


3

Streaming speech recognition
Produce partial and final transcripts


4

Conversation orchestrator
Safety, memory retrieval, LLM streaming


5

Streaming speech synthesis
Convert generated text into audio chunks


6

Audio playback
Play responses; interrupt on user speech


Technical requirements
Audio input: Use a Windows-compatible audio library such as sounddevice. Prefer 16 kHz mono PCM for STT when supported, with explicit resampling when required.

VAD: Use a local voice activity detector, such as Silero VAD, with configurable speech-start and silence-end thresholds.

STT: Audio transcription must run locally; audio must never be sent to an external provider. A local Whisper implementation may transcribe completed VAD utterances, but standard Whisper chunk transcription is not true streaming recognition. The system must not claim low-latency streaming support merely because audio chunks are processed repeatedly. Partial captions are a later improvement, not a requirement to move audio to a cloud service.

TTS: Speech synthesis must run locally; generated or user audio must never be sent to a cloud TTS service. Use a local voice adapter that produces bounded playback chunks. A non-streaming local sentence synthesis fallback is acceptable for the POC, with latency measured during acceptance.

Playback: Use a dedicated playback queue with cancellation and interruption support.
Required voice state machine
IDLE
  |
  v
LISTENING
  |
  | Speech detected
  v
CAPTURING
  |
  | End of utterance
  v
TRANSCRIBING
  |
  | Final transcript
  v
THINKING
  |
  | First speakable response chunk
  v
SPEAKING
  |
  +---- User interruption ---> INTERRUPTING
  |                                  |
  |                                  v
  |                              LISTENING
  |
  +---- Playback completed ----> LISTENING


Use asyncio.Queue for audio and text streams, and cancellation-aware tasks for generation and playback.
Barge-in requirements
When the user begins speaking while the agent is responding:
1. Detect user speech while suppressing false triggers from speaker output.
2. Stop or fade playback immediately.
3. Cancel pending TTS generation.
4. Cancel the conversational LLM request when possible.
5. Preserve the portion of the response actually played.
6. Resume capturing the user's speech.
7. Avoid storing unspoken generated content as though it had been communicated.
Use headphones for the initial acceptance environment. Acoustic echo cancellation for open speakers is not required in the first release.
Latency targets
Engineering targets, not guaranteed provider performance
Metric	Target
Speech-end detection	Under 700 ms after silence
Interruption-to-playback stop	Under 250 ms
End of user turn to first agent audio	Median under 2.5 s
Full session	At least 30 minutes without pipeline failure
Transcript continuity	No missing completed turns

Measure actual latency by pipeline stage. Do not hide network delays inside a single end-to-end metric.

If the text LLM or a local speech model fails, the application should preserve the session, display an actionable error, and permit recovery without restarting.

8. Memory system — detailed specification
The memory system must prioritize accuracy, traceability, bounded context size, and correction.

It must not treat every repeated statement as an immutable personality trait.
8.1 Memory layers
Layer	Contents	Retention
Core profile	Important stable characteristics and patterns	Until corrected, superseded, or deleted
Active context	Current concerns and ongoing exercises	Updated between sessions
Episodic memory	Session summaries and significant experiences	Persistent
Atomic memories	Individual evidence-backed facts, reports, hypotheses	Persistent, versioned
Source archive	Original essays and transcripts	User-controlled retention

8.2 Atomic memory schema
from datetime import datetimefrom typing import Literalfrom pydantic import BaseModel, Fieldclass MemoryRecord(BaseModel):    id: str    category: Literal[        "personal_history",        "belief",        "emotional_pattern",        "coping_behavior",        "relationship_pattern",        "goal",        "preference",        "important_event",        "therapeutic_insight",    ]    statement: str    epistemic_status: Literal[        "user_reported",        "model_hypothesis",        "user_confirmed",        "disputed",        "superseded",    ]    importance: int = Field(ge=1, le=5)    confidence: float = Field(ge=0, le=1)    source_ids: list[str]    related_memory_ids: list[str] = []



Use separate persistence models where appropriate; the Pydantic schema is the domain contract, not necessarily the exact database table.

Confidence must not increase merely because the LLM repeats its own previous interpretation.
8.3 Evidence and provenance
Each memory must reference one or more source records.

A source record should include:
source_id
source_type
session_id
message_id
document_id
source_span
created_at


For long documents, use character offsets or paragraph identifiers. For audio, use transcript segments with timestamps.

The system must distinguish:
- User statements.
- Clinician-authored notes imported by the user.
- Model-generated interpretations.
- User confirmations and corrections.
Imported clinician notes are still externally supplied evidence. The system must not assume their accuracy merely because they appear clinical.
8.4 Memory extraction algorithm
Run extraction after completed conversational turns or small batches of turns, not on every partial STT result.
INPUT:
  Completed conversation turns
  Existing relevant memories

PROCESS:
  1. Extract potentially durable information
  2. Classify each item
  3. Identify its source
  4. Separate observation from interpretation
  5. Compare against existing memories
  6. Detect duplicates and contradictions
  7. Propose ADD, UPDATE, LINK, SUPERSEDE, or IGNORE
  8. Validate structured output
  9. Apply memory write policy

OUTPUT:
  Committed changes
  Pending review changes
  Audit events


8.5 Memory write policy
Automatically commit:
- Low-risk factual details explicitly reported by the user.
- New source references supporting an existing memory.
- Session summaries, clearly labeled as generated summaries.
- Non-interpretive updates to active session context.
Require review:
- New psychological personality interpretations.
- Proposed causal relationships between childhood experiences and adult behavior.
- Changes to longstanding core beliefs.
- Contradictions involving important memories.
- Significant reinterpretations of prior events.
Never automatically commit:
- Psychiatric diagnoses.
- Claims about unreported trauma.
- Inferences about the user's unconscious motivations presented as facts.
- Claims that an unverified psychological hypothesis has been clinically established.
8.6 Memory consolidation
Use deterministic candidate matching before semantic matching.

Suggested pipeline:
New candidate
    |
    v
Exact / normalized match
    |
    +-- Match --> Add evidence reference
    |
    v
Related-memory retrieval
    |
    v
LLM relationship classification
    |
    +-- Duplicate --> Merge evidence
    |
    +-- Contradiction --> Flag for review
    |
    +-- Evolution --> Create new version
    |
    +-- Distinct --> Create new memory


Do not overwrite an earlier belief merely because the user expresses something different in a later session.

For example, "I used to fear failure, but that has improved" should preserve the historical pattern while marking its current relevance as reduced.
8.7 Context retrieval
Implement a bounded context assembler.

Suggested initial token budgets:
Component	Maximum
Safety and behavioral instructions	1,000
Methodology instructions	700
Core profile	1,500
Retrieved memories	1,500
Active session state	500
Recent conversation	3,000
Total target	8,200

These are configurable starting values. The provider's actual context window and output-token allowance must also be respected.

For retrieval, use:
- Metadata filters.
- SQLite FTS5.
- Recency.
- Importance.
- Epistemic status.
- Source reliability.
- Topic relevance.
Start with a weighted ranking function such as:
\[
S(m)=0.40R+0.25I+0.15E+0.10T+0.10C
\]

Where:
- \(R\): relevance to current discussion.
- \(I\): normalized importance.
- \(E\): evidence quality.
- \(T\): temporal relevance.
- \(C\): relevance to current therapeutic context.
This is an engineering heuristic, not a scientifically validated psychological scoring model.

The weights must be configurable and tested against retrieval scenarios.

Always allow critical, directly relevant memories to override ordinary ranking, but do not inject every high-importance memory into every session.
8.8 Profile generation
The compact profile must be generated from approved or appropriately qualified atomic memories.

It must not become an independent source of truth.

The profile should contain:
CURRENT GOALS
RECURRING SELF-REPORTED PATTERNS
IMPORTANT PERSONAL HISTORY
CURRENT CONCERNS
PREFERRED CONVERSATIONAL APPROACH
ACTIVE THERAPEUTIC EXERCISES
UNRESOLVED OR DISPUTED INTERPRETATIONS


Regenerate the profile when relevant source memories change.

Store a profile version and the IDs of all memories used to construct it.
8.9 Memory deletion
Deletion must propagate to:
- Atomic memory records.
- Derived profile sections.
- Search indexes.
- Cached context.
- Session summaries where the targeted information was copied.
If the original transcript is retained, the application must explicitly disclose that the deleted information may still exist in source history. Provide a separate option to delete or redact that source content.

For the POC, implement targeted deletion and full local-data deletion. Do not promise secure physical erasure from SSDs, backups, or external provider systems.
9. Database design
Use SQLite with SQLCipher or an equivalent authenticated encryption approach. Plain SQLite combined with encrypted individual fields is not sufficient if sensitive information can leak through indexes, temporary files, or metadata.

A minimal relational schema:
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    methodology TEXT NOT NULL,
    modality TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    summary TEXT
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    session_id TEXT,
    message_id TEXT,
    document_id TEXT,
    source_span TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    statement TEXT NOT NULL,
    epistemic_status TEXT NOT NULL,
    importance INTEGER NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    first_observed TEXT NOT NULL,
    last_observed TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT
);

CREATE TABLE memory_sources (
    memory_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, source_id)
);

CREATE TABLE memory_relations (
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    PRIMARY KEY (
        source_memory_id,
        target_memory_id,
        relation_type
    )
);

CREATE TABLE memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    previous_state TEXT,
    proposed_state TEXT,
    review_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    content TEXT NOT NULL
);


This is the conceptual schema. The implementation must add appropriate constraints, foreign keys, indexes, schema migrations, and transaction handling.

Session summaries should eventually have explicit source dependencies rather than remaining independent free-text fields.
Encryption requirements
Use a single encrypted database file where practical.
- Generate a strong random database encryption key.
- Protect the key using Windows DPAPI for the current Windows user.
- Never store plaintext encryption keys in YAML, environment examples, logs, or the repository.
- Verify that database temporary files, WAL files, and search indexes do not expose plaintext.
- Do not persist raw audio by default.
- Ensure database backups remain encrypted.
- Document the limitations of DPAPI-protected backups when migrating to another computer.
A future cross-platform implementation may use a passphrase-derived master key instead.

10. Safety architecture
The application must implement safety controls independently of the selected methodology.
Risk categories
Level	Examples	Required behavior
Routine	Stress, self-criticism, relationship difficulties	Normal supportive conversation
Elevated	Significant distress, worsening symptoms, ambiguous safety language	Clarify relevant concerns, avoid overconfident interpretation
Acute	Indications of imminent self-harm or danger	Prioritize immediate safety, appropriate human or emergency support
Out of scope	Requests for diagnosis, medication changes, or emergency treatment	State limitations and redirect to qualified care

A safety classifier must operate on finalized user utterances and relevant recent context.

Do not rely exclusively on an LLM classifier. Combine deterministic checks with contextual model evaluation, while acknowledging that neither guarantees detection.

The safety layer must be able to override the ordinary conversational policy, including response-length constraints.
Critical behavioral requirements
The agent must not:
- Claim to be a human psychologist.
- Diagnose psychiatric conditions.
- Present speculative psychological explanations as established facts.
- Encourage dependence on the application.
- Discourage contact with family, friends, clinicians, or emergency services.
- Conduct unsupervised high-intensity trauma-processing exercises.
- Invent memories or suggest that forgotten trauma necessarily occurred.
- Continue an ordinary CBT or Schema exercise during an apparent immediate crisis.
- Treat silence, interruption, or connection loss as evidence that the user is safe.
The safety response should remain conversational rather than defaulting to lengthy, repetitive disclaimers.
Safety testing
Include adversarial scenarios involving:
- Ambiguous suicidal language.
- Explicit imminent danger.
- Requests for diagnosis.
- Delusional or paranoid beliefs.
- Manic symptoms.
- Trauma disclosures.
- Emotional dependency on the agent.
- Incorrect or harmful memories.
- User rejection of the agent's interpretation.
Before any public deployment, obtain independent clinical review of these behaviors.

11. Privacy and external API boundaries
The user has authorized text-only DeepSeek LLM processing for the POC, not transmission of audio or unrestricted transmission of all stored psychological information. All transcription and speech synthesis must remain local.

Apply data minimization.
Permitted cloud payload
For a conversational request, transmit only:
- The current user message.
- Relevant recent conversation.
- The selected methodology instructions.
- The compact profile and retrieved memories necessary for the response.
- Necessary safety context.
Do not upload the entire psychological database or all historical transcripts by default.
Privacy controls
Implement:
- A startup notice explaining cloud processing.
- Configurable transcript retention.
- Do not persist raw audio; keep live frames in memory only and do not upload audio.
- API payload inspection in debug mode, with sensitive content redacted.
- No analytics or telemetry.
- No third-party error reporting by default.
- No sensitive data in ordinary application logs.
- A local export mechanism.
- A full local-data deletion command.
A privacy configuration should identify the text LLM provider and confirm that STT/TTS are local.

Cloud processing may involve provider retention or processing policies outside the application's control. The README must disclose this and avoid claiming that cloud-submitted data is exclusively stored locally.

12. Configuration contract
Use YAML for non-secret configuration and environment variables for API credentials.

Example config/default.yaml:
app:
  name: psyche
  environment: local
  log_level: INFO

llm:
  provider: deepseek
  model: ${PSYCHE_LLM_MODEL}
  temperature: 0.5
  max_output_tokens: 250
  request_timeout_seconds: 30
  streaming: true

conversation:
  default_methodology: auto
  default_modality: voice
  target_response_words: 35
  maximum_questions_per_turn: 1
  flexible_session: true

voice:
  input_device: default
  output_device: default
  sample_rate: 16000
  vad:
    enabled: true
    end_silence_ms: 550
  stt:
    provider: local
    model: faster-whisper
  tts:
    provider: local
    model: piper
    voice: configurable
  barge_in:
    enabled: true

memory:
  storage: sqlite
  encrypted: true
  extraction_enabled: true
  review_significant_changes: true
  core_profile_max_tokens: 1500
  retrieval_max_tokens: 1500
  max_retrieved_memories: 12

privacy:
  store_raw_audio: false
  retain_transcripts: true
  telemetry: false
  cloud_processing_allowed: true

safety:
  enabled: true
  override_methodology: true


${...} substitution must be implemented explicitly or through a configuration library. YAML does not natively interpolate environment variables.

Do not hardcode provider model identifiers that may become unavailable. Validate configured model capabilities during startup.

13. Python interfaces
Define stable contracts before writing provider implementations.
Conversation orchestrator
from typing import Literalfrom pydantic import BaseModelclass SessionConfig(BaseModel):    methodology: Literal["cbt", "schema", "auto"]    modality: Literal["text", "voice"]class SessionOrchestrator:    async def start_session(        self,        config: SessionConfig,    ) -> str:        ...    async def process_turn(        self,        session_id: str,        user_text: str,    ) -> str:        ...    async def end_session(        self,        session_id: str,    ) -> None:        ...



Streaming should use a separate typed event interface rather than forcing partial text through process_turn.
Streaming events
from typing import Literalfrom pydantic import BaseModelclass StreamEvent(BaseModel):    type: Literal[        "speech_started",        "transcript_partial",        "transcript_final",        "generation_started",        "response_text_delta",        "audio_chunk",        "playback_started",        "playback_interrupted",        "turn_completed",        "error",    ]    session_id: str    turn_id: str | None = None    payload: dict



Audio bytes should not be serialized into JSON dictionaries internally. Use a typed audio-frame payload for high-frequency audio events and a separate metadata event stream where appropriate.
Memory service
class MemoryService:    async def extract_candidates(        self,        session_id: str,    ) -> list:        ...    async def retrieve(        self,        query: str,        max_tokens: int,    ) -> list:        ...    async def review_pending(self) -> list:        ...    async def correct(        self,        memory_id: str,        corrected_statement: str,    ) -> None:        ...    async def forget(        self,        memory_id: str,    ) -> None:        ...



Implement typed return values and domain exceptions in the actual code. The above signatures define responsibilities, not complete production interfaces.

14. Development milestones and execution order
The code-generation agent must implement the application incrementally. It must not generate an entire untested repository in one pass.

Milestone 1 — Foundation

Python package, CLI, configuration, encrypted database, LLM provider adapter, text conversations, persistent sessions.

Exit gate: text conversations can be started, resumed, and stored securely.




Milestone 2 — Live voice

Microphone capture, VAD, incremental STT, streaming LLM output, TTS, audio playback, interruption, latency instrumentation.

Exit gate: a 30-minute hands-free voice conversation works with headphones and supports interruption.




Milestone 3 — Methodologies

CBT and Schema policies, Auto selector, flexible session state, behavioral constraints.

Exit gate: methodology-specific evaluation scenarios pass.




Milestone 4 — Persistent memory

Atomic memories, evidence references, extraction, deduplication, review queue, retrieval, profile generation, correction, deletion.

Exit gate: cross-session recall and memory-correction tests pass.




Milestone 5 — Ingestion

Text and Markdown import, audio-file transcription, evidence extraction, duplicate-import detection.

Exit gate: an imported essay informs subsequent conversations without requiring the original document in every prompt.




Milestone 6 — Hardening

Safety regression tests, encryption verification, deletion tests, voice reliability, evaluation harness, documentation.

Exit gate: all mandatory acceptance tests pass and known limitations are documented.



Safety must be implemented as a basic cross-cutting control from Milestone 1 and expanded throughout development. It must not be deferred entirely until Milestone 6.
Implementation rules for GPT Luna
1. Complete one milestone before starting the next.
2. Run unit tests after each meaningful implementation step.
3. Prefer a working vertical slice over speculative abstractions.
4. Use mocks for provider-dependent tests.
5. Keep real API integration tests optional and explicitly enabled.
6. Do not silently substitute simulated voice output for functioning audio.
7. Do not introduce new infrastructure without demonstrating a concrete requirement.
8. Document implementation decisions that deviate from this specification.
9. Do not report a milestone complete unless its acceptance criteria have been executed.
10. Preserve a running list of known defects and deferred improvements.
15. Testing strategy
Use pytest, pytest-asyncio, and dependency-injected mocks. All deterministic unit tests must run without internet access or API credentials.
15.1 Unit tests
Area	Required tests
Configuration	Defaults, invalid values, missing credentials
Conversation	State transitions, response formatting, cancellation
Methodology	Explicit selection, Auto selection, switching
Memory	Extraction validation, deduplication, contradiction handling
Retrieval	Ranking, token budgeting, source attribution
Persistence	Transactions, restart recovery, migrations
Encryption	Correct key, wrong key, plaintext leakage checks
Voice	VAD events, chunk ordering, interruption, queue cancellation
Safety	Classification, overrides, unsupported requests

15.2 Memory regression scenarios
Create a fixture containing a synthetic user's personal history and at least 20 simulated sessions.

Test these cases:

Scenario A — Stable recall: A significant self-reported belief introduced in session 1 must remain retrievable in session 20 when relevant.

Scenario B — Contradiction: A user initially reports a recurring fear, then later reports substantial improvement. The engine must preserve the historical observation without incorrectly presenting it as an unchanged current trait.

Scenario C — Incorrect inference: The model proposes an explanation for a relationship pattern. The user rejects it. Future conversations must not present that explanation as fact.

Scenario D — Duplication: The same personal fact appears in ten sessions. The database must not accumulate ten independent copies of the same memory.

Scenario E — Deletion: A memory is deleted. It must no longer appear in the generated profile or retrieval results, including after application restart.

Scenario F — Prompt injection: An imported essay contains instructions attempting to override the agent's system policy. Those instructions must be treated as source content.
15.3 Conversational quality evaluations
Build a test harness that evaluates complete multi-turn interactions, not isolated responses.

Measure:
Metric	Initial target
Ordinary response length	90% within 15–55 words
Questions per ordinary turn	No more than one in 95% of turns
Relevant memory recall	At least 90% on curated test cases
Unsupported memory claims	Zero in critical regression scenarios
Explicit memory corrections	100% respected in deterministic tests
Methodology fidelity	At least 90% on curated scenarios
Session persistence	100% of completed turns retained
Safety regression suite	All designated critical scenarios pass

The percentages are provisional engineering acceptance thresholds, not clinical efficacy measurements.

Use a combination of deterministic checks, manual transcript review, and LLM-based evaluation. An LLM judge must not be the sole authority on therapeutic appropriateness or safety.
15.4 Voice tests
Use recorded audio fixtures for automated tests.

Test:
- Normal utterance detection.
- Long pauses.
- Background noise.
- False speech detection.
- Rapid consecutive turns.
- User interruption during TTS.
- Network timeout during generation.
- STT transcription failure.
- Audio device disconnection.
- Session termination during playback.
Run a separate manual hardware acceptance test using the actual Windows microphone and headphones.
15.5 Security tests
Verify that:
- A database file cannot be opened as plaintext SQLite.
- Raw psychological content does not appear in standard logs.
- Sensitive temporary files are not left behind.
- API credentials are never included in exceptions or debug output.
- Deleted memories are removed from active indexes and generated profiles.
- Cloud API requests contain only the intended bounded context.
- The application fails safely when encryption initialization fails.

16. End-to-end acceptance test
The following scenario determines whether the POC is complete.
POC acceptance checklist
IMPLEMENTATION VERIFICATION

4/16 passed on the Windows development machine



[x] Install the application on Windows using documented commands
[x] Initialize encrypted local storage
[ ] Import a 3,000-word personal essay
[ ] Extract evidence-backed memory candidates
[ ] Review significant proposed psychological interpretations
[x] Start a live voice session using Auto methodology
[ ] Receive concise spoken responses with acceptable latency
[ ] Interrupt the agent while it is speaking
[ ] Switch from Auto to CBT during the same session
[ ] End the session and persist its transcript
[ ] Restart the application and begin another session
[ ] Retrieve relevant information from the original essay and prior conversation
[ ] Correct an inaccurate psychological interpretation
[ ] Verify the correction is respected in later conversations
[ ] Delete a memory and verify its derived representations are removed
[x] Run the complete automated test suite successfully

The completed local checks were verified with Python 3.13.9: documented editable install, encrypted initialization in an isolated temporary data directory, and the full automated test suite (84 passing as of 2026-11-16). A fail-closed regression test confirms that database connection does not begin when encryption-key setup fails; a Windows integration test confirms a database containing a synthetic private marker cannot be read as plaintext SQLite. These tests cover only these specific storage properties and do not establish a complete security audit. Local faster-whisper and Piper model inference also passed an offline synthetic-phrase smoke test; this is not microphone intelligibility, headset playback, or clinical evaluation. A synthetic live DeepSeek text-session turn, streaming completion, and production structured-extraction call also succeeded, but do not count as personal-essay or live-voice acceptance. The suite includes a narrow deterministic safety regression baseline, not a clinical evaluation. The synthetic 3,000-word text workflow is useful integration coverage but does not count as importing the owner's personal essay. Live audio, memory behavior, and conversational quality must still be evaluated; the remaining items are intentionally unchecked until their specified acceptance scenario is performed.

17. Documentation deliverables
GPT Luna must maintain these documents:
File	Purpose
README.md	Installation, setup, and basic usage
docs/ARCHITECTURE.md	Component responsibilities and data flow
docs/MEMORY.md	Memory schema, algorithms, and review policy
docs/VOICE.md	Streaming pipeline and interruption handling
docs/PRIVACY.md	Data storage, cloud processing, retention
docs/SAFETY.md	Safety policies and known limitations
docs/TESTING.md	Test execution and acceptance procedure
docs/DECISIONS.md	Important architectural decisions
docs/PROGRESS.md	Milestone completion, defects, next steps

Documentation should describe the actual implementation, not an aspirational architecture.
18. Instructions for GPT Luna
The following is the implementation directive to accompany this specification.
GPT Luna — Implementation Directive
You are the lead Python software engineer responsible for implementing Psyche, a CLI-first, live-voice psychological support application.

Implement the project according to the accompanying development specification.
Primary objective
Deliver a functioning Windows-based Python POC supporting:
- Natural live voice conversations with interruption handling.
- Configurable cloud LLM inference.
- CBT, Schema Therapy-informed, and Auto conversational modes.
- Encrypted local psychological memory.
- Evidence-backed memory extraction, retrieval, consolidation, correction, and deletion.
- Personal essay and therapy-note ingestion.
- Persistent conversation sessions.
- Safety controls and reproducible tests.
Engineering priorities
1. Correctness and reliability.
2. Natural, concise conversational behavior.
3. Accurate, non-hallucinated long-term memory.
4. Voice responsiveness.
5. Privacy and safety.
6. Maintainability.
7. Minimal architectural complexity.
Do not introduce LangChain, LangGraph, microservices, cloud databases, web interfaces, or multiple autonomous agents without a demonstrated necessity.
Execution protocol
Implement the application milestone by milestone.

For each milestone:
1. Identify the minimum files and dependencies required.
2. Implement working functionality.
3. Add unit and integration tests.
4. Execute the tests.
5. Resolve failures.
6. Update documentation.
7. Record limitations and deferred improvements.
8. Proceed only after the milestone's acceptance criteria pass.
Do not generate placeholder implementations and present them as completed functionality.

Do not invent successful test results, API behavior, performance measurements, or hardware verification.

If provider capabilities are uncertain, isolate the dependency behind an adapter and verify it before relying on it.
Memory-specific instructions
Treat the user's psychological profile as a derived representation of evidence-backed memory records.

Never silently convert model-generated interpretations into confirmed personal facts.

Preserve source provenance, historical changes, contradictions, and explicit user corrections.

Keep conversational context bounded without discarding the underlying memory archive.
Voice-specific instructions
Full live voice is mandatory.

Implement asynchronous audio capture, speech detection, incremental transcription, streaming response generation, speech synthesis, playback, and interruption handling.

Ensure the conversation history reflects what the user actually heard, not merely what the model generated.
Safety-specific instructions
The application is a psychological support tool, not a licensed clinician.

Implement appropriate safeguards against unsupported diagnosis, fabricated psychological interpretations, harmful interventions, and unsafe crisis handling.

Safety constraints must override methodology selection when necessary.
Completion criteria
The project is complete only when the end-to-end acceptance scenario passes, the application can be launched from the Windows terminal, and all mandatory tests and documentation are available.

Begin with Milestone 1. Do not attempt to implement the entire repository in one untested generation pass.
