import asyncio
import hashlib
from importlib.util import find_spec
import json
import re
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .config import load_config
from .storage.database import Database
from .storage.encryption import EncryptionError
from .llm.cloud import OpenAIProvider, ProviderError
from .conversation.orchestrator import ConversationOrchestrator
from .memory.extractor import MemoryExtractor
from .voice.audio import AudioDeviceError, SoundDeviceMicrophone, negotiate_input_format
from .voice.diagnostics import capture_input_levels
from .voice.playback import SoundDeviceAudioPlayer
from .voice.pipeline import VoiceTurnPipeline
from .voice.synthesis import OpenAITTSSynthesizer
from .voice.transcription import OpenAIRealtimeTranscriber
from .voice.vad import UtteranceSegmenter, WebRtcSpeechClassifier

app = typer.Typer(help="Local-first psychological support companion", no_args_is_help=True)
session_app = typer.Typer(help="Manage conversation sessions")
sessions_app = typer.Typer(help="Inspect stored sessions")
config_app = typer.Typer(help="View configuration")
memory_app = typer.Typer(help="Review and manage memory")
audio_app = typer.Typer(help="Inspect local audio devices")
app.add_typer(session_app, name="session")
app.add_typer(sessions_app, name="sessions")
app.add_typer(config_app, name="config")
app.add_typer(memory_app, name="memory")
app.add_typer(audio_app, name="audio")
console = Console()


def services():
    load_dotenv()
    config = load_config()
    db = Database(config.data_dir)
    return config, db


def _require_cloud_processing(config, db=None) -> None:
    if not getattr(getattr(config, "privacy", None), "cloud_processing_allowed", True):
        if db is not None:
            db.close()
        raise typer.BadParameter(
            "Cloud processing is disabled in configuration; this app has no local conversation provider."
        )


@app.command()
def init():
    """Initialize encrypted local storage."""
    try:
        _, db = services()
        db.close()
    except (EncryptionError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print("Encrypted local storage initialized.")
    console.print("Cloud conversations send bounded conversation context to the configured LLM provider.")


@config_app.command("show")
def config_show():
    """Display effective configuration without credentials."""
    config = load_config()
    console.print_json(json.dumps(config.model_dump(mode="json"), indent=2))


@app.command()
def doctor():
    """Check configuration and encrypted database availability."""
    try:
        config, db = services()
        db.close()
        console.print(f"[green]Encrypted database: OK[/green] ({config.data_dir})")
    except Exception as exc:
        console.print(f"[red]Storage check failed:[/red] {exc}")
        raise typer.Exit(1)
    audio_ready = all(find_spec(name) for name in ("sounddevice", "webrtcvad"))
    console.print("Audio capture/VAD dependencies: " + ("available" if audio_ready else "missing (install the optional voice dependencies)"))
    console.print("Live voice: " + ("experimental; provider/device acceptance still needs a real run" if audio_ready else "not ready; install optional voice dependencies"))
    console.print("LLM credentials: " + ("present (quota/connectivity not checked)" if __import__("os").getenv("PSYCHE_LLM_API_KEY") or __import__("os").getenv("OPENAI_API_KEY") else "missing (text sessions require a key)"))


def _audio_device_value(value: str | int | None) -> str | int | None:
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


@audio_app.command("check")
def audio_check(
    seconds: float = typer.Option(5.0, "--seconds", min=1.0, max=15.0),
    device: str | None = typer.Option(None, "--device", help="Input device index or name (defaults to voice.input_device / Windows default)."),
):
    """Measure local microphone levels without saving or uploading audio."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise typer.BadParameter("Install voice support with `pip install -e '.[voice]'`.") from exc

    config = load_config()
    selected_device = _audio_device_value(device if device is not None else config.voice.input_device)
    try:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        default_input = sd.default.device[0]
    except Exception as exc:
        console.print(f"[red]Could not enumerate audio devices:[/red] {exc}")
        raise typer.Exit(1) from exc

    device_table = Table("Index", "Input device", "Host API", "Channels", "Default")
    for index, info in enumerate(devices):
        input_channels = int(info.get("max_input_channels", 0))
        if input_channels < 1:
            continue
        api_index = int(info.get("hostapi", -1))
        api_name = host_apis[api_index].get("name", "unknown") if 0 <= api_index < len(host_apis) else "unknown"
        device_table.add_row(
            str(index), str(info.get("name", "Unknown")), api_name, str(input_channels),
            "yes" if index == default_input else "",
        )
    console.print(device_table)

    try:
        info, sample_rate, channels = negotiate_input_format(sd, selected_device, config.voice.sample_rate)
    except AudioDeviceError as exc:
        console.print(f"[red]Microphone check failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    api_index = int(info.get("hostapi", -1))
    api_name = host_apis[api_index].get("name", "unknown") if 0 <= api_index < len(host_apis) else "unknown"
    console.print(
        f"Selected input: #{info.get('index', '?')} {info.get('name', 'Unknown')} "
        f"({api_name}); capture format {sample_rate} Hz, {channels} channel(s)."
    )
    console.print(
        f"Capturing for about {seconds:g} seconds now. Speak a normal sentence at your usual distance. "
        "Audio stays in memory for level measurements only; it is not saved or uploaded."
    )
    try:
        result = capture_input_levels(sd, selected_device, config.voice.sample_rate, seconds)
    except AudioDeviceError as exc:
        console.print(f"[red]Microphone check failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    for index, level in enumerate(result["channel_levels"], start=1):
        dbfs = f"{level['dbfs']} dBFS" if level["dbfs"] is not None else "silence"
        console.print(
            f"Channel {index}: RMS {level['rms']} ({dbfs}), peak {level['peak']}, "
            f"nonzero {level['nonzero']}/{level['samples']} samples, clipped {level['clipped']}"
        )
    mixed = result["mixed_level"]
    mixed_dbfs = f"{mixed['dbfs']} dBFS" if mixed["dbfs"] is not None else "silence"
    console.print(
        f"Psyche mono mix: RMS {mixed['rms']} ({mixed_dbfs}), peak {mixed['peak']}, "
        f"nonzero {mixed['nonzero']}/{mixed['samples']} samples, clipped {mixed['clipped']}"
    )
    if result["statuses"]:
        console.print(f"[yellow]Input status notices:[/yellow] {', '.join(result['statuses'])}")
    if mixed["peak"] == 0:
        console.print("[red]No nonzero signal reached Psyche's mono capture mix during this check.[/red]")
    else:
        console.print("Signal detected. This level check does not determine whether speech is intelligible.")


@session_app.command("start")
def session_start(mode: str = typer.Option("text", "--mode"), method: str = typer.Option("auto", "--method")):
    """Start an interactive text or experimental voice conversation."""
    if mode not in {"text", "voice"}:
        raise typer.BadParameter("Mode must be text or voice.")
    if method not in {"auto", "cbt", "schema"}:
        raise typer.BadParameter("Method must be auto, cbt, or schema.")
    db = None
    try:
        config, db = services()
        _require_cloud_processing(config, db)
        provider = OpenAIProvider(config.llm.model, config.llm.request_timeout_seconds)
    except (EncryptionError, ProviderError) as exc:
        if db is not None:
            db.close()
        raise typer.BadParameter(str(exc)) from exc
    orch = ConversationOrchestrator(db, provider, config)
    if mode == "voice":
        if not audio_ready_for_session():
            db.close()
            raise typer.BadParameter("Install voice dependencies with `pip install -e '.[voice]'`.")
        sid = asyncio.run(orch.start_session(method, "voice"))
        console.print(f"Voice session {sid} · {method}. Speak to take turns; press Ctrl+C to stop.")
        _run_voice(orch, db, sid, config)
        return
    sid = asyncio.run(orch.start_session(method))
    console.print(f"Session {sid} · {method}. Type /help for commands.")
    _run_chat(orch, db, sid)


def audio_ready_for_session() -> bool:
    return all(find_spec(name) for name in ("sounddevice", "webrtcvad"))


def _run_voice(orch, db, sid, config):
    voice = config.voice
    microphone = SoundDeviceMicrophone(
        sample_rate=voice.sample_rate, frame_ms=voice.frame_ms, device=voice.input_device,
    )
    vad = UtteranceSegmenter(
        WebRtcSpeechClassifier(voice.vad.aggressiveness), sample_rate=voice.sample_rate,
        frame_ms=voice.frame_ms, end_silence_ms=voice.vad.end_silence_ms,
        pre_roll_ms=voice.vad.pre_roll_ms,
    )
    def render_voice_event(event):
        if event.type.value == "speech_started":
            console.print("\n[bold]You (live):[/bold] ", end="")
        elif event.type.value == "transcript_partial":
            console.print(event.payload.get("text", ""), end="", style="dim")
        elif event.type.value == "transcript_final":
            console.print(f"\n[bold]You (final):[/bold] {event.payload.get('text', '')}")
        elif event.type.value == "response_text_delta":
            if event.payload.get("first"):
                console.print("[bold cyan]Psyche:[/bold cyan] ", end="")
            console.print(event.payload.get("text", ""), end="", style="cyan")
        elif event.type.value in {"turn_completed", "playback_interrupted"}:
            console.print()

    pipeline = VoiceTurnPipeline(
        orch, microphone, vad,
        OpenAIRealtimeTranscriber(voice.stt_model, config.llm.request_timeout_seconds),
        OpenAITTSSynthesizer(voice.tts_model, voice.tts_voice, config.llm.request_timeout_seconds),
        SoundDeviceAudioPlayer(sample_rate=24000, device=voice.output_device),
        on_event=render_voice_event,
    )
    console.print(
        "Cloud processing notice: voice audio is sent for transcription; the transcript, bounded recent "
        "context, current profile, relevant approved memories, and prompts are sent to your configured provider. "
        "Raw audio is not stored locally; provider retention is outside this app's control. "
        "Echo suppression is not implemented, so headphones are recommended."
    )
    handed_off = False
    try:
        while True:
            try:
                asyncio.run(pipeline.run(sid))
                break
            except (KeyboardInterrupt, EOFError):
                asyncio.run(orch.end_session(sid))
                console.print("Voice session saved and ended.")
                break
            except Exception as exc:
                console.print(f"[red]Voice turn failed:[/red] {exc}")
                console.print("Completed user turns and any fully spoken response prefix remain saved.")
                try:
                    action = typer.prompt(
                        "Choose: [r]etry listening, [t]ext mode in this session, or [e]nd",
                        default="t",
                    ).strip().casefold()
                except (EOFError, KeyboardInterrupt):
                    action = "e"
                if action in {"r", "retry"}:
                    continue
                if action in {"t", "text"}:
                    handed_off = True
                    _run_chat(orch, db, sid)
                    return
                asyncio.run(orch.end_session(sid))
                console.print("Voice session saved and ended.")
                break
    finally:
        if not handed_off:
            db.close()


@session_app.command("resume")
def session_resume(session_id: str):
    """Continue an existing session by its ID."""
    db = None
    try:
        config, db = services()
        if not db.session_exists(session_id):
            db.close()
            raise typer.BadParameter("Session not found.")
        _require_cloud_processing(config, db)
        provider = OpenAIProvider(config.llm.model, config.llm.request_timeout_seconds)
    except (EncryptionError, ProviderError) as exc:
        if db is not None:
            db.close()
        raise typer.BadParameter(str(exc)) from exc
    _run_chat(ConversationOrchestrator(db, provider, config), db, session_id)


def _run_chat(orch, db, sid):
    console.print(
        "Cloud processing notice: your messages, bounded recent session context, current profile, "
        "relevant approved memories, and system/method instructions are sent to your configured LLM provider. "
        "Provider retention is outside this app's control."
    )
    try:
        while True:
            text = typer.prompt("You", prompt_suffix=": ", default="", show_default=False)
            command = text.strip().casefold()
            if command == "/help":
                console.print("/pause, /resume, /method [auto|cbt|schema], /memory, /end, /help")
            elif command == "/pause":
                console.print("Paused. Type /resume to continue or /end to save and finish.")
                while typer.prompt("Paused", default="/resume", show_default=False).strip().casefold() != "/resume":
                    pass
            elif command == "/resume":
                console.print("Session active.")
            elif command == "/method":
                method = typer.prompt("Method (auto/cbt/schema)").strip().casefold()
                if method not in {"auto", "cbt", "schema"}:
                    console.print("Choose auto, cbt, or schema.")
                else:
                    db.conn.execute("UPDATE sessions SET methodology=? WHERE id=?", (method, sid))
                    db.conn.commit()
                    console.print(f"Method set to {method}.")
            elif command == "/memory":
                for row in db.list_memories("approved"):
                    console.print(f"{row[0]} · {row[2]} [{row[3]}]")
            elif command == "/end":
                asyncio.run(orch.end_session(sid))
                console.print("Session saved and ended.")
                return
            elif text.strip():
                try:
                    reply = asyncio.run(orch.process_turn(sid, text))
                    console.print(f"[cyan]Psyche:[/cyan] {reply}")
                except ProviderError as exc:
                    console.print(f"[red]{exc}[/red] Your user turn is saved; retry by sending another message.")
    except (EOFError, KeyboardInterrupt):
        console.print("Session left open and saved. Resume it with `psyche session resume ID`.")
    finally:
        db.close()


@sessions_app.command("list")
def sessions_list():
    """List previous sessions."""
    _, db = services()
    table = Table("ID", "Method", "Mode", "Started", "Status")
    for row in db.list_sessions():
        table.add_row(*[str(value) for value in row])
    console.print(table)
    db.close()


@sessions_app.command("show")
def sessions_show(session_id: str):
    """Display a session transcript."""
    _, db = services()
    if not db.session_exists(session_id):
        db.close()
        raise typer.BadParameter("Session not found.")
    for message in db.get_message_records(session_id):
        note = {
            "interrupted": " [spoken prefix; interrupted]",
            "incomplete": " [spoken prefix; response incomplete]",
        }.get(message["delivery_status"], "")
        console.print(f"[bold]{message['role']}:[/bold] {message['content']}{note}")
    db.close()


@app.command("import")
def import_file(file: Path):
    """Import a UTF-8 text/Markdown source as verbatim review candidates."""
    if file.suffix.casefold() not in {".txt", ".md", ".markdown"}:
        raise typer.BadParameter("Only UTF-8 text and Markdown imports are currently supported.")
    try:
        content = file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise typer.BadParameter("Could not read the file as UTF-8 text.") from exc
    if not content.strip():
        raise typer.BadParameter("The source file is empty.")
    _, db = services()
    doc_id = db.import_document(file.name, content, hashlib.sha256(content.encode("utf-8")).hexdigest())
    if doc_id is None:
        db.close()
        console.print("Already imported: identical content hash.")
        return
    added, linked, unchanged = _add_verbatim_candidates(db, doc_id, content)
    db.close()
    console.print(
        f"Imported source {doc_id}; {added} new candidates pending review, "
        f"{linked} existing candidates linked, {unchanged} already-linked candidates unchanged."
    )


def _add_verbatim_candidates(db, document_id: str, content: str) -> tuple[int, int, int]:
    candidates = [match for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", content)
                  if len(match.group().strip()) >= 35
                  and not match.group().strip().startswith(("#", ">"))]
    added = linked = unchanged = 0
    for match in candidates:
        statement = match.group().strip()[:2000]
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        end = start + len(statement)
        _, created, new_evidence = db.add_memory_with_provenance(
            statement, "personal_history", document_id, "imported_document",
            f"char:{start}-{end}", status="pending",
        )
        if created:
            added += 1
        elif new_evidence:
            linked += 1
        else:
            unchanged += 1
    return added, linked, unchanged


@app.command("import-audio")
def import_audio(
    file: Path = typer.Argument(...),
    model: str = typer.Option("whisper-1", "--model"),
):
    """Transcribe an audio file via the configured cloud provider; raw audio is not retained."""
    allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".oga", ".ogg", ".flac"}
    if file.suffix.casefold() not in allowed_extensions:
        raise typer.BadParameter("Use a supported audio file: mp3, wav, m4a, mp4, mpeg, mpga, webm, oga, ogg, or flac.")
    if not file.is_file():
        raise typer.BadParameter("Audio file not found.")
    try:
        if file.stat().st_size > 25 * 1024 * 1024:
            raise typer.BadParameter("Audio files must be 25 MB or smaller.")
        config, db = services()
    except EncryptionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        if not config.privacy.cloud_processing_allowed:
            raise typer.BadParameter("Cloud processing is disabled in configuration.")
        digest = hashlib.sha256()
        with file.open("rb") as audio_file:
            for block in iter(lambda: audio_file.read(1024 * 1024), b""):
                digest.update(block)
        content_hash = digest.hexdigest()
        if db.has_document_hash(content_hash):
            console.print("This audio file was already imported; no upload or duplicate transcript was made.")
            return
        if not typer.confirm(
            "Upload this audio file to your configured provider for transcription? The transcript will be stored locally; raw audio will not.",
            default=False,
        ):
            console.print("Audio transcription cancelled; no upload was made.")
            return
        try:
            provider = OpenAIProvider(config.llm.model, config.llm.request_timeout_seconds)
            transcript = asyncio.run(provider.transcribe_audio(file, model))
        except ProviderError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        if not transcript:
            raise typer.BadParameter("The provider returned an empty transcript; no source was stored.")
        document_id = db.import_document(f"{file.name} (transcript)", transcript, content_hash)
        if document_id is None:
            console.print("This audio file was already imported; no duplicate transcript was stored.")
            return
        added, linked, unchanged = _add_verbatim_candidates(db, document_id, transcript)
        console.print(
            f"Transcript stored as source {document_id}; raw audio was not retained. "
            f"{added} candidates pending review, {linked} existing candidates linked, "
            f"{unchanged} already-linked candidates unchanged."
        )
        console.print(f"For opt-in semantic extraction, run `psyche memory extract --document-id {document_id}`.")
    finally:
        db.close()


@memory_app.command("show")
def memory_show():
    _, db = services()
    for row in db.list_memories():
        sources = db.get_memory_sources(row[0])
        source_text = ", ".join(f"{source[1]}:{source[0]} {source[2] or ''}" for source in sources)
        console.print(f"{row[0]} · {row[2]} [{row[3]} / {row[6]}] ({source_text})")
    db.close()


@memory_app.command("extract")
def memory_extract(
    document_id: str | None = typer.Option(None, "--document-id"),
    session_id: str | None = typer.Option(None, "--session-id"),
):
    """Extract source-grounded pending candidates from one imported document or session."""
    if bool(document_id) == bool(session_id):
        raise typer.BadParameter("Specify exactly one of --document-id or --session-id.")
    try:
        config, db = services()
        if not config.privacy.cloud_processing_allowed:
            db.close()
            raise typer.BadParameter("Cloud processing is disabled in configuration.")
        provider = OpenAIProvider(config.llm.model, config.llm.request_timeout_seconds)
    except (EncryptionError, ProviderError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    extractor = MemoryExtractor(provider, db)
    try:
        if document_id:
            document = db.get_document(document_id)
            if document is None:
                raise typer.BadParameter("Document not found.")
            console.print("This sends the selected imported document to your configured LLM provider.")
            totals = asyncio.run(extractor.extract(document[2], document[0], "imported_document"))
        else:
            if not db.session_exists(session_id):
                raise typer.BadParameter("Session not found.")
            messages = db.get_user_messages(session_id)
            if not messages:
                console.print("No user messages to extract.")
                return
            console.print("This sends the selected session's user messages to your configured LLM provider.")
            async def extract_messages():
                totals = {"added": 0, "linked": 0, "unchanged": 0, "rejected": 0}
                for message_id, content in messages:
                    current = await extractor.extract(content, message_id, "session_message")
                    for key, value in current.items():
                        totals[key] += value
                return totals

            totals = asyncio.run(extract_messages())
        console.print(
            f"Extraction complete: {totals['added']} pending candidates added, "
            f"{totals['linked']} existing candidates linked to more evidence, "
            f"{totals['unchanged']} already-linked candidates unchanged, "
            f"{totals['rejected']} unsupported candidates discarded."
        )
        console.print("Review pending items with `psyche memory review`; none were auto-approved.")
    except ProviderError as exc:
        console.print(f"[red]{exc}[/red] Any earlier chunks may already have created pending candidates; review them before retrying.")
    finally:
        db.close()


@memory_app.command("search")
def memory_search(query: str):
    _, db = services()
    for row in db.search_memories(query):
        console.print(f"{row[0]} · {row[2]} [{row[3]}]")
    db.close()


@memory_app.command("review")
def memory_review():
    _, db = services()
    for row in db.list_memories("pending"):
        sources = db.get_memory_sources(row[0])
        console.print(f"{row[0]} · {row[2]} [{row[3]}] (evidence: {sources})")
        choice = typer.prompt("Approve, reject, or skip", default="skip").strip().casefold()
        if choice in {"approve", "reject"}:
            db.set_memory_status(row[0], "approved" if choice == "approve" else "rejected")
    db.close()


@memory_app.command("relate")
def memory_relate(
    memory_id: str,
    related_memory_id: str,
    relation_type: str = typer.Option("possible_contradiction", "--type"),
):
    """Flag two memories for human comparison; this does not change retrieval."""
    _, db = services()
    relation_id = db.propose_memory_relation(memory_id, related_memory_id, relation_type)
    db.close()
    if relation_id is None:
        raise typer.BadParameter("Could not relate these memories. Check their IDs, status, and relation type.")
    console.print(f"Pending relationship {relation_id}; both memories remain unchanged.")


@memory_app.command("review-relations")
def memory_review_relations():
    """Review user-flagged possible contradictions or related memories."""
    _, db = services()
    for relation in db.list_memory_relations():
        (relation_id, memory_id, statement, related_id, related_statement,
         relation_type, _, _) = relation
        console.print(f"{relation_id} · {relation_type}")
        console.print(f"  {memory_id}: {statement}")
        console.print(f"  {related_id}: {related_statement}")
        choice = typer.prompt("Accept this relationship, dismiss it, or skip", default="skip").strip().casefold()
        if choice in {"accept", "dismiss"}:
            db.review_memory_relation(relation_id, choice == "accept")
    db.close()


@memory_app.command("evolve")
def memory_evolve(new_memory_id: str, old_memory_id: str):
    """Confirm that a newer sourced memory replaces an older approved memory."""
    _, db = services()
    try:
        new = next((row for row in db.list_memories() if row[0] == new_memory_id), None)
        old = next((row for row in db.list_memories() if row[0] == old_memory_id), None)
        if not new or not old:
            raise typer.BadParameter("Both memory IDs must exist.")
        console.print(f"New: {new[2]}\nOlder: {old[2]}")
        if not typer.confirm("Approve the new memory and mark the older memory superseded?", default=False):
            console.print("No memory statuses changed.")
            return
        if not db.supersede_memory(new_memory_id, old_memory_id):
            raise typer.BadParameter(
                "Evolution requires sourced memories in the same category, a pending/approved new item, "
                "and an approved older item."
            )
    finally:
        db.close()
    console.print("New memory approved; older memory retained in history as superseded.")


@memory_app.command("profile")
def memory_profile():
    """Generate and display a versioned profile from approved atomic memories."""
    _, db = services()
    try:
        profile = db.generate_memory_profile()
        console.print(f"Memory profile v{profile['version']} · {len(profile['memory_ids'])} source memories")
        console.print(profile["content"])
    finally:
        db.close()


@memory_app.command("correct")
def memory_correct(memory_id: str, statement: str = typer.Argument(...)):
    _, db = services()
    ok = db.correct_memory(memory_id, statement)
    db.close()
    if not ok:
        raise typer.BadParameter("Memory not found.")
    console.print("Memory corrected and marked user-confirmed.")


@memory_app.command("forget")
def memory_forget(memory_id: str, delete_source: bool = typer.Option(False, "--delete-source")):
    _, db = services()
    row = next((r for r in db.list_memories() if r[0] == memory_id), None)
    if not row:
        db.close()
        raise typer.BadParameter("Memory not found.")
    db.set_memory_status(memory_id, "deleted")
    if delete_source:
        for source_id, source_type, _, _ in db.get_memory_sources(memory_id):
            if source_type in {"document", "imported_document", "session_message"}:
                db.delete_source(source_id, {source_type})
    db.close()
    notice = " Linked source content deleted." if delete_source else " Source history may retain this information; pass --delete-source to remove linked sources."
    console.print("Memory removed from active retrieval." + notice)


@app.command()
def export(file: Path = typer.Argument(...)):
    """Export local data to JSON (unencrypted; keep the destination private)."""
    _, db = services()
    sessions = db.list_sessions()
    payload = {"sessions": sessions, "transcripts": {s[0]: db.get_messages(s[0]) for s in sessions},
               "memories": db.list_memories(), "documents": db.documents()}
    db.close()
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    console.print(f"Unencrypted export created: {file}")


@app.command("delete-local-data")
def delete_local_data(confirm: bool = typer.Option(False, "--confirm")):
    """Delete local database and wrapped key (does not guarantee physical erasure)."""
    config, db = services()
    db.close()
    if not confirm:
        raise typer.BadParameter("Irreversible. Repeat with --confirm.")
    for name in ("psyche.db", "psyche.db-wal", "psyche.db-shm", "psyche.db-journal", "database.dpapi"):
        path = config.data_dir / name
        if path.is_file():
            path.unlink()
    try:
        config.data_dir.rmdir()
    except OSError:
        pass
    console.print("Local data files deleted. This does not erase SSD remnants or provider-side copies.")


if __name__ == "__main__":
    app()
