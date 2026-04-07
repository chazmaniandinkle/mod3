"""
Mod³ TTS Server — gives Claude a voice via multiple TTS engines on Apple Silicon.

Multi-model support: Voxtral, Kokoro, Chatterbox, Spark.
Voice presets are resolved to the correct engine automatically.

Interfaces:
  MCP (default):  stdio-based MCP tools for Claude Code
  HTTP (--http):  REST API for OpenClaw and external consumers
  Both (--all):   MCP on stdio + HTTP on a port, shared model cache

Tools (MCP):
  speak(text, voice, speed, emotion) — non-blocking speech, returns job ID
  speech_status(job_id)              — check job or get latest metrics
  stop()                             — interrupt current speech
  list_voices()                      — list available voice presets
  set_output_device(device)          — list/switch audio output
  diagnostics()                      — engine state + last metrics
"""

import json
import logging
import threading
import time
import uuid
import wave
from collections import OrderedDict
from typing import Any

import anyio
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from bus import ModalityBus
from modality import ModalityType, ModuleStatus
from modules.voice import PlaceholderDecoder, VoiceModule
from pipeline_state import InterruptInfo, PipelineState

logger = logging.getLogger("mod3.server")

_MODEL_REGISTRY = {
    "voxtral": {
        "id": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "voices": [
            "casual_male",
            "casual_female",
            "cheerful_female",
            "neutral_male",
            "neutral_female",
            "fr_male",
            "fr_female",
            "es_male",
            "es_female",
            "de_male",
            "de_female",
            "it_male",
            "it_female",
            "pt_male",
            "pt_female",
            "nl_male",
            "nl_female",
            "ar_male",
            "hi_male",
            "hi_female",
        ],
        "default_voice": "casual_male",
    },
    "kokoro": {
        "id": "mlx-community/Kokoro-82M-bf16",
        "voices": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "default_voice": "af_heart",
        "supports_speed": True,
    },
    "chatterbox": {
        "id": "mlx-community/chatterbox-4bit",
        "voices": ["chatterbox"],
        "default_voice": "chatterbox",
        "supports_exaggeration": True,
    },
    "spark": {
        "id": "mlx-community/Spark-TTS-0.5B-bf16",
        "voices": ["spark_male", "spark_female"],
        "default_voice": "spark_male",
        "supports_pitch": True,
        "supports_speed": True,
    },
}


def _create_bus() -> ModalityBus:
    bus = ModalityBus()
    bus.register(VoiceModule(decoder=PlaceholderDecoder()))
    return bus


_bus = _create_bus()
_bus_vad_lock = threading.Lock()


def _get_voice_module() -> VoiceModule | None:
    module = getattr(_bus, "_modules", {}).get(ModalityType.VOICE)
    return module if isinstance(module, VoiceModule) else None


def _engine_module():
    import engine

    return engine


def _try_engine_module():
    try:
        return _engine_module(), None
    except Exception as exc:
        return None, exc


def _model_registry() -> dict[str, dict[str, Any]]:
    engine_module, _ = _try_engine_module()
    return engine_module.MODELS if engine_module is not None else _MODEL_REGISTRY


def _adaptive_player_class():
    from adaptive_player import AdaptivePlayer

    return AdaptivePlayer


def _resolve_voice_via_bus(voice: str) -> tuple[str, str]:
    voice_module = _get_voice_module()
    if voice_module is None or voice_module.encoder is None:
        raise ValueError("Voice module is not registered on the ModalityBus.")

    for engine_name, cfg in _model_registry().items():
        if voice in cfg["voices"]:
            return engine_name, voice

    raise ValueError(f"Unknown voice '{voice}'. Use list_voices() to see options.")


def _read_wav_as_mono_float32(file_path: str) -> tuple[bytes, int]:
    with wave.open(file_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.float32).tobytes(), sample_rate


def _set_bus_voice_state(
    *,
    status: ModuleStatus,
    active_job: str | None = None,
    current_text: str = "",
    progress: float | None = None,
    last_output_text: str | None = None,
    error: str | None = None,
) -> None:
    voice_module = _get_voice_module()
    if voice_module is None:
        return

    state = voice_module.state
    state.status = status
    state.active_job = active_job
    state.current_text = current_text
    state.last_activity = time.time()
    state.error = error
    if progress is not None:
        state.progress = progress
    if last_output_text is not None:
        state.last_output_text = last_output_text

mcp = FastMCP(
    "mod3",
    instructions=(
        "Mod³ voice channel with multi-model TTS (Voxtral, Kokoro, Chatterbox, Spark) "
        "running locally on Apple Silicon. "
        'Voice messages arrive as <channel source="mod3" speaker="..." confidence="...">. '
        "Use the speak tool to respond via voice. speak() is non-blocking. "
        "Use speech_status to check completion. Use stop to interrupt. "
        "Keep spoken text conversational and concise — this is voice, not a document. "
        "For permission prompts, reply verbally with 'yes [code]' or 'no [code]'."
    ),
)

# ---------------------------------------------------------------------------
# Claude Code channel capabilities
# ---------------------------------------------------------------------------

_CHANNEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "claude/channel": {},
    "claude/channel/permission": {},
}

# Store the write stream for emitting channel notifications outside request context
_write_stream = None
_write_stream_lock = threading.Lock()


async def emit_channel_event(content: str, meta: dict[str, str] | None = None):
    """Emit a channel notification to Claude Code.

    Sends a ``notifications/claude/channel`` JSON-RPC notification over the
    active MCP session.  Can be called from tool handlers or background tasks
    while the server is running.

    Args:
        content: The textual content to relay (e.g. transcribed speech).
        meta: Optional string-keyed metadata dict (speaker, confidence, etc.).
    """
    with _write_stream_lock:
        ws = _write_stream
    if ws is None:
        raise RuntimeError("MCP session not active — cannot emit channel event")

    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel",
        params={"content": content, "meta": meta or {}},
    )
    await ws.send(SessionMessage(message=JSONRPCMessage(notification)))


async def emit_permission_verdict(request_id: str, behavior: str):
    """Emit a permission verdict notification to Claude Code.

    Sends a ``notifications/claude/channel/permission`` JSON-RPC notification
    with a verdict (allow/deny) for a previously received permission request.

    Args:
        request_id: The 5-letter request ID from the original permission request.
        behavior: ``"allow"`` or ``"deny"``.
    """
    with _write_stream_lock:
        ws = _write_stream
    if ws is None:
        raise RuntimeError("MCP session not active — cannot emit permission verdict")

    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel/permission",
        params={"request_id": request_id, "behavior": behavior},
    )
    await ws.send(SessionMessage(message=JSONRPCMessage(notification)))
    logger.info("permission verdict sent: request_id=%s behavior=%s", request_id, behavior)


def _handle_permission_request(params: dict[str, Any]) -> None:
    """Handle an incoming permission request by speaking it aloud via TTS.

    Called when the read-stream interceptor detects a
    ``notifications/claude/channel/permission_request`` notification from
    Claude Code.  Formats a spoken prompt and plays it through the default
    voice so the user can respond verbally.

    Args:
        params: The notification params containing ``request_id``,
                ``tool_name``, ``description``, and ``input_preview``.
    """
    request_id = params.get("request_id", "unknown")
    tool_name = params.get("tool_name", "a tool")
    description = params.get("description", "")

    # Build a concise spoken prompt
    prompt_parts = [f"Claude wants to run {tool_name}"]
    if description:
        prompt_parts.append(f": {description}")
    prompt_parts.append(f". Say yes {request_id} or no {request_id}.")
    prompt_text = "".join(prompt_parts)

    logger.info("permission request received: id=%s tool=%s", request_id, tool_name)
    _start_speech(prompt_text, voice="bm_lewis", speed=1.25)


# Patch run_stdio_async to inject experimental capabilities and capture the
# write stream so emit_channel_event can send notifications at any time.
_original_run_stdio = mcp.run_stdio_async

_PERMISSION_REQUEST_METHOD = "notifications/claude/channel/permission_request"


async def _patched_run_stdio():
    global _write_stream
    async with stdio_server() as (read_stream, write_stream):
        with _write_stream_lock:
            _write_stream = write_stream

        # Wrap the read stream to intercept permission_request notifications.
        # These use a custom method that the MCP session cannot parse into a
        # typed ClientNotification, so we handle them here and forward
        # everything else to the MCP server unchanged.
        send_inner, receive_inner = anyio.create_memory_object_stream[SessionMessage | Exception](0)

        async def _filter_read_stream():
            async with read_stream, send_inner:
                async for message in read_stream:
                    if isinstance(message, Exception):
                        await send_inner.send(message)
                        continue

                    root = message.message.root
                    if isinstance(root, JSONRPCNotification) and root.method == _PERMISSION_REQUEST_METHOD:
                        # Handle permission request — speak it via TTS
                        _handle_permission_request(root.params or {})
                        continue

                    # All other messages pass through to the MCP server
                    await send_inner.send(message)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_filter_read_stream)
                await mcp._mcp_server.run(
                    receive_inner,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(
                        experimental_capabilities=_CHANNEL_CAPABILITIES,
                    ),
                )
        finally:
            with _write_stream_lock:
                _write_stream = None


mcp.run_stdio_async = _patched_run_stdio

# ---------------------------------------------------------------------------
# Reflex arc — shared pipeline state
# ---------------------------------------------------------------------------

pipeline_state = PipelineState()


async def _emit_interruption(info: InterruptInfo):
    """Emit a channel notification when playback is interrupted.

    Called by the inbound pipeline (VAD reflex) after pipeline_state.interrupt()
    returns an InterruptInfo.  Notifies Claude Code that speech was cut short.
    """
    await emit_channel_event(
        content=f"[interrupted — speech halted at '{info.delivered_text}']",
        meta={
            "source": "mod3-voice",
            "type": "interruption",
            "spoken_pct": str(round(info.spoken_pct, 2)),
            "reason": info.reason,
        },
    )


# ---------------------------------------------------------------------------
# Job tracking (MCP only — local speaker playback)
# ---------------------------------------------------------------------------

MAX_JOBS = 20
_last_metrics: dict | None = None
_output_device: int | str | None = None
_jobs: OrderedDict[str, dict] = OrderedDict()
_current_player: Any | None = None
_current_player_lock = threading.Lock()


def _prune_jobs():
    """Keep only the last MAX_JOBS entries."""
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)


# ---------------------------------------------------------------------------
# Adaptive playback (MCP speaker output)
# ---------------------------------------------------------------------------


def _start_speech(
    text: str,
    voice: str,
    stream: bool = True,
    streaming_interval: float = 1.0,
    speed: float = 1.0,
    emotion: float = 0.5,
) -> str:
    """Start non-blocking speech generation. Returns job ID immediately."""
    global _last_metrics, _current_player
    engine_module = _engine_module()
    AdaptivePlayer = _adaptive_player_class()
    engine, voice = _resolve_voice_via_bus(voice)
    model = engine_module.get_model(engine)
    player = AdaptivePlayer(sample_rate=model.sample_rate, device=_output_device)

    with _current_player_lock:
        _current_player = player

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "speaking",
        "engine": engine,
        "voice": voice,
        "text": text[:100],
        "start_time": time.time(),
        "metrics": None,
        "error": None,
        "player": player,
    }
    _prune_jobs()
    _set_bus_voice_state(
        status=ModuleStatus.ENCODING,
        active_job=job_id,
        current_text=text[:100],
        progress=0.0,
        error=None,
    )

    def _run():
        # Register with the reflex arc so inbound VAD can interrupt us
        pipeline_state.start_speaking(text, player)
        try:
            for chunk in engine_module.generate_audio(
                text,
                voice=voice,
                stream=stream,
                streaming_interval=streaming_interval,
                speed=speed,
                emotion=emotion,
            ):
                player.queue_audio(chunk.samples, chunk_meta=chunk.metadata if chunk.metadata else None)
                _set_bus_voice_state(
                    status=ModuleStatus.ENCODING,
                    active_job=job_id,
                    current_text=text[:100],
                )
                # Update position after each chunk so PipelineState tracks progress
                pipeline_state.update_position(*player.get_progress())
        except Exception as e:
            _jobs[job_id]["error"] = str(e)
        finally:
            player.mark_done()

        metrics = player.wait(timeout=120.0)
        # Final position update and clear speaking state
        pipeline_state.update_position(*player.get_progress())
        pipeline_state.stop_speaking()

        result = metrics.to_dict()
        result["engine"] = engine
        result["voice"] = voice
        _jobs[job_id]["metrics"] = result
        _jobs[job_id]["status"] = "error" if _jobs[job_id]["error"] else "done"
        _last_metrics = result
        _set_bus_voice_state(
            status=ModuleStatus.ERROR if _jobs[job_id]["error"] else ModuleStatus.IDLE,
            active_job=None,
            current_text="",
            progress=1.0 if not _jobs[job_id]["error"] else 0.0,
            last_output_text=text[:100],
            error=_jobs[job_id]["error"],
        )

        with _current_player_lock:
            global _current_player
            if _current_player is player:
                _current_player = None

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def speak(
    text: str,
    voice: str = "bm_lewis",
    stream: bool = True,
    speed: float = 1.25,
    emotion: float = 0.5,
) -> str:
    """Synthesize text to speech and play it through the user's speakers.

    Non-blocking: returns immediately with a job ID while audio plays in the
    background. Use speech_status(id) to check completion and get metrics.

    If speech is already playing, the current output is interrupted and the
    new text starts immediately. The response includes an "interrupted" field
    with details about what was playing, so the agent is always aware of the
    output channel's state without needing a separate status check.

    Args:
        text: The text to speak aloud. Keep it conversational.
        voice: Voice preset. Use list_voices() to see options.
               Defaults to "bm_lewis" (Kokoro).
        stream: If True, plays audio chunks as they generate (lower latency).
                If False, generates all audio first then plays (better prosody).
        speed: Speed multiplier (engines with speed support). Default 1.25.
        emotion: Emotion/exaggeration intensity 0.0-1.0 (Chatterbox only). Default 0.5.
    """
    if not text.strip():
        return json.dumps({"status": "error", "error": "Nothing to say"})

    # Check if something is currently playing and include awareness in response.
    current_info = None
    with _current_player_lock:
        if _current_player is not None:
            # Find the active job
            for jid, jdata in reversed(_jobs.items()):
                if jdata["status"] == "speaking":
                    elapsed = round(time.time() - jdata["start_time"], 1)
                    current_info = {
                        "job_id": jid,
                        "elapsed_sec": elapsed,
                        "text_preview": jdata.get("text", "")[:80],
                    }
                    break

    try:
        # If something is playing, interrupt it (default: barge-in on self).
        # This prevents double-output on the same device.
        if current_info is not None:
            with _current_player_lock:
                if _current_player is not None:
                    _current_player.flush()

        job_id = _start_speech(text, voice, stream=stream, speed=speed, emotion=emotion)
        result = {"status": "speaking", "job_id": job_id}
        if current_info is not None:
            result["interrupted"] = current_info
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def speech_status(job_id: str = "", verbose: bool = False) -> str:
    """Check status of a speech job, or get the most recent result.

    Args:
        job_id: The job ID returned by speak(). If empty, returns the latest job.
        verbose: If True, include per-chunk metrics. Default False (summary only).
    """
    if not job_id:
        if not _jobs:
            return json.dumps({"status": "idle", "message": "No speech jobs"})
        job_id = next(reversed(_jobs))

    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})

    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "speaking":
        result["elapsed_sec"] = round(time.time() - job["start_time"], 1)
    if job["metrics"]:
        metrics = job["metrics"]
        if not verbose and "chunks" in metrics:
            chunks = metrics["chunks"]["per_chunk"]
            rtfs = [c["rtf"] for c in chunks if c.get("rtf")]
            metrics = {
                **metrics,
                "chunks": {
                    "count": metrics["chunks"]["count"],
                    "avg_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else 0,
                    "min_rtf": round(min(rtfs), 2) if rtfs else 0,
                },
            }
        result["metrics"] = metrics
    if job["error"]:
        result["error"] = job["error"]
    return json.dumps(result)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def stop() -> str:
    """Stop current speech playback immediately."""
    with _current_player_lock:
        player = _current_player
    if player is None:
        return json.dumps({"status": "ok", "message": "Nothing playing"})

    player.flush()
    return json.dumps({"status": "ok", "message": "Speech interrupted"})


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vad_check(file_path: str, threshold: float = 0.5) -> str:
    """Check if an audio file contains speech using Silero VAD.

    Use this before transcription to avoid Whisper hallucinations on
    silence or ambient noise.

    Args:
        file_path: Path to a WAV audio file.
        threshold: Speech probability threshold 0-1 (default 0.5). Higher = stricter.
    """
    try:
        voice_module = _get_voice_module()
        if voice_module is None or voice_module.gate is None:
            from vad import detect_speech_file

            result = detect_speech_file(file_path, threshold=threshold)
            return json.dumps(
                {
                    "has_speech": result.has_speech,
                    "confidence": result.confidence,
                    "speech_ratio": result.speech_ratio,
                    "num_segments": result.num_segments,
                    "total_speech_sec": result.total_speech_sec,
                    "total_audio_sec": result.total_audio_sec,
                }
            )

        raw_audio, sample_rate = _read_wav_as_mono_float32(file_path)
        with _bus_vad_lock:
            previous_threshold = getattr(voice_module.gate, "threshold", threshold)
            voice_module.gate.threshold = threshold
            gate_result = voice_module.gate.check(raw_audio, sample_rate=sample_rate, sample_width=4)
            _bus.perceive(
                raw_audio,
                modality=ModalityType.VOICE,
                channel="mcp:vad_check",
                sample_rate=sample_rate,
                sample_width=4,
                transcript="speech detected",
            )
            voice_module.gate.threshold = previous_threshold

        return json.dumps(
            {
                "has_speech": gate_result.passed,
                "confidence": gate_result.confidence,
                "speech_ratio": gate_result.metadata.get("speech_ratio", 0.0),
                "num_segments": gate_result.metadata.get("num_segments", 0),
                "total_speech_sec": gate_result.metadata.get("total_speech_sec", 0.0),
                "total_audio_sec": gate_result.metadata.get("total_audio_sec", 0.0),
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def list_voices() -> str:
    """List all available voice presets grouped by engine."""
    if _get_voice_module() is None:
        logger.warning("list_voices called without a registered bus voice module")

    models = _model_registry()
    lines = []
    for engine, cfg in models.items():
        extras = []
        if cfg.get("supports_speed"):
            extras.append("speed")
        if cfg.get("supports_exaggeration"):
            extras.append("emotion")
        if cfg.get("supports_pitch"):
            extras.append("pitch")
        tag = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"  {engine}{tag}: {', '.join(cfg['voices'])}")
    return "Available voices:\n" + "\n".join(lines)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def diagnostics() -> str:
    """Return engine state and last generation metrics for debugging."""
    engine_module, engine_error = _try_engine_module()
    models = engine_module.MODELS if engine_module is not None else _MODEL_REGISTRY
    loaded_engines = engine_module.get_loaded_engines() if engine_module is not None else []
    engines = {}
    for name, cfg in models.items():
        engines[name] = {
            "loaded": name in loaded_engines,
            "model_id": cfg["id"],
            "voices": len(cfg["voices"]),
        }
    info = {
        "engines": engines,
        "bus": {
            "health": _bus.health(),
            "hud": _bus.hud(),
        },
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "speaking"),
        "total_jobs": len(_jobs),
        "output_device": _output_device,
        "last_metrics": _last_metrics,
    }
    if engine_error is not None:
        info["engine_import_error"] = str(engine_error)
    return json.dumps(info, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def set_output_device(device: str = "") -> str:
    """List audio output devices, or set the active one.

    Args:
        device: Device index (e.g. "3") or name substring (e.g. "AirPods").
                If empty, lists available devices without changing anything.
    """
    import sounddevice as sd

    global _output_device

    outputs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            is_default = i == sd.default.device[1]
            is_active = (
                (_output_device is None and is_default)
                or _output_device == i
                or (isinstance(_output_device, str) and _output_device in d["name"])
            )
            outputs.append({"index": i, "name": d["name"], "active": is_active})

    if not device:
        lines = [f"  [{'*' if d['active'] else ' '}] {d['index']}: {d['name']}" for d in outputs]
        return "Audio output devices (* = active):\n" + "\n".join(lines)

    if device.isdigit():
        _output_device = int(device)
    else:
        _output_device = device

    return json.dumps({"status": "ok", "device": _output_device})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_http(host: str = "0.0.0.0", port: int = 7860):
    """Start the HTTP API server."""
    import uvicorn

    from http_api import app

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mod³ TTS Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP API only")
    parser.add_argument("--all", action="store_true", help="Run both MCP (stdio) and HTTP")
    parser.add_argument("--channel", action="store_true", help="Run as channel server with voice input")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address")
    args = parser.parse_args()

    if args.http:
        _run_http(host=args.host, port=args.port)
    elif args.all:
        # HTTP in background thread, MCP on stdio
        http_thread = threading.Thread(
            target=_run_http,
            kwargs={"host": args.host, "port": args.port},
            daemon=True,
        )
        http_thread.start()
        mcp.run()
    elif args.channel:
        # Channel mode: MCP on stdio + inbound voice pipeline
        from bus import ModalityBus
        from inbound import InboundPipeline
        from modules.voice import VoiceModule

        bus = ModalityBus()
        bus.register(VoiceModule())
        inbound = InboundPipeline(bus=bus, pipeline_state=pipeline_state)
        inbound.start()
        try:
            mcp.run()  # MCP on stdio with channel capabilities
        finally:
            inbound.stop()
    else:
        mcp.run()
