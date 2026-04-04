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
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from mcp.server.fastmcp import FastMCP

from adaptive_player import AdaptivePlayer
from engine import MODELS, generate_audio, resolve_model, get_model, get_loaded_engines

mcp = FastMCP(
    "mod3",
    instructions=(
        "Mod³ TTS server with multi-model support (Voxtral, Kokoro, Chatterbox, Spark) "
        "running locally on Apple Silicon. "
        "Use the `speak` tool to say something out loud through the user's speakers. "
        "speak() is non-blocking — it returns immediately while audio plays in the background. "
        "Use `speech_status` to check completion and get metrics. "
        "Use `stop` to interrupt current speech. "
        "Keep spoken text conversational and concise — this is voice, not a document."
    ),
)

# ---------------------------------------------------------------------------
# Job tracking (MCP only — local speaker playback)
# ---------------------------------------------------------------------------

MAX_JOBS = 20
_last_metrics: dict | None = None
_output_device: int | str | None = None
_jobs: OrderedDict[str, dict] = OrderedDict()
_current_player: AdaptivePlayer | None = None
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
    engine, voice = resolve_model(voice)
    model = get_model(engine)
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

    def _run():
        try:
            for chunk in generate_audio(
                text, voice=voice, stream=stream,
                streaming_interval=streaming_interval,
                speed=speed, emotion=emotion,
            ):
                player.queue_audio(chunk.samples, chunk_meta=chunk.metadata if chunk.metadata else None)
        except Exception as e:
            _jobs[job_id]["error"] = str(e)
        finally:
            player.mark_done()

        metrics = player.wait(timeout=120.0)
        result = metrics.to_dict()
        result["engine"] = engine
        result["voice"] = voice
        _jobs[job_id]["metrics"] = result
        _jobs[job_id]["status"] = "error" if _jobs[job_id]["error"] else "done"
        _last_metrics = result

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

    try:
        job_id = _start_speech(text, voice, stream=stream, speed=speed, emotion=emotion)
        return json.dumps({"status": "speaking", "job_id": job_id})
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
            metrics = {**metrics, "chunks": {
                "count": metrics["chunks"]["count"],
                "avg_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else 0,
                "min_rtf": round(min(rtfs), 2) if rtfs else 0,
            }}
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
    from vad import detect_speech_file, is_model_loaded
    try:
        result = detect_speech_file(file_path, threshold=threshold)
        return json.dumps({
            "has_speech": result.has_speech,
            "confidence": result.confidence,
            "speech_ratio": result.speech_ratio,
            "num_segments": result.num_segments,
            "total_speech_sec": result.total_speech_sec,
            "total_audio_sec": result.total_audio_sec,
        })
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
    lines = []
    for engine, cfg in MODELS.items():
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
    engines = {}
    for name, cfg in MODELS.items():
        engines[name] = {
            "loaded": name in get_loaded_engines() or False,
            "model_id": cfg["id"],
            "voices": len(cfg["voices"]),
        }
    info = {
        "engines": engines,
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "speaking"),
        "total_jobs": len(_jobs),
        "output_device": _output_device,
        "last_metrics": _last_metrics,
    }
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
    else:
        mcp.run()
