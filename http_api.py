"""Mod³ HTTP API — REST interface for TTS synthesis and VAD.

Endpoints:
  POST /v1/synthesize  — text → audio bytes (WAV/PCM) + structured metrics
  POST /v1/audio/speech — OpenAI-compatible TTS endpoint
  POST /v1/vad         — audio file → speech detection result
  POST /v1/filter      — text → hallucination check
  GET  /v1/voices      — list available engines and voices
  GET  /v1/jobs        — list recent generation jobs with full metrics
  GET  /v1/jobs/{id}   — get a specific job's metrics
  GET  /health         — server health check
"""

import io
import struct
import time
import uuid
import wave
from collections import OrderedDict
from threading import Lock

from fastapi import FastAPI, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bus import ModalityBus
from engine import MODELS, generate_audio, get_loaded_engines
from modality import EncodedOutput, ModalityType
from modules.text import TextModule
from modules.voice import VoiceModule
from vad import detect_speech_file, is_hallucination
from vad import is_model_loaded as vad_loaded

app = FastAPI(title="Mod³", description="Local multi-model TTS on Apple Silicon")

try:
    from server import _bus as _shared_bus
except Exception:
    _shared_bus = ModalityBus()

_bus = _shared_bus
_bus_vad_lock = Lock()


def _ensure_bus_modules() -> None:
    modules = getattr(_bus, "_modules", {})
    if ModalityType.TEXT not in modules:
        _bus.register(TextModule())
    if ModalityType.VOICE not in modules:
        _bus.register(VoiceModule())


def _get_voice_module() -> VoiceModule | None:
    module = getattr(_bus, "_modules", {}).get(ModalityType.VOICE)
    return module if isinstance(module, VoiceModule) else None


def _resolve_voice_via_bus(voice: str) -> str:
    voice_module = _get_voice_module()
    if voice_module is None or voice_module.encoder is None:
        raise ValueError("Voice module is not registered on the ModalityBus.")

    for cfg in MODELS.values():
        if voice in cfg["voices"]:
            return voice

    raise ValueError(f"Unknown voice '{voice}'. Use /v1/voices to see options.")


def _read_wav_as_mono_float32(raw_wav: bytes) -> tuple[bytes, int]:
    import numpy as np

    with wave.open(io.BytesIO(raw_wav), "rb") as wav_file:
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


_ensure_bus_modules()

# ---------------------------------------------------------------------------
# Job ledger — full lifecycle tracking for every generation
# ---------------------------------------------------------------------------

MAX_JOBS = 100
_jobs: OrderedDict[str, dict] = OrderedDict()
_jobs_lock = Lock()


def _record_job(job: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    job["job_id"] = job_id
    with _jobs_lock:
        _jobs[job_id] = job
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)
    return job_id


def _update_job(job_id: str, updates: dict):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def encode_wav(samples, sample_rate: int) -> bytes:
    """Encode float32 samples as 16-bit PCM WAV."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(pcm)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    # WAV header (44 bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", 1))  # mono
    buf.write(struct.pack("<I", sample_rate))  # sample rate
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = Field(default="bm_lewis")
    speed: float = Field(default=1.25)
    emotion: float = Field(default=0.5)
    format: str = Field(default="wav", pattern="^(wav|pcm)$")


class SpeechRequest(BaseModel):
    """OpenAI-compatible TTS request."""

    model: str = Field(default="kokoro")
    input: str
    voice: str = Field(default="af_heart")
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/synthesize")
def synthesize(req: SynthesizeRequest):
    """Synthesize text to audio. Returns raw audio bytes + full metrics in headers and job ledger."""
    import numpy as np

    t_request = time.perf_counter()
    job_id = _record_job(
        {
            "type": "synthesize",
            "status": "generating",
            "requested_at": time.time(),
            "text": req.text[:200],
            "voice": req.voice,
            "speed": req.speed,
            "emotion": req.emotion,
            "format": req.format,
            "engine": None,
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    try:
        req.voice = _resolve_voice_via_bus(req.voice)
    except ValueError as e:
        _update_job(job_id, {"status": "error", "error": str(e)})
        return JSONResponse(status_code=400, content={"error": str(e), "job_id": job_id})

    t_gen_start = time.perf_counter()
    _update_job(job_id, {"timeline_append": True})
    _append_timeline(job_id, "generation_start", t_gen_start - t_request)

    chunks = list(
        generate_audio(
            req.text,
            voice=req.voice,
            speed=req.speed,
            emotion=req.emotion,
            stream=False,
        )
    )
    t_gen_end = time.perf_counter()

    if not chunks:
        _update_job(job_id, {"status": "error", "error": "No audio generated"})
        return JSONResponse(status_code=400, content={"error": "No audio generated", "job_id": job_id})

    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    duration = len(all_samples) / sample_rate
    gen_time = t_gen_end - t_gen_start

    # Per-chunk metrics
    chunk_metrics = []
    for c in chunks:
        if c.metadata:
            chunk_metrics.append(c.metadata)

    t_encode_start = time.perf_counter()
    if req.format == "pcm":
        pcm = (np.clip(all_samples, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        media_type = "audio/pcm"
    else:
        audio_bytes = encode_wav(all_samples, sample_rate)
        media_type = "audio/wav"
    t_encode_end = time.perf_counter()

    total_time = t_encode_end - t_request
    engine = chunks[0].metadata.get("engine", "") if chunks[0].metadata else ""

    # Finalize job record
    _append_timeline(job_id, "generation_complete", t_gen_end - t_request)
    _append_timeline(job_id, "encoding_complete", t_encode_end - t_request)
    _update_job(
        job_id,
        {
            "status": "complete",
            "engine": engine,
            "metrics": {
                "audio_duration_sec": round(duration, 3),
                "total_samples": len(all_samples),
                "sample_rate": sample_rate,
                "generation_time_sec": round(gen_time, 3),
                "encoding_time_sec": round(t_encode_end - t_encode_start, 4),
                "total_time_sec": round(total_time, 3),
                "rtf": round(duration / gen_time, 2) if gen_time > 0 else 0,
                "chunks": len(chunk_metrics),
                "per_chunk": chunk_metrics,
                "output_bytes": len(audio_bytes),
                "output_format": req.format,
            },
        },
    )

    headers = {
        "X-Mod3-Job-Id": job_id,
        "X-Mod3-Engine": engine,
        "X-Mod3-Voice": req.voice,
        "X-Mod3-Duration-Sec": f"{duration:.3f}",
        "X-Mod3-Sample-Rate": str(sample_rate),
        "X-Mod3-Gen-Time-Sec": f"{gen_time:.3f}",
        "X-Mod3-Total-Time-Sec": f"{total_time:.3f}",
        "X-Mod3-RTF": f"{duration / gen_time:.2f}" if gen_time > 0 else "0",
        "X-Mod3-Chunks": str(len(chunk_metrics)),
    }

    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.post("/v1/audio/speech")
def audio_speech(req: SpeechRequest):
    """OpenAI-compatible TTS endpoint. Accepts OpenAI format, returns WAV audio."""
    import numpy as np

    t_request = time.perf_counter()

    voice = req.voice
    try:
        voice = _resolve_voice_via_bus(voice)
    except ValueError:
        voice = "af_heart"

    job_id = _record_job(
        {
            "type": "audio_speech",
            "status": "generating",
            "requested_at": time.time(),
            "text": req.input[:200],
            "voice": voice,
            "speed": req.speed,
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    chunks = list(
        generate_audio(
            req.input,
            voice=voice,
            speed=req.speed,
            stream=False,
        )
    )
    t_gen_end = time.perf_counter()

    if not chunks:
        _update_job(job_id, {"status": "error", "error": "No audio generated"})
        return JSONResponse(status_code=500, content={"error": "No audio generated", "job_id": job_id})

    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    duration = len(all_samples) / sample_rate
    gen_time = t_gen_end - t_request

    audio_bytes = encode_wav(all_samples, sample_rate)
    total_time = time.perf_counter() - t_request
    engine = chunks[0].metadata.get("engine", "") if chunks[0].metadata else ""

    _update_job(
        job_id,
        {
            "status": "complete",
            "engine": engine,
            "metrics": {
                "audio_duration_sec": round(duration, 3),
                "generation_time_sec": round(gen_time, 3),
                "total_time_sec": round(total_time, 3),
                "rtf": round(duration / gen_time, 2) if gen_time > 0 else 0,
            },
        },
    )

    headers = {
        "X-Mod3-Job-Id": job_id,
        "X-Mod3-Engine": engine,
        "X-Mod3-Voice": voice,
        "X-Mod3-Duration-Sec": f"{duration:.3f}",
        "X-Mod3-Sample-Rate": str(sample_rate),
        "X-Mod3-Gen-Time-Sec": f"{gen_time:.3f}",
        "X-Mod3-Total-Time-Sec": f"{total_time:.3f}",
    }

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/vad")
async def vad_check(file: UploadFile):
    """Check if an audio file contains speech. Returns VAD result with timing."""
    import tempfile

    t_start = time.perf_counter()

    job_id = _record_job(
        {
            "type": "vad",
            "status": "processing",
            "requested_at": time.time(),
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    content = await file.read()
    t_load = time.perf_counter()

    voice_module = _get_voice_module()
    if voice_module is not None and voice_module.gate is not None:
        raw_audio, sample_rate = _read_wav_as_mono_float32(content)
        with _bus_vad_lock:
            gate_result = voice_module.gate.check(raw_audio, sample_rate=sample_rate, sample_width=4)
            _bus.perceive(
                raw_audio,
                modality=ModalityType.VOICE,
                channel="http:v1/vad",
                sample_rate=sample_rate,
                sample_width=4,
                transcript="speech detected",
            )

        class _Result:
            has_speech = gate_result.passed
            confidence = gate_result.confidence
            speech_ratio = gate_result.metadata.get("speech_ratio", 0.0)
            num_segments = gate_result.metadata.get("num_segments", 0)
            total_speech_sec = gate_result.metadata.get("total_speech_sec", 0.0)
            total_audio_sec = gate_result.metadata.get("total_audio_sec", 0.0)

        result = _Result()
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            result = detect_speech_file(tmp.name)

    t_end = time.perf_counter()
    processing_time = t_end - t_start

    _update_job(
        job_id,
        {
            "status": "complete",
            "metrics": {
                "has_speech": result.has_speech,
                "confidence": result.confidence,
                "speech_ratio": result.speech_ratio,
                "num_segments": result.num_segments,
                "total_speech_sec": result.total_speech_sec,
                "total_audio_sec": result.total_audio_sec,
                "processing_time_sec": round(processing_time, 4),
                "file_load_time_sec": round(t_load - t_start, 4),
                "vad_time_sec": round(t_end - t_load, 4),
            },
        },
    )

    return {
        "job_id": job_id,
        "has_speech": result.has_speech,
        "confidence": result.confidence,
        "speech_ratio": result.speech_ratio,
        "num_segments": result.num_segments,
        "total_speech_sec": result.total_speech_sec,
        "total_audio_sec": result.total_audio_sec,
        "processing_time_sec": round(processing_time, 4),
    }


@app.post("/v1/filter")
async def filter_transcription(req: dict):
    """Check if a transcription is a known Whisper hallucination.

    Body: {"text": "thank you"}
    Returns: {"is_hallucination": true, "text": "thank you"}
    """
    text = req.get("text", "")
    return {
        "is_hallucination": is_hallucination(text),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Job introspection
# ---------------------------------------------------------------------------


@app.get("/v1/jobs")
def list_jobs(limit: int = 20, type: str = ""):
    """List recent generation jobs with metrics. Optionally filter by type."""
    with _jobs_lock:
        jobs = list(reversed(_jobs.values()))
    if type:
        jobs = [j for j in jobs if j.get("type") == type]
    return {"jobs": jobs[:limit], "total": len(jobs)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """Get full details for a specific job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    return job


# ---------------------------------------------------------------------------
# Voices and health
# ---------------------------------------------------------------------------


@app.get("/v1/voices")
def voices():
    """List available engines and their voices."""
    engines = {}
    for name, cfg in MODELS.items():
        supports = []
        if cfg.get("supports_speed"):
            supports.append("speed")
        if cfg.get("supports_exaggeration"):
            supports.append("emotion")
        if cfg.get("supports_pitch"):
            supports.append("pitch")
        engines[name] = {
            "model_id": cfg["id"],
            "voices": cfg["voices"],
            "default_voice": cfg["default_voice"],
            "supports": supports,
        }
    return {"engines": engines}


@app.get("/health")
def health():
    """Health check with summary stats."""
    with _jobs_lock:
        total = len(_jobs)
        active = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))
        by_type = {}
        for j in _jobs.values():
            t = j.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
    return {
        "status": "ok",
        "engines_loaded": get_loaded_engines(),
        "vad_loaded": vad_loaded(),
        "jobs": {
            "total": total,
            "active": active,
            "by_type": by_type,
        },
    }


@app.get("/diagnostics")
def diagnostics():
    """Diagnostics snapshot with bus state."""
    with _jobs_lock:
        total = len(_jobs)
        active = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))
    return {
        "engines_loaded": get_loaded_engines(),
        "vad_loaded": vad_loaded(),
        "jobs": {
            "total": total,
            "active": active,
        },
        "bus": {
            "health": _bus.health(),
            "hud": _bus.hud(),
        },
    }


# ---------------------------------------------------------------------------
# Modality Bus endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/bus/hud")
def bus_hud():
    """Agent HUD — live state of all modalities, channels, and queues."""
    return _bus.hud()


@app.get("/v1/bus/health")
def bus_health():
    """Full modality bus health report."""
    return _bus.health()


@app.post("/v1/bus/perceive")
async def bus_perceive(file: UploadFile, modality: str = "voice", channel: str = ""):
    """Run raw input through the modality bus: gate → decode → cognitive event."""
    raw = await file.read()
    event = _bus.perceive(raw, modality=modality, channel=channel)
    if event is None:
        return {"status": "filtered", "modality": modality, "channel": channel}
    return {
        "status": "ok",
        "event": {
            "modality": event.modality.value,
            "content": event.content,
            "confidence": event.confidence,
            "source_channel": event.source_channel,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
        },
    }


@app.post("/v1/bus/act")
def bus_act(req: dict):
    """Route a cognitive intent through the bus: resolve modality → encode → queue.

    Body: {"content": "hello world", "modality": "voice", "channel": "discord-voice",
           "voice": "bm_lewis", "speed": 1.25}
    """
    from modality import CognitiveIntent, ModalityType

    content = req.get("content", "")
    modality = req.get("modality")
    channel = req.get("channel", "")
    metadata = {}
    for k in ("voice", "speed", "emotion"):
        if k in req:
            metadata[k] = req[k]

    intent = CognitiveIntent(
        modality=ModalityType(modality) if modality else None,
        content=content,
        target_channel=channel,
        metadata=metadata,
    )

    output = _bus.act(intent, channel=channel, blocking=True)
    assert isinstance(output, EncodedOutput), "Expected blocking act() to return EncodedOutput"

    return {
        "status": "ok",
        "modality": output.modality.value,
        "format": output.format,
        "duration_sec": output.duration_sec,
        "bytes": len(output.data),
        "metadata": output.metadata,
    }


def get_bus() -> ModalityBus:
    """Get the global bus instance (for server.py integration)."""
    return _bus


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_timeline(job_id: str, event: str, t: float):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job and "timeline" in job:
            job["timeline"].append({"event": event, "t": round(t, 4)})
