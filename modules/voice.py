"""Voice modality module — the first non-trivial modality.

Gate:    Silero VAD (is there speech?)
Decoder: placeholder for STT (Whisper — not yet integrated, returns raw transcript)
Encoder: Mod³ TTS engines (Kokoro, Voxtral, Chatterbox, Spark)

The encoder wraps engine.py and adaptive_player.py for local speaker output,
or returns raw audio bytes for channel delivery (Discord, HTTP).
"""

from __future__ import annotations

import io
import struct
import time

import numpy as np

from modality import (
    CognitiveEvent,
    CognitiveIntent,
    Decoder,
    EncodedOutput,
    Encoder,
    Gate,
    GateResult,
    ModalityModule,
    ModalityType,
    ModuleState,
    ModuleStatus,
)


# ---------------------------------------------------------------------------
# Gate: Silero VAD
# ---------------------------------------------------------------------------

class VoiceGate(Gate):
    """Voice activity detection gate using Silero VAD."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, raw: bytes, **kwargs) -> GateResult:
        from vad import detect_speech, is_hallucination

        sample_rate = kwargs.get("sample_rate", 16000)
        sample_width = kwargs.get("sample_width", 2)

        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(raw, dtype=np.float32)

        result = detect_speech(audio, sample_rate=sample_rate, threshold=self.threshold)

        return GateResult(
            passed=result.has_speech,
            confidence=result.confidence,
            reason=f"speech_ratio={result.speech_ratio} segments={result.num_segments}",
            metadata={
                "speech_ratio": result.speech_ratio,
                "num_segments": result.num_segments,
                "total_speech_sec": result.total_speech_sec,
                "total_audio_sec": result.total_audio_sec,
            },
        )


# ---------------------------------------------------------------------------
# Decoder: STT (placeholder — accepts pre-transcribed text for now)
# ---------------------------------------------------------------------------

class VoiceDecoder(Decoder):
    """Voice decoder. Currently accepts pre-transcribed text.
    Future: integrate Whisper/MLX STT directly."""

    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        # If raw is already text (pre-transcribed by external STT),
        # wrap it as a cognitive event
        transcript = kwargs.get("transcript")
        if transcript is None:
            transcript = raw.decode("utf-8", errors="replace")

        from vad import is_hallucination
        if is_hallucination(transcript):
            return CognitiveEvent(
                modality=ModalityType.VOICE,
                content="",
                confidence=0.0,
                metadata={"filtered": True, "reason": "hallucination", "original": transcript},
            )

        return CognitiveEvent(
            modality=ModalityType.VOICE,
            content=transcript,
            source_channel=kwargs.get("channel", ""),
            confidence=kwargs.get("confidence", 0.9),
        )


# ---------------------------------------------------------------------------
# Encoder: Mod³ TTS
# ---------------------------------------------------------------------------

def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 samples as 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HH", 1, 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<HH", 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class VoiceEncoder(Encoder):
    """TTS encoder using Mod³ engine (Kokoro, Voxtral, Chatterbox, Spark)."""

    def __init__(self, default_voice: str = "bm_lewis", default_speed: float = 1.25):
        self.default_voice = default_voice
        self.default_speed = default_speed
        self._state = ModuleState()

    @property
    def state(self) -> ModuleState:
        return self._state

    def encode(self, intent: CognitiveIntent) -> EncodedOutput:
        from engine import synthesize

        voice = intent.metadata.get("voice", self.default_voice)
        speed = intent.metadata.get("speed", self.default_speed)
        emotion = intent.metadata.get("emotion", 0.5)

        self._state.status = ModuleStatus.ENCODING
        self._state.current_text = intent.content[:100]
        self._state.last_activity = time.time()

        try:
            samples, sample_rate = synthesize(
                intent.content,
                voice=voice,
                speed=speed,
                emotion=emotion,
            )

            wav_bytes = _encode_wav(samples, sample_rate)
            duration = len(samples) / sample_rate

            self._state.status = ModuleStatus.IDLE
            self._state.last_output_text = intent.content[:100]
            self._state.progress = 1.0

            return EncodedOutput(
                modality=ModalityType.VOICE,
                data=wav_bytes,
                format="wav",
                duration_sec=duration,
                metadata={
                    "voice": voice,
                    "speed": speed,
                    "sample_rate": sample_rate,
                    "total_samples": len(samples),
                },
            )
        except Exception as e:
            self._state.status = ModuleStatus.ERROR
            self._state.error = str(e)
            raise


# ---------------------------------------------------------------------------
# Voice module
# ---------------------------------------------------------------------------

class VoiceModule(ModalityModule):
    """Voice modality — VAD gate, STT decoder, TTS encoder."""

    def __init__(
        self,
        default_voice: str = "bm_lewis",
        default_speed: float = 1.25,
        vad_threshold: float = 0.5,
    ):
        self._gate = VoiceGate(threshold=vad_threshold)
        self._decoder = VoiceDecoder()
        self._encoder = VoiceEncoder(default_voice=default_voice, default_speed=default_speed)

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.VOICE

    @property
    def gate(self) -> Gate:
        return self._gate

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def state(self) -> ModuleState:
        return self._encoder.state
