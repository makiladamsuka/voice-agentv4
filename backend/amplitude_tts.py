"""
amplitude_tts.py
================
Wraps the Deepgram TTS plugin to intercept raw PCM bytes as they stream
out of the synthesizer, compute two parallel smoothed RMS amplitude signals,
and forward them over UDP to the robot_eyes.py process.

UDP payload sent every ~40ms:
    {"amplitude_fast": float, "amplitude_slow": float}

Fast signal (α=0.6):  micro-reactions – syllable punches, lid snaps
Slow signal (α=0.05): emotional momentum – vertical float, saccade rate
"""

from __future__ import annotations

import asyncio
import math
import socket
import json
import struct
import time
from dataclasses import replace
from typing import Generator

from livekit.plugins.deepgram.tts import (
    TTS as DeepgramTTS,
    ChunkedStream as DeepgramChunkedStream,
    SynthesizeStream as DeepgramSynthesizeStream,
)
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


# ──────────────────────────────────────────────────────────────────────────────
# Shared amplitude state (updated by the TTS wrapper, read anywhere)
# ──────────────────────────────────────────────────────────────────────────────

_ampl_fast: float = 0.0   # α = 0.6  – tracks individual syllable peaks
_ampl_slow: float = 0.0   # α = 0.05 – tracks overall speech energy envelope

_ALPHA_FAST = 0.6
_ALPHA_SLOW = 0.05
_SEND_INTERVAL = 0.040    # 40 ms between UDP sends

# ── Pacing Logic ─────────────────────────────────────────────────────────────
_audio_buffer = bytearray()
_pacer_task: asyncio.Task | None = None
_SAMPLE_RATE = 24000  # Default for Deepgram Aura
_BYTES_PER_SAMPLE = 2 # 16-bit PCM
_CHUNK_MS = 40
_CHUNK_BYTES = int((_SAMPLE_RATE * _BYTES_PER_SAMPLE) * (_CHUNK_MS / 1000))

# ──────────────────────────────────────────────────────────────────────────────
# UDP sender – fire-and-forget, silently drops on any exception
# ──────────────────────────────────────────────────────────────────────────────

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_EYES_ADDR = ("127.0.0.1", 9000)
_last_sent = 0.0

def _rms(pcm_bytes: bytes) -> float:
    """Compute RMS of 16-bit PCM bytes in [0.0, 1.0]."""
    if len(pcm_bytes) < 2:
        return 0.0
    n = len(pcm_bytes) // 2
    # struct.unpack is faster than numpy on short chunks
    samples = struct.unpack(f"<{n}h", pcm_bytes[:n * 2])
    rms = math.sqrt(sum(s * s for s in samples) / n)
    
    # Normalise INT16 -> [0, 1] then apply a x4 gain boost
    # This ensures even quiet speech drives the physical eyes
    normalised = min(rms / 32768.0, 1.0)
    boosted = min(normalised * 4.0, 1.0)
    return boosted


def _process_chunk(pcm_bytes: bytes) -> None:
    """Buffer raw PCM bytes for the pacer loop to consume at 1x real-time."""
    global _audio_buffer
    _audio_buffer.extend(pcm_bytes)
    _ensure_pacer()

async def _pacer_loop() -> None:
    """Consumes the audio buffer at real-time speed and sends UDP updates."""
    global _ampl_fast, _ampl_slow, _audio_buffer
    
    while True:
        start_time = asyncio.get_event_loop().time()
        
        # Pull 40ms of audio from the buffer
        chunk = b""
        if len(_audio_buffer) >= _CHUNK_BYTES:
            chunk = bytes(_audio_buffer[:_CHUNK_BYTES])
            del _audio_buffer[:_CHUNK_BYTES]
        
        # Compute RMS or decay toward zero if buffer is empty
        if chunk:
            raw = _rms(chunk)
        else:
            # Decay signals slightly when no audio is playing to prevent hard snaps
            raw = 0.0
            
        _ampl_fast = _ALPHA_FAST * raw + (1.0 - _ALPHA_FAST) * _ampl_fast
        _ampl_slow = _ALPHA_SLOW * raw + (1.0 - _ALPHA_SLOW) * _ampl_slow

        # Always send UDP to keep eyes alive/synced
        try:
            payload = json.dumps({
                "amplitude_fast": round(_ampl_fast, 4),
                "amplitude_slow": round(_ampl_slow, 4),
            }).encode("utf-8")
            _sock.sendto(payload, _EYES_ADDR)
        except Exception:
            pass

        # Maintain precise 40ms cadence regardless of processing time
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, (_CHUNK_MS / 1000.0) - elapsed))

def _ensure_pacer() -> None:
    """Starts the pacer loop task if it's not already running."""
    global _pacer_task
    if _pacer_task is None or _pacer_task.done():
        try:
            loop = asyncio.get_running_loop()
            _pacer_task = loop.create_task(_pacer_loop())
        except RuntimeError:
            pass # No loop running yet


def _drain_to_zero() -> None:
    """Called when speech ends – clear buffer and decay signals."""
    global _ampl_fast, _ampl_slow, _audio_buffer
    _audio_buffer.clear()
    _ampl_fast = 0.0
    _ampl_slow = 0.0
    try:
        payload = json.dumps({
            "amplitude_fast": 0.0,
            "amplitude_slow": 0.0,
        }).encode("utf-8")
        _sock.sendto(payload, _EYES_ADDR)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Proxy AudioEmitter – wraps the real AudioEmitter and taps push()
# ──────────────────────────────────────────────────────────────────────────────

class _TappingEmitter:
    """Thin proxy around tts.AudioEmitter that intercepts raw byte pushes."""

    def __init__(self, real_emitter: tts.AudioEmitter):
        self._real = real_emitter

    # Proxy everything straight through, only tap push(bytes)
    def initialize(self, **kwargs):    return self._real.initialize(**kwargs)
    def start_segment(self, **kwargs): return self._real.start_segment(**kwargs)
    def end_segment(self):             return self._real.end_segment()
    def flush(self):                   return self._real.flush()
    def end_input(self):               return self._real.end_input()
    async def join(self):              return await self._real.join()
    async def aclose(self):            return await self._real.aclose()

    def push(self, data: bytes) -> None:
        """Intercept raw PCM bytes, compute RMS, then forward to real emitter."""
        _process_chunk(data)
        return self._real.push(data)

    def push_timed_transcript(self, delta_text):
        return self._real.push_timed_transcript(delta_text)

    def pushed_duration(self, idx=-1):
        return self._real.pushed_duration(idx=idx)

    @property
    def num_segments(self): return self._real.num_segments


# ──────────────────────────────────────────────────────────────────────────────
# Subclassed ChunkedStream – wraps output_emitter before passing to _run
# ──────────────────────────────────────────────────────────────────────────────

class _TappingChunkedStream(DeepgramChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        await super()._run(_TappingEmitter(output_emitter))


# ──────────────────────────────────────────────────────────────────────────────
# Subclassed SynthesizeStream
# ──────────────────────────────────────────────────────────────────────────────

class _TappingSynthesizeStream(DeepgramSynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        await super()._run(_TappingEmitter(output_emitter))


# ──────────────────────────────────────────────────────────────────────────────
# The public TTS class - drop-in replacement for deepgram.TTS(...)
# ──────────────────────────────────────────────────────────────────────────────

class AmplitudeTTS(DeepgramTTS):
    """
    Drop-in wrapper around deepgram.TTS that streams amplitude data to robot_eyes.

    Usage:
        from amplitude_tts import AmplitudeTTS
        session = AgentSession(
            tts=AmplitudeTTS(model="aura-luna-en"),
            ...
        )
    """

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _TappingChunkedStream:
        _ensure_pacer()
        return _TappingChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _TappingSynthesizeStream:
        _ensure_pacer()
        stream = _TappingSynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream
