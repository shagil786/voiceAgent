# src/voiceagent/telephony/stream.py
"""Full-duplex streaming audio pipeline with Silero/energy VAD,
frame-by-frame speech detection, and instant barge-in interruption.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from voiceagent.vad import FRAME_MS, is_speech


class AudioState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class BargeInEvent:
    timestamp: float
    interrupted_turn_id: str
    audio_offset_ms: float = 0.0


class BargeInController:
    """Manages playback state and triggers instant cancellation
    when customer speech is detected during agent speech."""

    def __init__(self, on_barge_in: Callable[[BargeInEvent], Any] | None = None):
        self._is_speaking = False
        self._active_turn_id: str | None = None
        self._playback_start_ts: float | None = None
        self._on_barge_in = on_barge_in
        self._cancelled = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def active_turn_id(self) -> str | None:
        return self._active_turn_id

    def start_speaking(self, turn_id: str) -> None:
        self._is_speaking = True
        self._active_turn_id = turn_id
        self._playback_start_ts = time.time()
        self._cancelled = False

    def stop_speaking(self) -> None:
        self._is_speaking = False
        self._active_turn_id = None
        self._playback_start_ts = None
        self._cancelled = False

    def trigger_barge_in(self) -> BargeInEvent | None:
        """Immediately halts ongoing playback on user interruption."""
        if not self._is_speaking or self._cancelled:
            return None
        self._cancelled = True
        offset_ms = 0.0
        if self._playback_start_ts:
            offset_ms = (time.time() - self._playback_start_ts) * 1000.0
        event = BargeInEvent(
            timestamp=time.time(),
            interrupted_turn_id=self._active_turn_id or "",
            audio_offset_ms=offset_ms,
        )
        self._is_speaking = False
        self._active_turn_id = None
        self._playback_start_ts = None
        if self._on_barge_in:
            if asyncio.iscoroutinefunction(self._on_barge_in):
                asyncio.create_task(self._on_barge_in(event))
            else:
                self._on_barge_in(event)
        return event


class StreamingVAD:
    """Consumes 20ms 16-bit PCM audio frames, tracks speech boundaries,
    and signals turn endpoints and barge-in events."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = FRAME_MS,
        speech_pad_ms: int = 60,
        silence_timeout_ms: int = 400,
        barge_in_controller: BargeInController | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.speech_pad_frames = max(1, speech_pad_ms // frame_ms)
        self.silence_timeout_frames = max(1, silence_timeout_ms // frame_ms)
        self.barge_in = barge_in_controller or BargeInController()

        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._collected_audio = bytearray()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def process_frame(self, frame: bytes) -> dict[str, Any]:
        """Process one 20ms audio frame.
        Returns a dict of event flags:
        {
            "speech_started": bool,
            "speech_ended": bool,
            "barge_in": BargeInEvent | None,
            "complete_audio": bytes | None,
        }
        """
        active = is_speech(frame, self.sample_rate)
        events: dict[str, Any] = {
            "speech_started": False,
            "speech_ended": False,
            "barge_in": None,
            "complete_audio": None,
        }

        if active:
            self._consecutive_speech += 1
            self._consecutive_silence = 0

            if not self._in_speech and self._consecutive_speech >= self.speech_pad_frames:
                self._in_speech = True
                events["speech_started"] = True
                # Check for barge-in: customer started speaking while agent was speaking
                if self.barge_in.is_speaking:
                    events["barge_in"] = self.barge_in.trigger_barge_in()

            if self._in_speech:
                self._collected_audio.extend(frame)
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

            if self._in_speech:
                self._collected_audio.extend(frame)
                if self._consecutive_silence >= self.silence_timeout_frames:
                    self._in_speech = False
                    events["speech_ended"] = True
                    events["complete_audio"] = bytes(self._collected_audio)
                    self._collected_audio.clear()

        return events
