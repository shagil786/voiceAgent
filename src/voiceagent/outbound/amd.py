# src/voiceagent/outbound/amd.py
"""Sub-600ms Answering Machine Detection (AMD).
Differentiates between a live human, an answering machine/voicemail,
and carrier IVR greetings in under 600ms of call connect.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from voiceagent.vad import FRAME_MS, is_speech


class CallParty(str, Enum):
    HUMAN = "HUMAN"
    MACHINE = "MACHINE"
    BEEP = "BEEP"
    SILENCE = "SILENCE"
    ANALYZING = "ANALYZING"


@dataclass
class AMDResult:
    classification: CallParty
    confidence: float
    speech_duration_ms: float
    silence_duration_ms: float
    latency_ms: float


class Sub600msAMD:
    """Telco-grade Answering Machine Detector operating within 600ms."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = FRAME_MS,
        max_decision_time_ms: int = 600,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_decision_time_ms = max_decision_time_ms

        self._elapsed_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._initial_burst_done = False
        self._classification = CallParty.ANALYZING

    @property
    def classification(self) -> CallParty:
        return self._classification

    def process_frame(self, frame: bytes) -> AMDResult:
        """Process 20ms frame during the first 600ms of connect."""
        self._elapsed_ms += self.frame_ms
        speech = is_speech(frame, self.sample_rate)

        if speech:
            if not self._initial_burst_done:
                self._speech_ms += self.frame_ms
            else:
                # Started speaking again after pause
                pass
        else:
            if self._speech_ms > 0:
                self._initial_burst_done = True
                self._silence_ms += self.frame_ms
            else:
                self._silence_ms += self.frame_ms

        # Check for Pure Tone / Beep (Answering machine beep)
        if self._is_carrier_beep(frame):
            self._classification = CallParty.BEEP
            return AMDResult(
                classification=CallParty.BEEP,
                confidence=0.95,
                speech_duration_ms=self._speech_ms,
                silence_duration_ms=self._silence_ms,
                latency_ms=self._elapsed_ms,
            )

        # Heuristic 1: Quick Human Greeting ("Hello?")
        # Human pattern: 150ms-450ms speech burst followed by > 120ms pause within 600ms window
        if self._initial_burst_done and 150 <= self._speech_ms <= 500 and self._silence_ms >= 120:
            self._classification = CallParty.HUMAN
            return AMDResult(
                classification=CallParty.HUMAN,
                confidence=0.92,
                speech_duration_ms=self._speech_ms,
                silence_duration_ms=self._silence_ms,
                latency_ms=self._elapsed_ms,
            )

        # Heuristic 2: Automated Voicemail / Carrier Greeting
        # Machine pattern: Speech continues uninterrupted past 550ms without pause
        if self._speech_ms >= 550:
            self._classification = CallParty.MACHINE
            return AMDResult(
                classification=CallParty.MACHINE,
                confidence=0.90,
                speech_duration_ms=self._speech_ms,
                silence_duration_ms=self._silence_ms,
                latency_ms=self._elapsed_ms,
            )

        # Heuristic 3: Timeout decision at 600ms
        if self._elapsed_ms >= self.max_decision_time_ms:
            if self._speech_ms >= 150 and self._silence_ms >= 100:
                self._classification = CallParty.HUMAN
            elif self._speech_ms == 0:
                self._classification = CallParty.SILENCE
            else:
                self._classification = CallParty.MACHINE

            return AMDResult(
                classification=self._classification,
                confidence=0.85,
                speech_duration_ms=self._speech_ms,
                silence_duration_ms=self._silence_ms,
                latency_ms=self._elapsed_ms,
            )

        return AMDResult(
            classification=CallParty.ANALYZING,
            confidence=0.0,
            speech_duration_ms=self._speech_ms,
            silence_duration_ms=self._silence_ms,
            latency_ms=self._elapsed_ms,
        )

    def _is_carrier_beep(self, frame: bytes) -> bool:
        """Detect concentrated spectral energy around standard 1000Hz beep."""
        n = len(frame) // 2
        if n == 0:
            return False
        samples = struct.unpack(f"<{n}h", frame[: n * 2])
        zero_crossings = sum(1 for i in range(1, len(samples)) if (samples[i-1] >= 0 > samples[i]) or (samples[i-1] < 0 <= samples[i]))
        approx_freq = (zero_crossings * self.sample_rate) / (2 * n)
        # Check if frequency matches standard voicemail beep (~950Hz - 1050Hz) with high amplitude
        rms = math.sqrt(sum(s * s for s in samples) / n)
        return 900 <= approx_freq <= 1100 and rms > 5000
