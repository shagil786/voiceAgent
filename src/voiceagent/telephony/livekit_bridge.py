"""Offline-testable bridge between room audio and the turn loop.

Pure pump mechanics: the session owns a `StreamingVAD` plus a playback
queue, and emits completed utterances via an `on_utterance` callback.
No ASR/TTS/orchestrator imports — that wiring lives in the worker layer.
"""
from __future__ import annotations

from collections import deque
from itertools import count
from typing import Any, Callable

from voiceagent.telephony.audio import chunk_frames, resample_16k_to_48k
from voiceagent.telephony.stream import BargeInController, StreamingVAD

_FRAME_BYTES_16K_20MS = 320 * 2  # 20ms @16k mono int16 = 640 bytes


class BridgeSession:
    """Feeds 16k PCM in, serves 48k 10ms playback chunks out."""

    def __init__(
        self,
        on_utterance: Callable[[bytes], tuple[str, bytes]],
        on_barge_in: Callable[[Any], Any] | None = None,
    ) -> None:
        self._on_utterance = on_utterance
        self._on_barge_in_external = on_barge_in
        self._play: deque[bytes] = deque()
        self._stopped = False
        self._turn_ids = count(1)
        self._vad = StreamingVAD(
            barge_in_controller=BargeInController(on_barge_in=self._handle_barge_in)
        )

    def _handle_barge_in(self, event: Any) -> None:
        # Stop audition, keep session: drop queued audio, stay usable.
        self._play.clear()
        if self._on_barge_in_external is not None:
            self._on_barge_in_external(event)

    def feed_pcm16(self, frame_20ms: bytes) -> None:
        if self._stopped:
            return
        if len(frame_20ms) != _FRAME_BYTES_16K_20MS:
            raise ValueError(
                f"feed_pcm16 expects exactly {_FRAME_BYTES_16K_20MS} bytes "
                f"(20ms @16k mono int16), got {len(frame_20ms)}"
            )
        events = self._vad.process_frame(frame_20ms)
        audio = events.get("complete_audio")
        if audio is not None:
            _, reply_wav_16k = self._on_utterance(bytes(audio))
            if len(reply_wav_16k) % 2:
                raise ValueError("reply wav must be int16-aligned")
            upsampled = resample_16k_to_48k(reply_wav_16k)
            self._play.extend(chunk_frames(upsampled, 10, 48000))
            self._vad.barge_in.start_speaking(f"turn-{next(self._turn_ids)}")

    def take_playback(self) -> bytes | None:
        if self._stopped or not self._play:
            return None
        chunk = self._play.popleft()
        if not self._play:
            self._vad.barge_in.stop_speaking()
        return chunk

    def barge_in(self) -> None:
        """External trigger (e.g. loud uplink): clears pending playback."""
        self._vad.barge_in.trigger_barge_in()

    def stop(self) -> None:
        self._stopped = True
        self._play.clear()
        self._vad.barge_in.stop_speaking()
