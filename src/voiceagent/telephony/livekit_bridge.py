"""Offline-testable bridge between room audio and the turn loop.

Pure pump mechanics: the session owns a `StreamingVAD` plus a playback
queue, and emits completed utterances via an `on_utterance` callback.
No ASR/TTS/orchestrator imports — that wiring lives in the worker layer.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from itertools import count
from typing import Any, Callable

from voiceagent.telephony.audio import chunk_frames, resample_16k_to_48k
from voiceagent.telephony.stream import BargeInController, StreamingVAD

_FRAME_BYTES_16K_20MS = 320 * 2  # 20ms @16k mono int16 = 640 bytes

logger = logging.getLogger(__name__)


class BridgeSession:
    """Feeds 16k PCM in, serves 48k 10ms playback chunks out."""

    def __init__(
        self,
        on_utterance: Callable[[bytes], tuple[str, bytes]],
        on_barge_in: Callable[[Any], Any] | None = None,
        *,
        threaded: bool = False,
    ) -> None:
        """`threaded=True` runs each utterance turn on a worker thread so the
        audio pump NEVER freezes: the event loop keeps consuming frames,
        playback keeps flowing and barge-in stays live while ASR/brain/TTS
        work. Without it a turn (1-8s of sync ASR+HTTP) stalls the whole
        room loop — the caller's next words go unheard (dead air) or flush
        in later as a stale burst that triggers a ghost reply. The sync mode
        (default) keeps offline tests deterministic."""
        self._on_utterance = on_utterance
        self._on_barge_in_external = on_barge_in
        self._play: deque[bytes] = deque()
        self._stopped = False
        self._turn_ids = count(1)
        self._threaded = threaded
        self._utterance_q: queue.Queue[bytes] = queue.Queue()
        self._worker: threading.Thread | None = None
        if self._threaded:
            self._worker = threading.Thread(
                target=self._turn_worker, name="va-turn-worker", daemon=True)
            self._worker.start()
        self._vad = StreamingVAD(
            barge_in_controller=BargeInController(on_barge_in=self._handle_barge_in)
        )

    def _turn_worker(self) -> None:
        """Serialize utterance turns off the audio thread. One worker keeps
        ASR/brain/TTS sequential (whisper and piper are not safe under
        concurrent calls); utterances arriving while a turn runs are answered
        in order after it."""
        while not self._stopped:
            try:
                audio = self._utterance_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                _, reply_wav_16k = self._on_utterance(bytes(audio))
            except Exception:
                logger.warning("turn worker: utterance failed",
                               exc_info=True)
                continue
            if not reply_wav_16k:
                continue  # non-speech / skipped turn: no playback
            if self._stopped:
                return
            if len(reply_wav_16k) % 2:
                logger.warning("turn worker: reply wav not int16-aligned; "
                               "dropping")
                continue
            upsampled = resample_16k_to_48k(reply_wav_16k)
            self._play.extend(chunk_frames(upsampled, 10, 48000))
            self._vad.barge_in.start_speaking(f"turn-{next(self._turn_ids)}")

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
        if audio is None:
            return
        if self._threaded:
            self._utterance_q.put(bytes(audio))
            return
        _, reply_wav_16k = self._on_utterance(bytes(audio))
        if len(reply_wav_16k) % 2:
            raise ValueError("reply wav must be int16-aligned")
        upsampled = resample_16k_to_48k(reply_wav_16k)
        self._play.extend(chunk_frames(upsampled, 10, 48000))
        self._vad.barge_in.start_speaking(f"turn-{next(self._turn_ids)}")

    def has_pending_playback(self) -> bool:
        """True while unplayed chunks remain (barge-in clears the queue)."""
        return bool(self._play)

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
        if self._worker is not None and self._worker.is_alive():
            # Drain the queue so a mid-turn worker cannot append after stop
            # and cannot hold the interpreter on shutdown.
            self._worker.join(timeout=5.0)
