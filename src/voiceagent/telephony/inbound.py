"""Inbound worker: webhook → join → greet → loop → hangup.

LiveKit is transport only — every spoken turn is one governed
`Orchestrator.handle_turn`, including the greeting (never a canned file).

All `livekit` imports are function-level (lazy) so unit tests never require
the dep on the cold path.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

# Greeting goes through the governed turn, never a canned audio file.
GREETING_TRANSCRIPT = "(Inbound call connected — greet the caller.)"

_ROOM_SAMPLE_RATE = 48000
_PIPELINE_SAMPLE_RATE = 16000


# --- default ASR/TTS (temp-WAV seams over the existing engines) -------------

def _default_asr(pcm16: bytes, language: str | None = None) -> str:
    """16k mono int16 PCM -> text via temp WAV + `transcribe_wav_routed`."""
    from voiceagent.asr import transcribe_wav_routed

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_PIPELINE_SAMPLE_RATE)
            w.writeframes(pcm16)
        return transcribe_wav_routed(path, language)
    finally:
        Path(path).unlink(missing_ok=True)


def _default_tts(text: str, language: str | None = None) -> bytes:
    """Reply text -> 16k-ish mono int16 PCM frames via temp-WAV `speak`.

    Reuses the existing file-output `speak()` pattern (synthesize to a temp
    WAV, then read the frames back); no new TTS API is added.
    """
    from voiceagent.tts import speak

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        speak(text, language=language, out_path=path)
        with wave.open(path, "rb") as w:
            return w.readframes(w.getnframes())
    finally:
        Path(path).unlink(missing_ok=True)


# --- turn wiring --------------------------------------------------------------

def make_turn_fn(
    orchestrator: Any,
    session_id: str,
    asr: Callable[[bytes], str] | None = None,
    tts: Callable[[str], Any] | None = None,
    language: str | None = None,
) -> Callable[[bytes], tuple[str, bytes]]:
    """Build the utterance callback: PCM in -> `(reply_text, reply_wav16)` out.

    `asr`/`tts` are injectable (no model loads in CI); defaults are the
    temp-WAV `transcribe_wav_routed` + `speak` seams above. An injectable
    `tts` may return raw wav bytes or a `(text, wav_bytes)` tuple.
    """
    asr_fn = asr if asr is not None else (lambda pcm: _default_asr(pcm, language))
    tts_fn = tts if tts is not None else (lambda text: _default_tts(text, language))

    def turn_fn(pcm16: bytes) -> tuple[str, bytes]:
        user_text = asr_fn(pcm16)
        result = orchestrator.handle_turn(session_id, user_text)
        reply = result.reply
        wav_out = tts_fn(reply)
        if isinstance(wav_out, (tuple, list)):
            wav_out = wav_out[1]
        return reply, bytes(wav_out)

    return turn_fn


# --- webhook -------------------------------------------------------------------

def _default_validate(config: Any) -> Callable[[str, str], Any]:
    """LiveKit-signed webhook validation via `WebhookReceiver`.

    NOTE: `WebhookReceiver` takes a `TokenVerifier(api_key, api_secret)`
    (verified against installed `livekit-api==1.2.1` source) — NOT
    `(key, secret)` positionally.
    """
    def validate(body: str, sig: str) -> Any:
        # Constructed per request (fail closed): missing key/secret or a
        # bad signature raises here, and the handler maps any error to
        # False — a bad config never crashes worker startup or serving.
        try:
            from livekit.api import WebhookReceiver
            from livekit.api.access_token import TokenVerifier

            receiver = WebhookReceiver(
                TokenVerifier(config.livekit_key, config.livekit_secret)
            )
            return receiver.receive(body, sig)
        except Exception as exc:
            raise ValueError(f"webhook validation failed: {exc}") from exc

    return validate


def _event_fields(event: Any) -> tuple[Any, Any]:
    """Extract `(event_name, room_name)` from a dict fake or a livekit
    `WebhookEvent` protobuf."""
    if isinstance(event, dict):
        room = event.get("room") or {}
        return event.get("event"), room.get("name")
    room = getattr(event, "room", None)
    return getattr(event, "event", None), getattr(room, "name", None)


def webhook_handler(
    config: Any,
    join_room: Callable[[str], Any],
    validate: Callable[[str, str], Any] | None = None,
) -> Callable[[str, str], bool]:
    """Return a `(body, signature) -> bool` webhook handler.

    Validates via livekit `WebhookReceiver` (or the injected `validate`
    fake), filters to `room_started` events whose room name starts with the
    configured prefix, and calls `join_room(room_name)` on match.
    """
    prefix = config.livekit_room_prefix if config is not None else "call-"
    validate_fn = validate if validate is not None else _default_validate(config)

    def handler(body: str, sig: str) -> bool:
        try:
            event = validate_fn(body, sig)
        except Exception:
            return False
        name, room_name = _event_fields(event)
        if name != "room_started":
            return False
        if not isinstance(room_name, str) or not room_name.startswith(prefix):
            return False
        join_room(room_name)
        return True

    return handler


# --- room session -----------------------------------------------------------------

def _deps_get(deps: Any, key: str, default: Any = None) -> Any:
    if deps is None:
        return default
    if isinstance(deps, dict):
        return deps.get(key, default)
    return getattr(deps, key, default)


def _mint_worker_token(config: Any, room_name: str, identity: str) -> str:
    """Mint a join token for the worker participant (lazy livekit import)."""
    from livekit.api import AccessToken
    from livekit.api.access_token import VideoGrants

    token = AccessToken(config.livekit_key, config.livekit_secret)
    token.with_identity(identity).with_grants(
        VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True)
    )
    return token.to_jwt()


def wait_for_sip_track(
    get_tracks: Callable[[], Any],
    timeout_s: float = 15,
    sleep: Callable[[float], None] | None = None,
) -> Any | None:
    """Poll `get_tracks()` until the first SIP audio track appears.

    Returns the track, or `None` if `timeout_s` elapses first. `sleep` is
    injectable (default: real `time.sleep`) so tests run instantly.
    """
    import time

    _sleep = sleep if sleep is not None else time.sleep
    deadline = time.monotonic() + timeout_s
    while True:
        track = get_tracks()
        if track is not None:
            return track
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        _sleep(min(0.5, remaining))


def ensure_room_sample_rate(sample_rate: int) -> None:
    """Fail fast when room audio is not the 48kHz mono contract."""
    if sample_rate != _ROOM_SAMPLE_RATE:
        raise ValueError(
            f"room audio must be {_ROOM_SAMPLE_RATE}Hz, got {sample_rate}"
        )


async def _wait_for_sip_track_async(
    get_tracks: Callable[[], Any], timeout_s: float
) -> Any | None:
    """Async mirror of `wait_for_sip_track` (awaits `asyncio.sleep`).

    The sync helper must not block the room's event loop — stalled loops
    stop processing the very subscription events the poll waits for — so
    the live path polls here while tests cover the sync twin.
    """
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        track = get_tracks()
        if track is not None:
            return track
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(0.5, remaining))


async def _run_room_async(room_name: str, config: Any, deps: Any) -> bool:
    """Join, greet via the governed turn, pump audio, leave on disconnect.

    Returns True when the call ran, False when no SIP track appeared
    before the timeout (room left, never greeted into the void).
    """
    import asyncio

    from livekit import rtc

    from voiceagent.telephony.audio import resample_48k_to_16k
    from voiceagent.telephony.livekit_bridge import BridgeSession

    orchestrator = _deps_get(deps, "orchestrator")
    session_id = _deps_get(deps, "session_id", room_name)
    language = _deps_get(deps, "language")
    asr = _deps_get(deps, "asr")
    tts = _deps_get(deps, "tts")

    turn_fn = make_turn_fn(orchestrator, session_id, asr=asr, tts=tts, language=language)
    session = BridgeSession(on_utterance=lambda pcm: turn_fn(pcm))

    token = _mint_worker_token(config, room_name, f"worker-{room_name}")
    room = rtc.Room()
    disconnected: asyncio.Event = asyncio.Event()

    def _on_leave(*_args: Any) -> None:
        disconnected.set()

    # Event names verified against installed `livekit==1.1.17` room.py;
    # callbacks take *args because `emit` forwards event payloads.
    room.on("disconnected", _on_leave)
    room.on("participant_disconnected", _on_leave)
    await room.connect(config.livekit_url, token)

    try:
        # Subscribe: first remote SIP audio track. The SIP participant may
        # join after us, so wait (bounded) instead of dropping dead calls.
        def get_tracks() -> Any | None:
            for _p, pub in list(room.remote_participants.items()):
                for _tid, tpub in list(pub.track_publications.items()):
                    if tpub.kind == rtc.TrackKind.KIND_AUDIO and tpub.track is not None:
                        return tpub.track
            return None

        timeout_s = _deps_get(deps, "sip_track_timeout_s", 15)
        track = await _wait_for_sip_track_async(get_tracks, timeout_s)
        if track is None:
            return False

        source = rtc.AudioSource(_ROOM_SAMPLE_RATE, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("worker-reply", source)
        await room.local_participant.publish_track(audio_track)

        # Greeting: one governed turn, spoken once after subscribe.
        greet_result = orchestrator.handle_turn(session_id, GREETING_TRANSCRIPT)
        greet_wav = tts(greet_result.reply) if tts is not None else _default_tts(
            greet_result.reply, language
        )
        if isinstance(greet_wav, (tuple, list)):
            greet_wav = greet_wav[1]
        _publish_pcm16(source, bytes(greet_wav))

        stream = rtc.AudioStream(track)
        pending = b""
        async for frame_event in stream:
            frame = frame_event.frame
            ensure_room_sample_rate(frame.sample_rate)
            pcm16 = resample_48k_to_16k(bytes(frame.data))
            pending += pcm16
            while len(pending) >= 640:
                session.feed_pcm16(pending[:640])
                pending = pending[640:]
                chunk = session.take_playback()
                while chunk is not None:
                    source.capture_frame(
                        rtc.AudioFrame(chunk, _ROOM_SAMPLE_RATE, 1, len(chunk) // 2)
                    )
                    chunk = session.take_playback()
            if disconnected.is_set():
                break
        return True
    finally:
        session.stop()
        await room.disconnect()


def _publish_pcm16(source: Any, reply_wav_16k: bytes) -> None:
    """Upsample a greeting reply to 48k and publish via `AudioSource`."""
    from livekit import rtc

    from voiceagent.telephony.audio import chunk_frames, resample_16k_to_48k

    upsampled = resample_16k_to_48k(bytes(reply_wav_16k))
    for chunk in chunk_frames(upsampled, 10, _ROOM_SAMPLE_RATE):
        source.capture_frame(rtc.AudioFrame(chunk, _ROOM_SAMPLE_RATE, 1, len(chunk) // 2))


def run_room_session(room_name: str, config: Any, deps: Any = None) -> bool:
    """Join `room_name`, run the greet→loop→hangup session, then return.

    Returns True when the call ran, False when no SIP track appeared
    before the timeout. Synchronous wrapper (one worker thread per room
    runs its own event loop); the async body above holds the only
    `from livekit import rtc` import, keeping unit-test cold paths
    dependency-free.
    """
    import asyncio

    return asyncio.run(_run_room_async(room_name, config, deps))
