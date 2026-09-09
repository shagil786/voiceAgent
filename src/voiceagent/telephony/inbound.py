"""Inbound worker: webhook → join → greet → loop → hangup.

LiveKit is transport only — every spoken turn is one governed
`Orchestrator.handle_turn`, including the greeting (never a canned file).

All `livekit` imports are function-level (lazy) so unit tests never require
the dep on the cold path.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

from voiceagent.langid import detect_language

logger = logging.getLogger(__name__)

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


def _mask_pii(text: str) -> str:
    """Mask phone-shaped digit runs (7+) in log lines — transcripts and the
    accumulated phone slot flow through logger.info below, and log
    aggregators are outside retention/erasure reach. Short runs (order ids,
    amounts, ratings) stay readable for debugging."""
    return re.sub(r"\d{7,}", "****", text)


def _default_tts(text: str, language: str | None = None) -> bytes:
    """Reply text -> 16k-ish mono int16 PCM frames via temp-WAV `speak`.

    Reuses the existing file-output `speak()` pattern (synthesize to a temp
    WAV, then read the frames back); no new TTS API is added.

    Voice-match containment: when the trunk declares a language but the
    brain's reply is detected as another language that HAS a registered
    voice, the reply speaks in its own voice instead of garbling non-Latin
    script through the declared voice (the 2026-09 Thai shape). Detection
    without a voice keeps the declared voice (best effort, warn as usual).
    """
    from voiceagent.tts import HINGLISH_VOICE_LANG, VOICE_REGISTRY, speak

    effective = language
    if language:
        # Tags ("en-US") never match registry keys — normalize to the base
        # code first (avoids a spurious fallback warning on every turn).
        declared_base = str(language).strip().lower().replace(
            "_", "-").split("-")[0] or None
        if declared_base in VOICE_REGISTRY:
            effective = declared_base
        detected = detect_language(text)
        if detected == "hinglish":
            detected = HINGLISH_VOICE_LANG  # same mapping as voice_for
        if (detected != declared_base and detected in VOICE_REGISTRY):
            logger.warning(
                "tts voice mismatch: trunk declares %r but reply detects "
                "as %r — speaking with the reply's voice", language,
                detected)
            effective = detected

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        speak(text, language=effective, out_path=path)
        with wave.open(path, "rb") as w:
            pcm = w.readframes(w.getnframes())
            rate = w.getframerate()
        # Piper voices are natively 22050 Hz; the pipeline runs at 16 kHz.
        # Publish the pipeline rate so 16k->48k room resampling is exact.
        if rate != _PIPELINE_SAMPLE_RATE:
            from voiceagent.telephony.audio import resample_to_16k
            pcm = resample_to_16k(pcm, rate)
        return pcm
    finally:
        Path(path).unlink(missing_ok=True)


# --- turn wiring --------------------------------------------------------------

_RE_ENDERS = ("bye", "goodbye", "good night", "have a good day",
              "have a great day", "that's all", "that is all", "that will be all")


def _is_clear_farewell(text: str) -> bool:
    """High-confidence call-ending signal, used as the hangup GUARD.

    A single ambiguous utterance must never disconnect a live call: ASR
    fabricates fluent farewells on noise/babble ('thank you guys, I'll see
    you guys in my next video' was heard on a call where the caller said no
    such thing) and long conversational lines are not clean endings. Only a
    SHORT closing carrying a hard ender token ends the call; everything else
    keeps the line open (the brain already replied, so the caller simply
    continues)."""
    t = text.strip().lower()
    if not t or len(t) > 60:
        return False  # long lines are conversations, not farewells
    if any(h in t for h in _RE_ENDERS):
        return True
    if len(t) <= 16 and ("thank" in t or "thanks" in t or "ok bye" in t):
        return True  # short thanks-only / ok-bye closes
    return False


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
    # Phone slot (ADR-002-adjacent mechanics): callers give their number in
    # pieces across turns; the slot accumulates digits (spoken words
    # converted) so the brain can look orders up without re-asking.
    phone_digits = ""

    _re_phone_context = re.compile(r"number|phone|contact|\u0928\u0902\u092c\u0930", re.I)

    def _absorb_phone_digits(text: str) -> None:
        """Only PHONE-SHAPED utterances feed the slot:
        - a phone/number keyword in the sentence, OR
        - one long digit run (>=7 — a number read straight out), OR
        - a continuation while the slot is already open AND the utterance is
          purely number-ish (every word is a number word / digits).
        'My order is 4821' must NOT poison the slot with order digits."""
        nonlocal phone_digits
        from voiceagent.entities import _token_value
        digits = "".join(re.findall(r"\d+", text))
        alpha_toks = re.findall(r"[a-z]+", text.lower())
        numberish = (all(_token_value(t) is not None for t in alpha_toks)
                     if alpha_toks else bool(digits))
        if not digits and (_re_phone_context.search(text)
                           or (phone_digits and numberish)):
            # keyword present (first read) OR a continuation while the slot
            # is already open: convert the number-word tokens only —
            # "my"/"is" contribute nothing to the digits.
            digits = "".join(str(_token_value(t) or "")
                             for t in alpha_toks)
        phone_shaped = (_re_phone_context.search(text)
                        or max((len(run) for run in re.findall(r"\d+", text)),
                               default=0) >= 7
                        or (phone_digits and digits and numberish))
        if digits and phone_shaped:
            phone_digits = (phone_digits + digits)[-16:]

    def _text_for_brain(user_text: str) -> str:
        if len(phone_digits) >= 10:
            return (f"{user_text} (caller phone digits so far — may be "
                    f"partial: {phone_digits})")
        return user_text

    _re_nonspeech = re.compile(r"^[\W一-鿿ぁ-ゟァ-ヿ]+$")

    def turn_fn(pcm16: bytes) -> tuple[str, bytes]:
        from voiceagent.orchestrator import _FALLBACK_REPLY
        from voiceagent.swarm.frontier import FrontierError

        t0 = time.monotonic()
        user_text = asr_fn(pcm16)
        # ASR silence gate: Qwen hallucinates lone CJK glyphs / punctuation on
        # background noise ('的。'). Such a transcript is NO speech — skip the
        # whole governed turn (no brain call, no reply) instead of answering
        # nobody. Real words (any script) always pass.
        _absorb_phone_digits(user_text)
        raw_text = user_text  # pre-augmentation transcript (farewell check)
        user_text = _text_for_brain(user_text)
        stripped = user_text.strip()
        if not stripped:
            return "", b""  # empty ASR output is non-speech too
        cjk_only = (stripped and re.search(r"[一-鿿ぁ-ゟァ-ヿ]", stripped)
                    and not re.search(r"[A-Za-z0-9\u0900-\u097F]", stripped))
        if (len(stripped) <= 2 and _re_nonspeech.match(stripped)) or cjk_only:
            logger.info("turn: skipped non-speech ASR output %r",
                        _mask_pii(user_text[:40]))
            return "", b""
        t_asr = time.monotonic()
        try:
            result = orchestrator.handle_turn(session_id, user_text)
        except FrontierError:
            # Brain outage must not kill the call with dead air: speak the
            # handoff line (the same graceful degradation as an empty model
            # reply) and let the room loop continue.
            logger.warning("turn: brain unreachable — serving handoff line",
                           exc_info=True)

            class _OutageResult:
                reply = _FALLBACK_REPLY
                actions = [{"action": "escalate_to_human",
                            "verdict": "ESCALATE", "ok": False,
                            "error": "brain_unreachable"}]

            result = _OutageResult()
        t_brain = time.monotonic()
        reply = result.reply
        wav_out = tts_fn(reply)
        if isinstance(wav_out, (tuple, list)):
            wav_out = wav_out[1]
        wav_out = bytes(wav_out)
        t_tts = time.monotonic()
        # Turn evidence: transcript -> governed actions -> reply -> timing.
        # This is how a silent/garbled/late/wrong-tool turn is diagnosed
        # after the fact.
        acts = getattr(result, "actions", None) or []
        if any(a.get("action") == "end_call" and a.get("ok") for a in acts):
            # Hangup guard: only a CLEAR short farewell disconnects. An
            # end_call on a long/ambiguous transcript (often an ASR
            # fabrication — e.g. a caller who never said goodbye) keeps the
            # line open; the agent's reply already played, so the caller can
            # just keep talking.
            if _is_clear_farewell(raw_text):
                turn_fn.call_ended = True  # room loop hangs up after playback
            else:
                logger.info(
                    "end_call ALLOW but transcript %r is not a clear "
                    "farewell — keeping the call alive",
                    _mask_pii(raw_text[:100]))
        act_sig = "; ".join(
            f"{a.get('action')}={a.get('verdict')}/{'ok' if a.get('ok') else (a.get('error') or 'err')}"
            for a in acts)
        logger.info(
            "turn: asr=%.2fs brain=%.2fs tts=%.2fs | caller=%r | tools=[%s] | reply[%s]=%r",
            t_asr - t0, t_brain - t_asr, t_tts - t_brain,
            _mask_pii(user_text[:120]), act_sig, detect_language(reply),
            _mask_pii(reply[:120]),
        )
        return reply, wav_out

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
    from livekit import rtc

    from voiceagent.telephony.audio import resample_48k_to_16k
    from voiceagent.telephony.livekit_bridge import BridgeSession

    orchestrator = _deps_get(deps, "orchestrator")
    session_id = _deps_get(deps, "session_id", room_name)
    language = _deps_get(deps, "language")
    asr = _deps_get(deps, "asr")
    tts = _deps_get(deps, "tts")

    turn_fn = make_turn_fn(orchestrator, session_id, asr=asr, tts=tts, language=language)
    session = BridgeSession(on_utterance=lambda pcm: turn_fn(pcm),
                                threaded=True)

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

    # Bound before `try`: the no-SIP-track early return flows through the
    # same `finally`, which must never touch an unassigned variable.
    pump_stop = asyncio.Event()
    pump_task: asyncio.Task | None = None

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

        # Playback pump: the ONLY writer to `source`, paced by
        # `await capture_frame` (backpressure = real-time playback). It
        # polls `session.take_playback()` directly, so a barge-in clearing
        # the session queue silences it within one 10ms chunk. The feed
        # loop below never touches `source`, keeping VAD/barge-in live
        # while the agent is speaking.
        pump_task = asyncio.create_task(
            _playback_pump(source, session, pump_stop)
        )

        # Greeting: the tenant's DECLARED greeting text is spoken instantly
        # (no brain roundtrip — the first-second experience is declared data).
        # No declared greeting -> one governed greeting turn (legacy path).
        declared_greeting = _deps_get(deps, "greeting") or ""
        if declared_greeting.strip():
            logger.info("greeting: declared text (%d chars)", len(declared_greeting))
            greet_text = declared_greeting
        else:
            from voiceagent.orchestrator import _FALLBACK_REPLY
            from voiceagent.swarm.frontier import FrontierError
            try:
                greet_result = orchestrator.handle_turn(
                    session_id, GREETING_TRANSCRIPT)
                greet_text = greet_result.reply
            except FrontierError:
                # Brain outage on pickup: greet with the handoff line rather
                # than dropping the call before it starts.
                logger.warning("greeting: brain unreachable — serving "
                               "handoff line", exc_info=True)
                greet_text = _FALLBACK_REPLY
        greet_wav = tts(greet_text) if tts is not None else _default_tts(
            greet_text, language
        )
        if isinstance(greet_wav, (tuple, list)):
            greet_wav = greet_wav[1]
        await _publish_pcm16(source, bytes(greet_wav))

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
            if getattr(turn_fn, "call_ended", False):
                # The farewell must be HEARD before we hang up: wait for the
                # session queue to drain (the pump consumes it in real time)
                # instead of breaking on the turn that triggered end_call.
                if not session.has_pending_playback():
                    logger.info("call ended by agent (farewell played)")
                    break
            if disconnected.is_set():
                break
        return True
    finally:
        pump_stop.set()
        if pump_task is not None:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
        session.stop()
        await room.disconnect()


async def _playback_pump(source: Any, session: Any, stop: asyncio.Event) -> None:
    """Drain `session.take_playback()` into `source` until `stop` is set.

    `AudioSource.capture_frame` is a coroutine — awaiting it is what
    actually queues the frame AND paces playback to real time (its
    backpressure releases one frame per 10ms). Polling the session's own
    queue means a barge-in (which clears that queue) stops playback on
    the next iteration.
    """
    from livekit import rtc

    while not stop.is_set():
        chunk = session.take_playback()
        if chunk is None:
            await asyncio.sleep(0.005)
            continue
        await source.capture_frame(
            rtc.AudioFrame(chunk, _ROOM_SAMPLE_RATE, 1, len(chunk) // 2)
        )


async def _publish_pcm16(source: Any, reply_wav_16k: bytes) -> None:
    """Upsample a greeting reply to 48k and publish via `AudioSource`.

    `AudioSource.capture_frame` is a coroutine in livekit rtc (its await
    provides backpressure/real-time pacing); each frame MUST be awaited
    or nothing is queued and the caller hears silence.
    """
    from livekit import rtc

    from voiceagent.telephony.audio import chunk_frames, resample_16k_to_48k

    upsampled = resample_16k_to_48k(bytes(reply_wav_16k))
    for chunk in chunk_frames(upsampled, 10, _ROOM_SAMPLE_RATE):
        await source.capture_frame(
            rtc.AudioFrame(chunk, _ROOM_SAMPLE_RATE, 1, len(chunk) // 2)
        )


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
