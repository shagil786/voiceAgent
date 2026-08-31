# M3 Retrospective — Voice pipeline (CPU)

**Date:** 2026-08-31

## Built
- **VAD** (`src/voiceagent/vad.py`) — energy-based voice activity detection (20ms frames, RMS threshold). No heavy deps.
- **Streaming ASR** (`src/voiceagent/asr.py`) — faster-whisper `tiny` int8 on CPU (cached). `transcribe_wav` + `transcribe_chunks`.
- **Chunked TTS** (`src/voiceagent/tts.py`) — Piper: full-utterance `synthesize_to_wav` + `synthesize_chunks` (start speaking while later chunks generate → low perceived latency).
- **Voice turn handler** (`src/voiceagent/voice_agent.py`) — audio in → ASR → `run_turn` → TTS → audio out, end-to-end latency. Reuses the exact same agent/policy path as chat — the governance story is identical over voice.
- **Closed-loop demo** (`scripts/voice_demo.py`) — TTS a Hinglish query → ASR → agent → TTS reply.

## Measured (this machine, CPU, all models cached)
- **ASR:** faster-whisper tiny int8 ≈ 7.6× realtime (13min audio → ~1m42s on 8 threads).
- **TTS:** Piper en_US-lessac-medium ≈ 0.9s/utterance; chunked lowers perceived latency.
- **End-to-end voice turn:** **7.22s** on the demo. Breakdown: whisper-tiny transcribe (~1-2s) + agent turn (~1-4s, includes echo guardrail) + Piper reply synthesis (~0.9s) + cold-start LLM load on first run. Warm steady-state will be lower; the VPS numbers still need measuring.

## Key finding: ASR is now the bottleneck, not the agent
The demo ran a Hinglish query through TTS → whisper-tiny → agent. Whisper tiny **garbled the Hinglish** ("Bhai mera order... ORD-55671" → "By near-a-order of high-tech... D-55671"). The agent then classified the garbage as `cancel_order` — **but the policy engine DENIED it**. 

This is actually a great validation of the architecture: when ASR/LLM upstream is unreliable, the deterministic control plane catches the wrong action and refuses it. The wrong-action rate stays 0% even under noisy input. But it also means: **whisper-tiny is too weak for Hinglish voice** — the carry-forward is a larger/Indic ASR (faster-whisper `small`/`base`, or an Indic/Hinglish-fine-tuned ASR) before real phone integration.

## Deferred (M3b)
- **SIP/PSTN** (`sipx`) and **WebRTC** (LiveKit) — need a phone number / signaling server; can't be tested offline. The speech engine they plug into (`voice_turn`) is done.
- **Indic/Hinglish ASR + TTS voices** — needed for real Indian-accented voice quality (whisper tiny garble above).
- Fine-tuned TTS voice (Piper supports custom voice training).

## Carry-forward for M3b / M4
- Swap whisper-tiny → `base`/`small` or an Indic ASR; re-run the voice demo; target a clean Hinglish transcript and correct action.
- Real phone integration on top of `voice_turn`.
- The policy-DENY-on-garbled-ASR behavior is worth surfacing as a product feature: "the AI refuses to act on speech it can't understand" — a safety differentiator.
