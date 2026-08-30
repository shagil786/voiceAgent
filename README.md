# VoiceAgent — M0 Benchmark Spike

Builds the CPU-only pipeline (ASR → RAG → small LLM → TTS) and measures
turn latency, resolution rate, hallucination rate, and cost/conversation
over ~1,000 English + Hinglish eval conversations.

Gate: latency ≤ 2s and resolution ≥ 75%. See docs/superpowers/specs/2026-08-30-voiceagent-design.md.
