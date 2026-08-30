# VoiceAgent — Product Design

**Date:** 2026-08-30
**Status:** Draft for review

## 1. Problem & Thesis

AI customer-support agents are a mature, well-funded category (Intercom/Fin, Sierra, Decagon, Ada, PolyAI). "A multilingual RAG voice agent" is not a defensible pitch. Incumbents claim 45–70+ languages, voice, chat, omnichannel, and enterprise controls.

The defensible opportunity is narrower:

> **"The AI customer-support agent you can actually control — running on infrastructure you own."**

The concrete wedge for India / emerging markets:

> **"Your AI support agent runs on a ₹3,000/month VPS, speaks Hinglish with measured quality, never takes an unauthorized action, and you pay only for what it resolves."**

Every claim in that sentence is either verifiable (benchmarks, cost per resolution), defensible (policy engine, data flywheel), or un-copyable by incumbents without cannibalizing their business model (CPU-only deployment, per-conversation pricing).

## 2. What We Are Not

We are not "another AI chatbot." We are not competing head-on with Sierra/PolyAI on voice quality or breadth of languages. We do not sell "our model is small." We do not price on tokens.

## 3. Product Line

| Product | What it is | Price model |
|---|---|---|
| **VoiceAgent Core** | AI voice + chat support agent, CPU-only, English + Hinglish, RAG, policy-enforced actions, human handoff | ₹8–₹12 per resolved conversation; free if not resolved |
| **Control Plane** | Policy engine (policy-as-code), full decision/audit log, permissions dashboard, tool gateway | Bundled with Core; standalone ₹50k/month |
| **Eval Suite** | Synthetic customer simulation, red teaming / prompt-injection tests, regression benchmarks, quality scoring | ₹25k/month or included in Enterprise |

## 4. Deployment Target: CPU / Small VPS, No GPU

This is the architectural commitment that makes the product commercially interesting. No GPU. No customer GPU procurement.

Target baseline: a 4–8 vCPU, 16GB RAM VPS (₹3k–₹5k/month), or the customer's existing server. Fully offline inference — no outbound network calls for ASR/LLM/embedding/TTS.

### Model stack (CPU-viable)

| Stage | Choice | Rationale |
|---|---|---|
| ASR | whisper.cpp (tiny/small) with VAD + streaming partial hypotheses | 20–50ms/chunk on CPU; partial results keep latency perceived-low |
| LLM | 1–3B class (Qwen 2.5, Phi-4, etc.) quantized 4-bit, llama.cpp / ONNX Runtime | ~1–2s/turn on 8 vCPU; fast enough with turn-taking design |
| Embeddings | sentence-transformers small model on CPU | no separate embedding service |
| Vector store | sqlite-vec (in-process) | no separate vector DB service to run |
| Reranker | small cross-encoder on CPU, optional | improves retrieval quality at low cost |
| Guardrails | small local safety classifier | deterministic-ish, runs locally |
| TTS | Piper / Coqui, Hindi + English voices | 50–100ms/utterance on CPU |
| Policy engine | embedded, deterministic | evaluated locally, zero network |

End-to-end latency target: **1.5–2s per turn** — competitive with a human agent and acceptable for voice. Voice turn-taking must make the system *appear* responsive while longer reasoning happens (streaming ASR, speculative generation, async tool calls).

**The pitch sentence nobody in the market can say:** *"Your entire AI support agent runs on a ₹3,000/month VPS, with your data, fully offline, handling tens of thousands of conversations a month."*

## 5. v1 Scope: English + Hinglish, Deep

One language pair, done at measured 95% quality, beats 12 languages at 75%.

- **English** — full support.
- **Hindi** — full support, formal Hindi (Devanagari).
- **Hinglish** — detection and handling of code-switching within a single conversation (e.g. *"Bhai mera order abhi tak nahi aaya"* → *"Actually, can you check if it's been shipped?"* → *"Kannada alli heli"* → intent/context preserved across language switches).

Other Indian languages (Tamil, Kannada, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Urdu) come after the engine is proven on the hardest pair.

Rationale: Hinglish is the most common informal language in Indian commerce. Incumbents claim "Hindi support" but their evals are on formal Hindi, not code-switched Hinglish. Doing it well is genuine technical defensibility, and the Hinglish conversation data becomes part of the moat.

## 6. Architecture

```
Voice call (PSTN/SIP) or chat
        ↓
VAD + Whisper streaming (CPU, partial hypotheses)
        ↓
[transcript]
        ↓
Intent classifier + RAG retrieval (CPU, sqlite-vec)
        ↓
Small LLM (1–3B, quantized, CPU) + guardrails
        ↓
proposed action (e.g. refund(10000))
        ↓
Policy Engine (deterministic, YAML policies)
        ↓
Allow / Deny / Escalate / Require-Auth
        ↓
Tool Gateway (customer's real APIs; preconditions, idempotency, timeouts)
        ↓
TTS (Piper/Coqui, CPU) → voice response
```

### Key design principles

- **Isolation:** ASR, LLM, RAG, policy, tools, TTS are separate units with clear interfaces, independently testable.
- **Determinism of actions:** the LLM proposes; only the policy engine disposes. The LLM is never trusted for authorization.
- **Offline:** everything runs in-process or on the local machine; no vendor dependency.
- **Observable:** every action attempt is logged (see §8).

## 7. Policy Engine (The Product, Not a Feature)

This is the centerpiece and the moat.

### Policy-as-code

```yaml
refund:
  max_without_approval: 5000
  require_auth: true
  require_order_ownership: true
  human_approval_over: 5000

order_cancellation:
  allowed_until: shipped

account_changes:
  require_otp: true

human_escalation:
  required_for: [fraud, legal, chargeback, high_value_refund]
```

- Policies are versioned, tested code. Every policy change runs the eval suite against simulated traffic before deploy (CI-style).
- Customers see a permissions dashboard: `✓ Read order status / ✓ Create ticket / ⚠ Refund ₹5,000+ (approval) / ✗ Access other customers / ✗ Execute arbitrary APIs`.

### Decision log (audit trail)

Every proposed action is logged: who proposed it, what it was, whether it was allowed/denied/escalated, why, and the outcome. This is what regulated companies (fintech, insurance, healthcare, telecom, government) pay for.

### Tool gateway

Wraps the customer's real APIs with precondition checks, idempotency, timeouts, and rollback where possible. This is a distributed-systems problem, not a YAML file — plan implementation accordingly (see §9).

### Standalone opportunity

The Control Plane can sit in front of *any* AI agent, including competitors'. This lets us sell "governance over AI actions" to companies that already use another agent — we don't have to beat Sierra on voice to win.

## 8. Data Flywheel (Real Moat)

Every conversation trains the next one.

```
Customer conversation
        ↓
Resolved by AI or Human
        ↓
Logged to evaluation store
  [transcript, language, intent, policy decisions, human actions, outcome]
        ↓
Two feedback loops:
  ├─ Evaluation: does the eval set cover this intent / language pattern?
  └─ Fine-tuning: enough Hinglish examples for this intent?
```

### v1 deliverables

- **Conversation log:** every interaction with structured metadata.
- **Evaluation suite:** 1,000+ test cases over the top 20 support intents, in English and Hinglish, with ground-truth answers.
- **Regression gates:** every deployment runs the eval suite; resolution-rate regression blocks the deploy.
- **Published benchmarks:** quarterly publication (e.g. *"93.7% resolution on Hinglish refund queries"*) — both a quality guarantee and sales collateral.

A competitor can download the same model; they cannot download our labeled Hinglish support conversations with verified outcomes.

## 9. Build Order & Milestones

Phased so each stage de-risks the next.

1. **M0 — Benchmarks before product.** Two-week spike: 1,000 real conversations from one partner; pipeline ASR → RAG → small LLM → TTS; measure turn latency, resolution rate, hallucination rate, cost/conversation. Publish the table. If latency >2s or resolution <75%, revise the thesis before building further.
2. **M1 — Control Plane core.** Policy engine (YAML), decision log, permissions dashboard, tool gateway with preconditions/idempotency/timeouts. This is a standalone sellable product and the moat.
3. **M2 — Agent (English + Hinglish, chat first).** RAG + small LLM + guardrails, human handoff with full context serialization, per-resolved-conversation billing instrumentation.
4. **M3 — Voice.** Streaming ASR (VAD, partial hypotheses), CPU TTS, voice turn-taking, PSTN/SIP integration.
5. **M4 — Eval Suite as product.** Simulation, red teaming, injection tests, regression + language tests, quality scoring; publish quarterly benchmarks.
6. **M5 — Vertical expansion.** Second vertical + second language pair, driven by the data flywheel.

## 10. Pricing (per resolved conversation)

| Tier | Price per resolved conversation | Est. monthly volume | Est. monthly cost |
|---|---|---|---|
| Startup | ₹8–₹12 | ≤5,000 | ₹40k–₹60k |
| Growth | ₹6–₹10 | 5,000–25,000 | ₹30k–₹2.5L |
| Enterprise | ₹4–₹8 | 25,000+ | ₹1L–₹8L+ |
| On-prem license | Annual ₹5L–₹15L | Unlimited | Fixed |

**Core rule:** if the AI doesn't resolve a conversation (escalates to human), it's free. Customer pays only for automated resolutions.

- Zero-risk trial for the customer.
- Hard incentive for us to raise resolution rate.
- Pricing *is* the ROI story: "₹10/resolved vs ₹100/human agent."
- On-prem license: fixed cost, no data leaves their network.

## 11. Go-to-Market: Sell to a Different Buyer

| Buyer | Title | Pain | Value prop |
|---|---|---|---|
| **Primary** | CISO / Compliance Officer | Data sovereignty, audit trail, unauthorized actions | "Full decision log. Every AI action gated by policy and logged. Data never leaves your infrastructure." |
| **Secondary** | COO / VP Operations | Cost per interaction, resolution rate | "Pay only for resolved conversations. 60% cost reduction." |
| **Tertiary** | Head of Customer Support | Agent workload, Hinglish coverage, escalation quality | "Hinglish support that works. Full-context handoff." |

**Sales motion:** enter via CISO (governance/control) → COO becomes budget holder (ROI) → support head becomes champion (resolution numbers). This inverts the incumbent playbook (sell to ops first, hope compliance doesn't kill the deal).

## 12. How to Measure Success (v1)

- End-to-end turn latency ≤ 2s on target VPS.
- Resolution rate ≥ 75% (M0 spike) → ≥ 90% (post-flywheel) on eval suite.
- Wrong-action rate (unauthorized action actually executed) = 0.0%.
- Prompt-injection resistance ≥ 99% on eval suite.
- Cost per resolved conversation ≤ ₹12 delivered.
- Hinglish eval accuracy ≥ 95% on the published eval set.

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CPU latency too high for voice | Streaming ASR partials, turn-taking design, async tools, speculative decoding; benchmark before committing (M0) |
| Hinglish harder than expected on small models | Dedicated Hinglish eval set; data flywheel fine-tuning; start chat-first to gather data cheaply |
| Policy engine is a distributed-systems problem | Tool gateway with idempotency/timeouts/rollback treated as first-class; build Control Plane first (M1) |
| Incumbents copy "per-conversation" pricing | Their SaaS cost structure can't match CPU on-prem; data flywheel compounds |
| Open-source stack replication | Moat is data + eval + policy + workflows + deployment + observability, not the model |

## 14. Non-Goals (v1)

- No 12-language support. English + Hindi + Hinglish only.
- No GPU targets.
- No token-based pricing.
- No competing on voice quality with PolyAI-class vendors.
- No generic-chatbot demos; the demo is the policy-enforced bank/ecommerce flow.
