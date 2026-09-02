"""Closed-loop voice demo: TTS a Hinglish query -> ASR it -> agent -> TTS reply.
Usage: python scripts/voice_demo.py
Prints transcript, reply, policy decision, and end-to-end voice-turn latency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.tts import synthesize_to_wav
from voiceagent.voice_agent import voice_turn


def build_live_agent():
    from voiceagent.knowledge import load_docs, build_index
    from voiceagent.llm import list_available_models, load_llm
    from voiceagent.agent import build_agent
    from voiceagent.intent import IntentClassifier, INTENT_EXEMPLARS
    from voiceagent.policy import load_policies
    from voiceagent.decisionlog import DecisionLog
    from voiceagent.sentiment import SentimentStore
    from voiceagent.tenant import Tenant

    # M6b: the tenant BUNDLE is the composition root — identity, exemplars,
    # policies, and knowledge all come from the tenant's namespace, each
    # falling back to the platform default when the bundle omits it.
    tenant = Tenant.load("data/tenants/default")
    docs = load_docs(tenant.knowledge_dir() or "data/knowledge")
    index = build_index(docs)
    models = list_available_models("data/models")
    if not models:
        sys.exit("no models in data/models/ — run scripts/smoke_llm.py qwen2.5-0.5b-q4 first")
    m = next((x for x in models if x["name"] == "qwen2.5-0.5b-q4"), models[0])
    llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
    clf = IntentClassifier(exemplars=tenant.intent_exemplars())
    policy = load_policies(tenant.policy_file() or "data/policies/policies.yaml")
    log = DecisionLog()
    # M6b: the learnable frustration lexicon — novel expressions are
    # captured as candidates, reviewed ones promote into the live lexicon.
    sentiment = SentimentStore("data/out/sentiment.db")
    return build_agent(index, llm, classifier=clf, policy=policy,
                       decision_log=log, tenant=tenant.config,
                       sentiment_store=sentiment), log


def main():
    agent, log = build_live_agent()
    q = "Bhai mera order abhi tak nahi aaya, order id ORD-55671 hai"
    query_wav = "data/out/voice-demo/query.wav"
    reply_wav = "data/out/voice-demo/reply.wav"
    Path(query_wav).parent.mkdir(parents=True, exist_ok=True)

    print("TTS: speaking sample query ...")
    synthesize_to_wav(q, query_wav)

    print("Voice turn (ASR -> agent -> TTS):")
    res = voice_turn(agent, query_wav, out_audio=reply_wav)

    print(f"  transcript : {res.transcript}")
    print(f"  reply      : {res.reply}")
    print(f"  action     : {res.action}")
    print(f"  decision   : {res.decision}")
    print(f"  latency    : {res.latency_s:.2f}s")
    print(f"  reply audio: {res.tts_path}")
    print(f"  decisions  : {len(log.entries())} recorded in this session")


if __name__ == "__main__":
    main()
