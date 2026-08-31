"""VoiceAgent CLI REPL — the live demo. Type a (Hinglish) support query,
see the reply, the action, and the policy decision with reasons."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.dataset import load_conversations
from voiceagent.knowledge import load_docs, build_index
from voiceagent.llm import list_available_models, load_llm
from voiceagent.agent import build_agent
from voiceagent.intent import IntentClassifier
from voiceagent.policy import load_policies
from voiceagent.decisionlog import DecisionLog
from voiceagent.chat import run_turn


def build_live_agent():
    docs = load_docs("data/knowledge")
    index = build_index(docs)
    models = list_available_models("data/models")
    if not models:
        sys.exit("no models in data/models/ — run scripts/smoke_llm.py qwen2.5-0.5b-q4 first")
    # Prefer the fast 0.5B
    m = next((x for x in models if x["name"] == "qwen2.5-0.5b-q4"), models[0])
    llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
    clf = IntentClassifier()
    policy = load_policies("data/policies/policies.yaml")
    log = DecisionLog()
    return build_agent(index, llm, classifier=clf, policy=policy, decision_log=log), log


if __name__ == "__main__":
    agent, log = build_live_agent()
    print("VoiceAgent CLI — type a support query (Ctrl-D to exit)")
    print("  authenticated=on turns on auth for this query\n")
    i = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        auth = False
        if line.startswith("authenticated=on "):
            auth = True
            line = line[len("authenticated=on "):]
        conv_id = f"demo-{i}"
        out = run_turn(agent, line, authenticated=auth, conv_id=conv_id)
        i += 1
        print(f"\n[agent] {out['reply']}")
        print(f"[action] {out['action'] or 'none'}  [policy] {out['decision'] or 'n/a'}")
        for r in out["reasons"]:
            print(f"   · {r}")
        print()
    print(f"\n{len(log.entries())} decisions recorded in this session.")
