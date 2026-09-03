"""VoiceAgent demo HTTP server (stdlib only).
Usage: python scripts/chat_server.py [port]   (default 8000)
Open http://127.0.0.1:8000 in a browser.
"""
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.chat import run_turn
from voiceagent.chat_server import build_html
from voiceagent.memory import SQLiteMemory, public_dict

AGENT = None
MEMORY: SQLiteMemory | None = None
DEFAULT_CONV_ID = "demo-http"


def _pick_local_model(models: list[dict]) -> dict:
    """VOICEAGENT_MODEL (candidate name or exact filename in data/models) >
    legacy hard-pick qwen2.5-0.5b-q4 > first available."""
    want = os.environ.get("VOICEAGENT_MODEL")
    if want:
        m = next((x for x in models
                  if x["name"] == want
                  or Path(x["model_path"]).name == want), None)
        if m is None:
            sys.exit(f"VOICEAGENT_MODEL={want!r} matches nothing in "
                     f"data/models/ (available: {[x['name'] for x in models]})")
        return m
    # Prefer the fine-tuned Hinglish model, fall back to the base model.
    return next((x for x in models if x["name"] == "qwen2.5-0.5b-hinglish-q4"),
                next((x for x in models if x["name"] == "qwen2.5-0.5b-q4"),
                     models[0]))


def _build_live_agent():
    """Load the full pipeline (index, classifier, policy, log, fast LLM).

    LLM selection: VOICEAGENT_LLM_BASE_URL + VOICEAGENT_LLM_MODEL serve via
    any OpenAI-compatible endpoint (no local GGUF needed); otherwise a local
    GGUF from data/models is picked via _pick_local_model()."""
    from voiceagent.knowledge import load_docs, build_index
    from voiceagent.llm import (build_llm_from_env, list_available_models,
                                load_llm)
    from voiceagent.agent import build_agent
    from voiceagent.intent import IntentClassifier
    from voiceagent.policy import load_policies
    from voiceagent.decisionlog import DecisionLog

    docs = load_docs("data/knowledge")
    index = build_index(docs)
    remote = build_llm_from_env()
    if remote is not None:
        llm = remote
        serving = f"{remote.model} @ {remote.base_url} (OpenAI-compatible API)"
    else:
        models = list_available_models("data/models")
        if not models:
            sys.exit("no models in data/models/ — run scripts/smoke_llm.py "
                     "qwen2.5-0.5b-q4 first, or set VOICEAGENT_LLM_BASE_URL + "
                     "VOICEAGENT_LLM_MODEL to serve via an OpenAI-compatible API")
        m = _pick_local_model(models)
        llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
        serving = m["model_path"]
    print(f"serving model: {serving}")
    clf = IntentClassifier()
    policy = load_policies("data/policies/policies.yaml")
    log = DecisionLog()
    from voiceagent.tools import MockERP, ToolGateway, GovernedToolRunner
    erp = MockERP()
    gateway = ToolGateway(erp=erp)
    runner = GovernedToolRunner(gateway, policy, decision_log=log)
    return build_agent(index, llm, classifier=clf, policy=policy, decision_log=log,
                       tool_runner=runner, erp=erp)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            body = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/history":
            params = urllib.parse.parse_qs(query)
            conv_id = params.get("conv_id", [DEFAULT_CONV_ID])[0]
            turns = MEMORY.history(conv_id) if MEMORY is not None else []
            self._json([public_dict(t) for t in turns])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/turn":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        text = payload.get("text", "")
        auth = bool(payload.get("authenticated", False))
        conv_id = str(payload.get("conv_id", DEFAULT_CONV_ID))
        if not text:
            self._json({"error": "empty text"}, 400)
            return
        out = run_turn(AGENT, text, authenticated=auth, conv_id=conv_id,
                       memory=MEMORY)
        self._json(out)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the console clean


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    Path("data/out").mkdir(parents=True, exist_ok=True)
    MEMORY = SQLiteMemory("data/out/memory.db")
    AGENT = _build_live_agent()
    print(f"VoiceAgent demo at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
