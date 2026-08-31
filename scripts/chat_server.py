"""VoiceAgent demo HTTP server (stdlib only).
Usage: python scripts/chat_server.py [port]   (default 8000)
Open http://127.0.0.1:8000 in a browser.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.chat import run_turn
from voiceagent.chat_server import build_html

AGENT = None


def _build_live_agent():
    """Load the full pipeline (index, classifier, policy, log, fast LLM)."""
    from voiceagent.knowledge import load_docs, build_index
    from voiceagent.llm import list_available_models, load_llm
    from voiceagent.agent import build_agent
    from voiceagent.intent import IntentClassifier
    from voiceagent.policy import load_policies
    from voiceagent.decisionlog import DecisionLog

    docs = load_docs("data/knowledge")
    index = build_index(docs)
    models = list_available_models("data/models")
    if not models:
        sys.exit("no models in data/models/ — run scripts/smoke_llm.py qwen2.5-0.5b-q4 first")
    m = next((x for x in models if x["name"] == "qwen2.5-0.5b-q4"), models[0])
    llm = load_llm(m["model_path"], params=m["params"], size_mb=m["size_mb"])
    clf = IntentClassifier()
    policy = load_policies("data/policies/policies.yaml")
    log = DecisionLog()
    return build_agent(index, llm, classifier=clf, policy=policy, decision_log=log)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        if not text:
            self._json({"error": "empty text"}, 400)
            return
        out = run_turn(AGENT, text, authenticated=auth, conv_id="demo-http")
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
    AGENT = _build_live_agent()
    print(f"VoiceAgent demo at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
