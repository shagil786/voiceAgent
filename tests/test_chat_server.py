# tests/test_chat_server.py
"""Demo-server hardening: the shared rate limiter (per-IP fixed window +
env wiring) and the live HTTP behavior (429 + Retry-After on /api/*) of the
real server in scripts/chat_server.py."""
import importlib.util
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from voiceagent.chat_server import build_html, rate_limiter_from_env

# Load the REAL demo server script (module-style); its Handler class wires
# the rate limiter into do_GET/do_POST.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chat_server.py"
_spec = importlib.util.spec_from_file_location("demo_chat_server", _SCRIPT)
demo_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_server)


def test_demo_server_import_does_not_leak_dotenv_into_process_env(monkeypatch):
    # Regression: scripts/chat_server.py loaded .env at module level, so
    # exec'ing it (as above) injected VOICEAGENT_TTS_VOICES into os.environ
    # and rerouted every TTS voice suite-wide. Entry-point loading only.
    monkeypatch.delenv("VOICEAGENT_TTS_VOICES", raising=False)
    probe_spec = importlib.util.spec_from_file_location(
        "demo_chat_server_probe", _SCRIPT)
    assert probe_spec is not None and probe_spec.loader is not None
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    assert "VOICEAGENT_TTS_VOICES" not in os.environ


def test_build_html_has_form_and_endpoint():
    html = build_html()
    assert "textarea" in html
    assert "/api/turn" in html
    assert "fetch" in html
    # The inline JS string escapes must survive as backslash-n (raw PAGE): a
    # non-raw string interpolates them into literal newlines inside JS string
    # literals -> SyntaxError -> go() undefined -> Send button dead.
    assert "\\n[action]" in html
    script = html.split("<script>", 1)[1]
    # No string literal in the served JS may contain a raw newline.
    for quote in ("'", '"'):
        for chunk in script.split(quote)[1::2]:
            assert "\n" not in chunk, f"raw newline inside JS string: {chunk!r}"


# --- RateLimiter unit behavior ------------------------------------------------

def test_rate_limiter_allows_within_budget_then_blocks():
    rl = demo_server.RateLimiter(2, 60.0)
    t0 = 1000.0
    assert rl.allow("ip1", now=t0)
    assert rl.allow("ip1", now=t0 + 1)
    assert not rl.allow("ip1", now=t0 + 2)  # over budget for the window
    assert rl.allow("ip2", now=t0 + 2)      # other keys are independent


def test_rate_limiter_window_expiry_frees_budget():
    rl = demo_server.RateLimiter(1, 60.0)
    t0 = 1000.0
    assert rl.allow("ip1", now=t0)
    assert not rl.allow("ip1", now=t0 + 30)  # still in window
    assert rl.allow("ip1", now=t0 + 60.5)    # first hit aged out


def test_rate_limiter_retry_after_reports_window_reset():
    rl = demo_server.RateLimiter(1, 60.0)
    t0 = 1000.0
    rl.allow("ip1", now=t0)
    assert rl.retry_after("ip1", now=t0 + 10) == 51
    assert rl.retry_after("never-seen", now=t0) == 1


# --- env wiring ---------------------------------------------------------------

def test_rate_limiter_from_env_unset_disables():
    assert rate_limiter_from_env({}) is None


def test_rate_limiter_from_env_builds_limiter():
    rl = rate_limiter_from_env({"VOICEAGENT_HTTP_RATE_LIMIT": "30"})
    assert rl is not None
    assert rl.max_events == 30
    assert rl.window_s == 60.0


def test_rate_limiter_from_env_invalid_values_disable():
    assert rate_limiter_from_env(
        {"VOICEAGENT_HTTP_RATE_LIMIT": "abc"}) is None
    assert rate_limiter_from_env(
        {"VOICEAGENT_HTTP_RATE_LIMIT": "-5"}) is None
    assert rate_limiter_from_env(
        {"VOICEAGENT_HTTP_RATE_LIMIT": "10",
         "VOICEAGENT_HTTP_RATE_WINDOW_S": "0"}) is None


# --- client identity ----------------------------------------------------------

def test_client_ip_xff_only_when_trust_proxy():
    """X-Forwarded-For is client-supplied: trusting it by default lets an
    attacker rotate the header and bypass the limiter (measured 20/20
    through a 3/min limit). Default = socket peer; XFF only when the caller
    explicitly passes trust_proxy=True (deployments behind a trusted
    reverse proxy set it via VOICEAGENT_TRUST_PROXY)."""

    class H:
        headers = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
        client_address = ("127.0.0.1", 55555)

    # default: XFF ignored — the socket peer is the identity
    assert demo_server._client_ip(H()) == "127.0.0.1"
    # opt-in trust: leftmost XFF hop wins
    assert demo_server._client_ip(H(), trust_proxy=True) == "203.0.113.7"


def test_rate_limiter_default_distrusts_xff_and_env_opts_in():
    rl = demo_server.rate_limiter_from_env({"VOICEAGENT_HTTP_RATE_LIMIT": "30"})
    assert rl is not None and rl.trust_proxy is False
    rl2 = demo_server.rate_limiter_from_env({
        "VOICEAGENT_HTTP_RATE_LIMIT": "30",
        "VOICEAGENT_TRUST_PROXY": "true"})
    assert rl2 is not None and rl2.trust_proxy is True




def test_client_ip_falls_back_to_socket_peer():
    class H:
        headers = {}
        client_address = ("127.0.0.1", 55555)
    assert demo_server._client_ip(H()) == "127.0.0.1"


# --- live HTTP behavior of scripts/chat_server.py -----------------------------

def test_api_turn_rate_limited_after_budget(monkeypatch):
    monkeypatch.setattr(demo_server, "MEMORY", None)
    monkeypatch.setattr(demo_server, "RATE_LIMITER",
                        demo_server.RateLimiter(1, 60.0))
    monkeypatch.setattr(demo_server, "run_turn",
                        lambda orch, text, **kw: {"reply": "ok"})
    srv = HTTPServer(("127.0.0.1", 0), demo_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/api/turn"
        req = urllib.request.Request(
            url, data=json.dumps({"text": "hi"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        # Second POST from the same client exceeds the budget -> 429.
        try:
            urllib.request.urlopen(req)
            got_429 = False
        except urllib.error.HTTPError as exc:
            got_429 = True
            assert exc.code == 429
            assert int(exc.headers["Retry-After"]) >= 1
            assert json.loads(exc.read())["error"] == "rate limit exceeded"
        assert got_429
        # The shared per-IP budget also gates /api/history.
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{srv.server_port}/api/history")
            got_429_history = False
        except urllib.error.HTTPError as exc:
            got_429_history = exc.code == 429
        assert got_429_history
    finally:
        srv.shutdown()
        srv.server_close()


def test_static_page_is_not_rate_limited(monkeypatch):
    monkeypatch.setattr(demo_server, "RATE_LIMITER",
                        demo_server.RateLimiter(0, 60.0))  # nothing allowed
    srv = HTTPServer(("127.0.0.1", 0), demo_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{srv.server_port}/") as resp:
            assert resp.status == 200
            assert b"VoiceAgent" in resp.read()
    finally:
        srv.shutdown()
        srv.server_close()