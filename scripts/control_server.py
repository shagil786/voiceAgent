#!/usr/bin/env python3
"""Agent-side control API server — the seam the voiceagent-console repo talks to.

Usage: python scripts/control_server.py [port] [host]   (default 8081, 127.0.0.1)
Requires VOICEAGENT_CONTROL_TOKEN (bearer auth, fail-closed) and optionally
VOICEAGENT_AUDIT_DB / VOICEAGENT_MEMORY_DB / VOICEAGENT_DEPLOY_ROOT.

Endpoints: GET /api/control/{status,calls,ratings,summary},
POST /api/control/onboard/{preview,deploy}. See voiceagent.control.
"""
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from voiceagent.dotenv import load_dotenv  # single source; call only inside main(), never at module level
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from voiceagent.control import server_from_env  # noqa: E402

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    cls = server_from_env()
    if not cls.token:
        print("refusing to start: VOICEAGENT_CONTROL_TOKEN unset "
              "(control plane is bearer-auth, fail-closed)", file=sys.stderr)
        raise SystemExit(2)
    httpd = ThreadingHTTPServer((host, port), cls)
    print(f"control API listening on {host}:{port}")
    httpd.serve_forever()
