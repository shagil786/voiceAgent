# tests/test_import_contract.py — the CI-assertable architecture boundary
# (service split, 2026-09-14): with service URLs declared, importing the
# brain + dispatchers loads ZERO ML libraries; with no URLs, importing the
# dispatchers alone is also ML-free (legacy modules load lazily at call
# time). Probes run in fresh subprocesses — the pytest session itself may
# have torch resident from other tiers.
import json
import subprocess
import sys

ML_MODULES = ["torch", "faiss", "sentence_transformers", "transformers",
              "faster_whisper", "piper", "librosa", "peft"]

REMOTE_PROBE = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.runtime            # the brain assembly seam
import voiceagent.asr_client, voiceagent.tts_client
import voiceagent.voice_agent        # the voice-loop entry
bad = [m for m in {ml} if m in sys.modules]
print(json.dumps({{"bad": bad}}))
""".format(ml=json.dumps(ML_MODULES))

LEGACY_PROBE = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.asr_client, voiceagent.tts_client   # dispatchers only
print(json.dumps({{"ml_at_import": [m for m in {ml} if m in sys.modules]}}))
""".format(ml=json.dumps(ML_MODULES))


def _run(probe: str, env_extra: dict) -> dict:
    import os
    env = dict(os.environ)
    env.pop("VOICEAGENT_ASR_URL", None)
    env.pop("VOICEAGENT_TTS_URL", None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_brain_imports_zero_ml_with_services_declared():
    out = _run(REMOTE_PROBE, {"VOICEAGENT_ASR_URL": "ws://127.0.0.1:8710/ws",
                              "VOICEAGENT_TTS_URL": "ws://127.0.0.1:8711/ws"})
    assert out["bad"] == [], "brain loaded ML modules: %s" % out["bad"]


def test_dispatchers_import_clean_without_urls():
    out = _run(LEGACY_PROBE, {})
    assert out["ml_at_import"] == [], out["ml_at_import"]
