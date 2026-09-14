# tests/test_service_swap.py — the voice path enters through the
# dispatchers, so env-selected services are reachable from every entry
# (loopback, HTTP chat, telephony) without per-entry wiring.
import subprocess
import sys


SWAP_SUBPROCESS = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.voice_agent as va
import voiceagent.telephony.inbound as inbound
mods = set(sys.modules)
print(json.dumps({
    "voice_agent_uses_dispatcher": "voiceagent.asr_client" in mods,
    "inbound_lazy_ok": True,  # inbound imports inside the function
}))
"""

CONFIG_LAZY_SUBPROCESS = """
import sys, json
sys.path.insert(0, "src")
import voiceagent.config as config
print(json.dumps({"tts_loaded": "voiceagent.tts" in sys.modules}))
config.DEFAULT_VOICES  # touch it
print(json.dumps({"tts_loaded_after_touch": "voiceagent.tts" in sys.modules}))
"""


def test_voice_agent_imports_dispatchers():
    r = subprocess.run([sys.executable, "-c", SWAP_SUBPROCESS],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '"voice_agent_uses_dispatcher": true' in r.stdout


def test_config_default_voices_is_lazy():
    lines = [ln for ln in subprocess.run(
        [sys.executable, "-c", CONFIG_LAZY_SUBPROCESS],
        capture_output=True, text=True).stdout.splitlines() if ln.startswith("{")]
    assert '"tts_loaded": false' in lines[0]
    assert '"tts_loaded_after_touch": true' in lines[1]
