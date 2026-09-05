# tests/test_config.py
from voiceagent.config import RuntimeConfig, load_config

def test_defaults_reproduce_today():
    c = load_config(env={})
    assert c.models_dir == "data/models"
    assert "qwen3-0.6b-q4" in c.candidate_models
    assert set(c.voices) >= {"en", "hi"}

def test_env_over_tenant_over_default():
    from voiceagent.tenant import TenantConfig
    t = TenantConfig.load("data/tenants/example-acme/tenant.json")
    c = load_config(env={"VOICEAGENT_MODELS_DIR": "/m",
                         "VOICEAGENT_CANDIDATE_MODELS": "a,b",
                         "VOICEAGENT_VOICES": "hi:/v/hi.onnx",
                         "VOICEAGENT_HF_TOKEN": "hf_x"}, tenant=t)
    assert (c.models_dir, c.candidate_models) == ("/m", ["a", "b"])
    assert c.voices["hi"] == "/v/hi.onnx" and c.hf_token == "hf_x"

def test_empty_candidate_models_falls_back_to_defaults():
    from voiceagent.config import default_candidate_models
    c = load_config(env={"VOICEAGENT_CANDIDATE_MODELS": "  "})
    assert c.candidate_models == default_candidate_models()

def test_tenant_mapping_voices_env_wins_per_key():
    tenant = {"voices": {"hi": "/t/hi.onnx"}}
    c = load_config(env={}, tenant=tenant)
    assert c.voices["hi"] == "/t/hi.onnx"
    c2 = load_config(env={"VOICEAGENT_VOICES": "hi:/v/hi.onnx"}, tenant=tenant)
    assert c2.voices["hi"] == "/v/hi.onnx"

def test_secrets_not_in_repr():
    c = load_config(env={"VOICEAGENT_HF_TOKEN": "hf_x",
                          "VOICEAGENT_FRONTIER_KEY": "sk-abc"})
    assert "hf_x" not in repr(c)
    assert "sk-abc" not in repr(c)


def test_livekit_credentials_accept_both_spellings():
    # LiveKit's console convention is LIVEKIT_API_KEY/SECRET; the limb plan
    # originally used LIVEKIT_KEY/SECRET. A correctly-named .env must never
    # silently yield None creds (that fail-closes the webhook validator and
    # the limb never answers a call).
    from voiceagent.config import load_config
    api = load_config(env={"LIVEKIT_URL": "wss://x",
                           "LIVEKIT_API_KEY": "k1",
                           "LIVEKIT_API_SECRET": "s1"})
    assert api.livekit_key == "k1" and api.livekit_secret == "s1"
    legacy = load_config(env={"LIVEKIT_KEY": "k2", "LIVEKIT_SECRET": "s2"})
    assert legacy.livekit_key == "k2" and legacy.livekit_secret == "s2"
    # API_ spelling wins when both are present (deterministic precedence).
    both = load_config(env={"LIVEKIT_API_KEY": "a", "LIVEKIT_KEY": "b",
                            "LIVEKIT_API_SECRET": "c", "LIVEKIT_SECRET": "d"})
    assert both.livekit_key == "a" and both.livekit_secret == "c"
