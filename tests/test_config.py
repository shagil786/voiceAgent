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
