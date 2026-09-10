"""One .env loader for all scripts (voiceagent.dotenv): override honored,
missing file ok, never overrides real env, entry-point-only rule."""
import os


def test_override_points_at_missing_file(monkeypatch, tmp_path):
    from voiceagent.dotenv import load_dotenv
    env = tmp_path / ".env"
    env.write_text("FROM_FILE=1\nSHARED=from-file\n", encoding="utf-8")
    monkeypatch.delenv("FROM_FILE", raising=False)
    monkeypatch.setenv("SHARED", "from-env")
    monkeypatch.setenv("VOICEAGENT_DOTENV_PATH", str(tmp_path / "nope.env"))
    load_dotenv(env)
    assert "FROM_FILE" not in os.environ  # missing override: no fallback
    assert os.environ["SHARED"] == "from-env"


def test_default_path_loads_without_clobbering(monkeypatch, tmp_path):
    from voiceagent.dotenv import load_dotenv
    env = tmp_path / ".env"
    env.write_text('A=1\nB="two"\n# comment\n\nC=\'three\'\n', encoding="utf-8")
    monkeypatch.delenv("VOICEAGENT_DOTENV_PATH", raising=False)
    for k in ("A", "B", "C"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("B", "keep-me")
    load_dotenv(env)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "keep-me"  # real env always wins
    assert os.environ["C"] == "three"


def test_all_scripts_share_the_helper():
    # No script may carry its own copy again (the suite-wide leak came
    # from a divergent copy). Nine files, one implementation.
    import pathlib
    import voiceagent.dotenv as helper
    for script in ["chat_server", "livekit_worker", "chat_relay",
                   "control_server", "forget_caller", "live_conversation",
                   "livekit_dial", "local_call"]:
        text = (pathlib.Path("scripts") / f"{script}.py").read_text()
        assert "def load_dotenv" not in text, script
        assert "from voiceagent.dotenv import load_dotenv" in text, script
    assert helper.load_dotenv.__module__ == "voiceagent.dotenv"
