# tests/test_chat_server.py
from voiceagent.chat_server import build_html

def test_build_html_has_form_and_endpoint():
    html = build_html()
    assert "textarea" in html
    assert "/api/turn" in html
    assert "fetch" in html