# tests/test_service_protocol.py — wire codec: JSON header + one optional
# binary frame per message, version-checked, req_id-correlated.
import json

import pytest

from voiceagent.service_protocol import (
    PROTOCOL_VERSION, FrameReader, ProtocolError, error_body, json_text,
    require)


def test_text_message_is_self_complete():
    r = FrameReader()
    msg = r.feed(json_text({"v": 1, "op": "describe"}))
    assert msg.op == "describe" and msg.req_id is None and msg.payload is None


def test_declared_payload_completes_on_binary_frame():
    r = FrameReader()
    assert r.feed(json_text({"v": 1, "op": "transcribe", "req_id": 7,
                             "language": "en", "sample_rate": 16000,
                             "channels": 1, "sampwidth": 2,
                             "has_payload": True})) is None
    msg = r.feed(b"\x01\x02\x03\x04")
    assert msg.op == "transcribe" and msg.req_id == 7
    assert msg.payload == b"\x01\x02\x03\x04"
    assert msg.fields["sample_rate"] == 16000


def test_binary_after_self_complete_header_is_error():
    r = FrameReader()
    r.feed(json_text({"v": 1, "op": "describe"}))
    with pytest.raises(ProtocolError):
        r.feed(b"a")


def test_binary_before_header_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(b"a")


def test_wrong_version_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(json.dumps({"v": 99, "op": "describe"}))


def test_missing_op_rejected():
    with pytest.raises(ProtocolError):
        FrameReader().feed(json.dumps({"v": PROTOCOL_VERSION}))


def test_require_and_error_body():
    with pytest.raises(ProtocolError, match="language"):
        require({"op": "transcribe"}, "language")
    err = error_body(5, "model_load_failed", "boom")
    assert err == {"v": PROTOCOL_VERSION, "op": "error", "req_id": 5,
                   "code": "model_load_failed", "message": "boom"}
