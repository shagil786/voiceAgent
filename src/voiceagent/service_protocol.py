"""Wyoming-style narrow-waist wire protocol for VoiceAgent capability
services (spec 2026-09-14-service-split-design.md).

v1 framing: every message is one JSON text frame; a message MAY carry
exactly one binary frame, announced by `"has_payload": true` in its header
(raw PCM for transcribe requests, WAV bytes for synthesize responses).
Payload-less messages are self-complete the moment their header arrives —
no close-time flush races, and both directions agree via the explicit
flag. Audio shape is declared in the JSON header
(sample_rate/channels/sampwidth) — never assumed. Unknown version or op is
a protocol error: close the connection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """A frame violated the v1 wire contract."""


@dataclass
class Message:
    op: str
    req_id: int | None
    fields: dict
    payload: bytes | None


def json_text(fields: dict) -> str:
    body = {"v": PROTOCOL_VERSION, **fields}
    return json.dumps(body)


def require(fields: dict, key: str) -> None:
    if key not in fields:
        raise ProtocolError(f"missing required field '{key}'")


def error_body(req_id: int | None, code: str, message: str) -> dict:
    return {"v": PROTOCOL_VERSION, "op": "error", "req_id": req_id,
            "code": code, "message": message}


class FrameReader:
    """Assembles (JSON header, optional single binary payload) messages.

    A header WITHOUT has_payload is self-complete: feed() returns its
    Message immediately. A header WITH has_payload stashes until the next
    binary frame completes it. A binary frame with no stashed header, or a
    new header while a payload is still pending, is a ProtocolError."""

    def __init__(self):
        self._header: dict | None = None

    def feed(self, msg: str | bytes) -> Message | None:
        if isinstance(msg, (bytes, bytearray, memoryview)):
            if self._header is None:
                raise ProtocolError("binary frame before JSON header")
            header, self._header = self._header, None
            return Message(op=header["op"], req_id=header.get("req_id"),
                           fields=header, payload=bytes(msg))
        try:
            header = json.loads(msg)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"invalid JSON header: {e}") from e
        if not isinstance(header, dict) or header.get("v") != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version in {header!r}")
        if "op" not in header:
            raise ProtocolError("header missing 'op'")
        if header.get("has_payload"):
            self._header = header
            return None
        if self._header is not None:
            raise ProtocolError("new header while a binary payload is "
                                "still pending")
        return Message(op=header["op"], req_id=header.get("req_id"),
                       fields=header, payload=None)
