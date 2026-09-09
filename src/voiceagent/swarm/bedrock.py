"""Bedrock Converse provider — the swarm's optional AWS fallback brain.

Maps the bridge's OpenAI-style chat contract onto Amazon Bedrock's Converse
API so DeepSeek / GLM (or any Converse-served model) can back the agent when
the primary OpenAI-compatible frontier rate-limits. stdlib only — no boto3.

Auth is AWS SigV4 (not a bearer key): the signer below is dependency-free and
unit-tested against the published AWS SigV4 example vector.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import quote

from voiceagent.swarm.frontier import (
    FrontierError,
    FrontierReply,
    FrontierToolCall,
)

SERVICE = "bedrock"
RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass
class BedrockConfig:
    region: str
    model_id: str
    access_key: str
    secret_key: str
    session_token: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_base_delay_s: float = 0.4


def config_from_env(env: Mapping[str, str] | None = None) -> BedrockConfig | None:
    """Build the Bedrock provider from AWS standard env vars + the model id.
    Returns None when any required field is missing (fallback stays off)."""
    e = os.environ if env is None else env
    access_key = e.get("AWS_ACCESS_KEY_ID")
    secret_key = e.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        return None
    region = (e.get("VOICEAGENT_BEDROCK_REGION")
              or e.get("AWS_REGION") or e.get("AWS_DEFAULT_REGION"))
    model_id = e.get("VOICEAGENT_BEDROCK_MODEL_ID")
    if not region or not model_id:
        return None
    return BedrockConfig(
        region=region, model_id=model_id,
        access_key=access_key, secret_key=secret_key,
        session_token=e.get("AWS_SESSION_TOKEN"),
    )


# --- SigV4 (dependency-free) ----------------------------------------------

def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sigv4_sign(access_key: str, secret_key: str, region: str, service: str,
               method: str, path: str, query: str, headers_to_sign: dict,
               payload_hash: str, amzdate: str, datestamp: str,
               ) -> tuple[str, str]:
    """Return (Authorization header value, signed-headers string) for one
    AWS SigV4 request. `headers_to_sign` maps lowercase header name -> value;
    names are sorted into the canonical form. Tested against the AWS example
    vector (iam, GET /, 20150830T123600Z)."""
    names = sorted(headers_to_sign)
    canonical_headers = "".join(
        f"{name}:{headers_to_sign[name].strip()}\n" for name in names)
    signed_headers = ";".join(names)
    canonical_request = "\n".join([
        method, path, query, canonical_headers, signed_headers, payload_hash,
    ])
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amzdate,
        scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])
    k_date = _hmac(f"AWS4{secret_key}".encode("utf-8"), datestamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()
    authorization = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                     f"SignedHeaders={signed_headers}, Signature={signature}")
    return authorization, signed_headers


# --- Converse <-> OpenAI message mapping ------------------------------------

def _to_converse(model_id: str, messages: list[dict],
                 tools: list[dict] | None, temperature: float,
                 max_tokens: int) -> dict:
    system_blocks: list[dict] = []
    converse_msgs: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system_blocks.append({"text": content or ""})
        elif role == "user":
            converse_msgs.append(
                {"role": "user", "content": [{"text": content or ""}]})
        elif role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"text": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                blocks.append({"toolUse": {
                    "toolUseId": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "input": args if isinstance(args, dict) else {},
                }})
            converse_msgs.append(
                {"role": "assistant", "content": blocks or [{"text": ""}]})
        elif role == "tool":
            converse_msgs.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": m.get("tool_call_id") or "",
                "content": [{"text": content or ""}],
            }}]})
    body: dict[str, Any] = {
        "modelId": model_id,
        "messages": converse_msgs,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system_blocks:
        body["system"] = system_blocks
    if tools:
        body["toolConfig"] = {"tools": [
            {"toolSpec": {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "inputSchema": {"json": t["function"].get("parameters")
                                or {"type": "object", "properties": {}}},
            }}
            for t in tools if t.get("type") == "function"
        ]}
    return body


def _from_converse(raw: dict, model_id: str) -> FrontierReply:
    msg = (raw.get("output") or {}).get("message", {})
    texts: list[str] = []
    tool_calls: list[FrontierToolCall] = []
    for block in msg.get("content") or []:
        if "text" in block:
            texts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(FrontierToolCall(
                id=tu.get("toolUseId") or "",
                name=tu.get("name") or "",
                arguments=tu.get("input") or {},
            ))
    return FrontierReply(
        content="\n".join(texts) if texts else None,
        tool_calls=tool_calls,
        model=model_id,
        latency_s=0.0,
        raw=raw,
    )


# --- transport + client -----------------------------------------------------

Transport = Callable[[str, bytes, dict, float], dict]


def _bedrock_transport(url: str, body: bytes, headers: dict,
                       timeout_s: float) -> dict:
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "voiceagent-bedrock/1.0", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


class BedrockConverseClient:
    """Converse backend exposing the same `.chat()` contract as FrontierClient,
    so the failover chain can interleave OpenAI-compatible and Bedrock
    providers transparently."""

    def __init__(self, config: BedrockConfig,
                 transport: Transport | None = None,
                 sleep_fn: Callable[[float], None] | None = None):
        self.config = config
        self._transport = transport or _bedrock_transport
        self._sleep = sleep_fn or time.sleep

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | dict = "auto", temperature: float = 0.4,
             max_tokens: int = 512) -> FrontierReply:
        cfg = self.config
        host = f"bedrock-runtime.{cfg.region}.amazonaws.com"
        path = f"/model/{quote(cfg.model_id, safe='')}/converse"
        body = json.dumps(_to_converse(cfg.model_id, messages, tools,
                                       temperature, max_tokens)).encode("utf-8")
        now = datetime.now(timezone.utc)
        amzdate = now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = now.strftime("%Y%m%d")
        to_sign = {
            "host": host,
            "x-amz-date": amzdate,
            "x-amz-content-sha256": _sha256_hex(body),
        }
        if cfg.session_token:
            to_sign["x-amz-security-token"] = cfg.session_token
        authorization, _ = sigv4_sign(
            cfg.access_key, cfg.secret_key, cfg.region, SERVICE,
            "POST", path, "", to_sign, _sha256_hex(body), amzdate, datestamp)
        headers = {**to_sign, "Authorization": authorization}
        url = f"https://{host}{path}"
        t0 = time.perf_counter()
        raw = self._request_with_retries(url, body, headers)
        reply = _from_converse(raw, cfg.model_id)
        reply.latency_s = time.perf_counter() - t0
        return reply

    def _request_with_retries(self, url: str, body: bytes,
                              headers: dict) -> dict:
        attempts = self.config.max_retries + 1
        delay = self.config.retry_base_delay_s
        for attempt in range(attempts):
            try:
                return self._transport(url, body, headers,
                                       self.config.timeout_s)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    OSError) as exc:
                retryable = (isinstance(exc, urllib.error.HTTPError)
                             and exc.code in RETRYABLE_HTTP_CODES) or (
                    isinstance(exc, (urllib.error.URLError, OSError)))
                if not retryable or attempt >= attempts - 1:
                    if isinstance(exc, urllib.error.HTTPError):
                        detail = ""
                        try:
                            detail = exc.read().decode("utf-8", "replace")[:400]
                        except Exception:
                            pass
                        raise FrontierError(
                            f"Bedrock HTTP {exc.code}: {detail}") from exc
                    raise FrontierError(
                        f"Bedrock unreachable: {exc}") from exc
                self._sleep(delay * (2 ** attempt))
        raise FrontierError("Bedrock request failed")  # pragma: no cover
