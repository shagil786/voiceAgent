# tests/test_bedrock.py
"""Bedrock Converse provider: SigV4 signing (against the AWS example vector),
OpenAI<->Converse message/tool mapping, and failover integration."""
import json

import pytest

from voiceagent.swarm.bedrock import (
    BedrockConfig,
    BedrockConverseClient,
    _from_converse,
    _to_converse,
    sigv4_sign,
)
from voiceagent.swarm.frontier import (
    FailoverClient,
    FrontierClient,
    FrontierConfig,
    FrontierError,
)


# --- SigV4 correctness (AWS published example vector) ----------------------

def test_sigv4_matches_aws_example_vector():
    # https://docs.aws.amazon.com/IAM/latest/UserGuide/signature-v4-examples.html
    authorization, signed_headers = sigv4_sign(
        access_key="AKIDEXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1", service="iam",
        method="GET", path="/", query="Action=ListUsers&Version=2010-05-08",
        headers_to_sign={
            "content-type": "application/x-www-form-urlencoded; charset=utf-8",
            "host": "iam.amazonaws.com",
            "x-amz-date": "20150830T123600Z",
        },
        payload_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        amzdate="20150830T123600Z", datestamp="20150830",
    )
    assert signed_headers == "content-type;host;x-amz-date"
    assert "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7" in authorization


# --- OpenAI <-> Converse mapping -------------------------------------------

def test_to_converse_maps_roles_and_tool_calls():
    messages = [
        {"role": "system", "content": "You are a voice agent."},
        {"role": "user", "content": "book a slot"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "function": {"name": "book_appointment",
                                      "arguments": '{"day":"mon"}'}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": "done"},
    ]
    tools = [{"type": "function", "function": {
        "name": "book_appointment", "description": "Book it",
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}},
    }}]
    body = _to_converse("zai.glm-4.7-flash", messages, tools, 0.4, 512)
    assert body["modelId"] == "zai.glm-4.7-flash"
    assert body["system"] == [{"text": "You are a voice agent."}]
    assert body["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.4}
    msgs = body["messages"]
    assert msgs[0] == {"role": "user", "content": [{"text": "book a slot"}]}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["toolUse"]["name"] == "book_appointment"
    assert msgs[1]["content"][0]["toolUse"]["input"] == {"day": "mon"}
    assert msgs[2] == {"role": "user", "content": [{"toolResult": {
        "toolUseId": "t1", "content": [{"text": "done"}]}}]}
    assert body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"][
        "properties"]["day"]["type"] == "string"


def test_from_converse_maps_text_and_tool_use():
    raw = {"output": {"message": {"role": "assistant", "content": [
        {"text": "Sure, booking now."},
        {"toolUse": {"toolUseId": "t9", "name": "book_appointment",
                     "input": {"day": "mon"}}},
    ]}}, "stopReason": "tool_use"}
    reply = _from_converse(raw, "zai.glm-4.7-flash")
    assert reply.content == "Sure, booking now."
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "book_appointment"
    assert reply.tool_calls[0].arguments == {"day": "mon"}


# --- provider failover across OpenAI + Bedrock -----------------------------

def test_failover_openai_then_bedrock():
    primary = FrontierConfig(base_url="https://primary.example/v1",
                             model="m1", api_key="***", max_retries=0)
    bedrock = BedrockConfig(region="us-west-2", model_id="zai.glm-4.7-flash",
                            access_key="AKID", secret_key="SECRET")

    def openai_t(url, payload, headers, timeout_s):
        raise FrontierError("HTTP 429")

    def bedrock_t(url, body, headers, timeout_s):
        assert "/model/zai.glm-4.7-flash/converse" in url
        return {"output": {"message": {"content": [{"text": "bedrock says hi"}]}}}

    chain = FailoverClient([
        FrontierClient(primary, transport=openai_t),
        BedrockConverseClient(bedrock, transport=bedrock_t),
    ])
    reply = chain.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "bedrock says hi"
    assert reply.model == "zai.glm-4.7-flash"


def test_config_from_env_requires_all_fields():
    from voiceagent.swarm.bedrock import config_from_env
    assert config_from_env({}) is None
    full = {
        "AWS_ACCESS_KEY_ID": "AKID", "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "ap-south-1", "VOICEAGENT_BEDROCK_MODEL_ID": "zai.glm-4.7-flash",
    }
    cfg = config_from_env(full)
    assert cfg is not None and cfg.model_id == "zai.glm-4.7-flash"
    assert cfg.region == "ap-south-1"


def test_config_from_env_bearer_mode():
    from voiceagent.swarm.bedrock import config_from_env
    cfg = config_from_env({
        "VOICEAGENT_BEDROCK_API_KEY": "ABSKxyz",
        "VOICEAGENT_BEDROCK_MODEL_ID": "zai.glm-4.7-flash",
        "VOICEAGENT_BEDROCK_REGION": "ap-south-1",
    })
    assert cfg is not None
    assert cfg.api_key == "ABSKxyz"
    assert cfg.region == "ap-south-1"
    # no IAM creds needed when the API key is present
    assert config_from_env({"VOICEAGENT_BEDROCK_MODEL_ID": "m"}) is None
