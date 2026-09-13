# tests/test_fixture_backend.py — the data-only GenericBackend.
"""Platform code with zero domain nouns: ANY tenant is demoable from a
JSON fixture. Guarded: protocol conformance, fail-closed errors, the
create/patch operation effects, lifecycle introspection."""
import json

import pytest

from voiceagent.fixture_backend import FixtureGenericBackend
from voiceagent.generic_backend import GenericBackendError

FIXTURE = {
    "resources": {
        "gadget": {"G-1": {"gadget_id": "G-1", "kind": "basic",
                           "owner": "+15550001"},
                   "G-2": {"gadget_id": "G-2", "kind": "pro",
                           "owner": "+15550002"}},
        "order": {},
    },
    "operations": {
        "searchGadgets": {"response": {"available": ["G-1", "G-2"]}},
        "createOrder": {"create": {"resource": "order", "id_prefix": "O-",
                                   "id_key": "order_id"},
                        "response": {"status": "PLACED"}},
        "cancelOrder": {"patch": {"resource": "order",
                                  "id_from": "order_id",
                                  "set": {"status": "CANCELLED"}},
                        "response": {}},
    },
    "lifecycles": {"order": ["PLACED", "CANCELLED"]},
}


@pytest.fixture()
def be(tmp_path):
    f = tmp_path / "fx.json"
    f.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return FixtureGenericBackend(f)


def test_get_resource_deep_copies(be):
    rec = be.get_resource("gadget", "G-1")
    rec["kind"] = "MUTATED"
    assert be.get_resource("gadget", "G-1")["kind"] == "basic"


def test_get_resource_missing_is_none(be):
    assert be.get_resource("gadget", "G-999") is None


def test_unknown_resource_type_fails_closed(be):
    with pytest.raises(GenericBackendError, match="resource_type"):
        be.get_resource("spacecraft", "X-1")


def test_list_requires_filter_and_exact_matches(be):
    with pytest.raises(GenericBackendError, match="filter"):
        be.list_resources("gadget")
    out = be.list_resources("gadget", {"kind": "pro"})
    assert [r["gadget_id"] for r in out] == ["G-2"]


def test_execute_operation_canned_response(be):
    assert be.execute_operation("searchGadgets", {}) == \
        {"available": ["G-1", "G-2"]}


def test_execute_operation_unknown_fails_closed(be):
    with pytest.raises(GenericBackendError, match="unsupported operation"):
        be.execute_operation("launchRocket", {})


def test_create_effect_mints_record_and_merges_response(be):
    out = be.execute_operation("createOrder",
                               {"gadget_id": "G-1", "owner": "+15550001"})
    assert out["order_id"] == "O-1001"
    assert out["status"] == "PLACED"
    stored = be.get_resource("order", "O-1001")
    assert stored["gadget_id"] == "G-1"
    out2 = be.execute_operation("createOrder", {"gadget_id": "G-2"})
    assert out2["order_id"] == "O-1002"  # counter increments


def test_create_into_undeclared_resource_fails_closed(tmp_path):
    fixture = dict(FIXTURE)
    fixture["resources"] = {"gadget": {}}
    fixture["operations"] = {
        "createWidget": {"create": {"resource": "widget",
                                    "id_prefix": "W-",
                                    "id_key": "widget_id"},
                         "response": {}},
    }
    f = tmp_path / "fx-undeclared.json"
    f.write_text(json.dumps(fixture), encoding="utf-8")
    be = FixtureGenericBackend(f)
    with pytest.raises(GenericBackendError, match="resource_type"):
        be.execute_operation("createWidget", {"kind": "basic"})
    # No phantom resource map was minted: "widget" stays undeclared.
    with pytest.raises(GenericBackendError, match="resource_type"):
        be.get_resource("widget", "W-1001")


def test_patch_effect_mutates_and_returns_updated_record(be):
    be.execute_operation("createOrder", {"gadget_id": "G-1"})
    out = be.execute_operation("cancelOrder", {"order_id": "O-1001"})
    assert out["status"] == "CANCELLED"
    assert be.get_resource("order", "O-1001")["status"] == "CANCELLED"


def test_patch_unknown_id_fails_closed(be):
    with pytest.raises(GenericBackendError, match="order_not_found"):
        be.execute_operation("cancelOrder", {"order_id": "O-404"})


def test_lifecycle_introspection(be):
    assert be.get_lifecycle_states("order") == ["PLACED", "CANCELLED"]
    with pytest.raises(NotImplementedError):
        be.get_lifecycle_states("gadget")


def test_protocol_conformance(be):
    from voiceagent.generic_backend import GenericBackend
    assert isinstance(be, GenericBackend)


def test_missing_or_invalid_file_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        FixtureGenericBackend(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        FixtureGenericBackend(bad)
