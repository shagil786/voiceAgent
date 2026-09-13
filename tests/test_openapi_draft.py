# tests/test_openapi_draft.py — the deterministic OpenAPI drafter.
"""Pure mechanics: verb→risk, operationId→name, schemas→params. The trust
chain has no LLM; every draft is status: proposed for human review."""
import pytest

from voiceagent.openapi_draft import DraftResult, parse_openapi

HOTEL_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Grand Hotel PMS", "version": "1.0.0"},
    "paths": {
        "/rooms/search": {
            "post": {
                "operationId": "searchRooms",
                "summary": "Search available rooms for a stay window.",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "check_in": {"type": "string"},
                            "check_out": {"type": "string"},
                            "guests": {"type": "integer"},
                        },
                        "required": ["check_in", "check_out"],
                    }}}}},
        },
        "/rooms/{room_id}": {
            "get": {"operationId": "getRoom",
                    "summary": "Fetch one room.",
                    "parameters": [{"name": "room_id", "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"}}]},
        },
        "/bookings": {
            "post": {
                "operationId": "createBooking",
                "summary": "Create a booking.",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "guest_name": {"type": "string"},
                            "room_id": {"type": "string"},
                            "check_in": {"type": "string"},
                            "guests": {"type": "integer"},
                        },
                        "required": ["guest_name", "room_id", "check_in"],
                    }}}}},
        },
        "/bookings/{booking_id}": {
            "get": {"operationId": "getBooking",
                    "summary": "Fetch a booking.",
                    "parameters": [{"name": "booking_id", "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"}}]},
            "patch": {"operationId": "modifyBooking",
                      "summary": "Move a booking's check-out date.",
                      "parameters": [{"name": "booking_id", "in": "path",
                                      "required": True,
                                      "schema": {"type": "string"}}],
                      "requestBody": {"required": True, "content": {
                          "application/json": {"schema": {
                              "type": "object",
                              "properties": {"new_check_out":
                                             {"type": "string"}},
                              "required": ["new_check_out"],
                          }}}}},
            "delete": {"operationId": "cancelBooking",
                       "summary": "Cancel a booking.",
                       "parameters": [{"name": "booking_id", "in": "path",
                                       "required": True,
                                       "schema": {"type": "string"}}]},
        },
    },
}


def by_name(result):
    return {op["tool_name"]: op for op in result.operations}


def test_hotel_happy_path_full_shape():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert set(ops) == {"search_rooms", "get_room", "create_booking",
                        "get_booking", "modify_booking", "cancel_booking"}


def test_post_search_drafts_as_read_with_no_side_effects():
    op = by_name(parse_openapi(HOTEL_SPEC))["search_rooms"]
    assert op["risk_class"] == "read"
    assert op["side_effects"] is False
    assert op["operation"] == "searchRooms"
    assert op["params"] == ["check_in", "check_out"]  # optional guests dropped
    assert op["description"] == "Search available rooms for a stay window."


def test_optional_request_body_property_dropped_and_noted():
    result = parse_openapi(HOTEL_SPEC)
    assert by_name(result)["create_booking"]["params"] == \
        ["guest_name", "room_id", "check_in"]
    assert any("guests" in n and "optional" in n for n in result.notes)


def test_fetch_routing_get_with_trailing_id():
    op = by_name(parse_openapi(HOTEL_SPEC))["get_room"]
    assert op["operation"] == "__fetch__"
    assert op["resource_type"] == "room"
    assert op["id_param"] == "room_id"
    assert op["risk_class"] == "read"


def test_resource_singularized_from_first_segment():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["get_booking"]["resource_type"] == "booking"
    assert ops["get_booking"]["operation"] == "__fetch__"


def test_patch_is_mutating_delete_is_high():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["modify_booking"]["risk_class"] == "mutating"
    assert ops["modify_booking"]["side_effects"] is True
    assert ops["modify_booking"]["params"] == ["booking_id", "new_check_out"]
    assert ops["cancel_booking"]["risk_class"] == "high"
    assert ops["cancel_booking"]["side_effects"] is True


def test_param_types_extracted_from_schemas():
    ops = by_name(parse_openapi(HOTEL_SPEC))
    assert ops["create_booking"]["param_types"] == \
        {"guest_name": "string", "room_id": "string", "check_in": "string"}
    assert ops["search_rooms"]["param_types"] == \
        {"check_in": "string", "check_out": "string"}


def test_swagger_2_rejected_with_clear_error():
    with pytest.raises(ValueError, match="OpenAPI 3"):
        parse_openapi({"swagger": "2.0", "paths": {}})


def test_missing_operation_id_synthesized_and_noted():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"post": {
                "summary": "Make a widget.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {"size": {"type": "string"}},
                    "required": ["size"]}}}}}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["tool_name"] == "post_widget"
    assert op["operation"] == "post_widget"
    assert any("synthesized" in n for n in result.notes)


def test_collection_get_without_single_filter_is_undraftable():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"get": {"operationId": "listWidgets"}}}}
    result = parse_openapi(spec)
    assert result.operations == ()
    assert any("exactly one required filter" in n for n in result.notes)


def test_collection_get_with_one_required_filter_becomes_list_lookup():
    spec = {"openapi": "3.1.0",
            "paths": {"/widgets": {"get": {
                "operationId": "findWidgets",
                "parameters": [{"name": "owner_phone", "in": "query",
                                "required": True,
                                "schema": {"type": "string"}}]}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["operation"] == "__list__"
    assert op["filter_param"] == "owner_phone"
    assert op["resource_type"] == "widget"


def test_refund_named_post_escalates_to_high():
    spec = {"openapi": "3.1.0",
            "paths": {"/payments/refund": {"post": {
                "operationId": "processRefund",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {"payment_id": {"type": "string"}},
                    "required": ["payment_id"]}}}}}}}}
    (op,) = parse_openapi(spec).operations
    assert op["risk_class"] == "high"


def test_refs_resolved_and_unscalar_required_property_dropped_with_note():
    spec = {"openapi": "3.1.0",
            "components": {"schemas": {"Guest": {
                "type": "object",
                "properties": {"guest_name": {"type": "string"}},
                "required": ["guest_name"]}}},
            "paths": {"/bookings": {"post": {
                "operationId": "createBooking",
                "requestBody": {"content": {"application/json": {"schema": {
                    "$ref": "#/components/schemas/Guest"}}}}}}}}
    result = parse_openapi(spec)
    (op,) = result.operations
    assert op["params"] == ["guest_name"]
    assert op["param_types"] == {"guest_name": "string"}


def test_post_without_any_params_is_skipped_with_note():
    spec = {"openapi": "3.1.0",
            "paths": {"/ping": {"post": {"operationId": "ping"}}}}
    result = parse_openapi(spec)
    assert result.operations == ()
    assert any("no required parameters" in n for n in result.notes)


# --- report, artifact writer, enrich seam ------------------------------------

from voiceagent.openapi_draft import discovery_report, enrich, \
    write_proposals_yaml
from voiceagent.proposals import load_proposals_yaml


def test_discovery_report_speaks_the_onboarding_moment():
    report = discovery_report(parse_openapi(HOTEL_SPEC),
                              title="Grand Hotel PMS")
    assert "Discovered API: Grand Hotel PMS" in report
    assert "6 operations: 3 read, 2 mutating, 1 high-risk" in report
    assert "[HIGH RISK] cancel_booking(booking_id)" in report
    assert "[read] search_rooms(check_in, check_out)" in report
    assert "status: proposed" in report


def test_write_then_load_roundtrip_is_the_approval_artifact(tmp_path):
    result = parse_openapi(HOTEL_SPEC)
    out = tmp_path / "proposals.yaml"
    names = write_proposals_yaml(result, out)
    assert set(names) == {"search_rooms", "get_room", "create_booking",
                          "get_booking", "modify_booking", "cancel_booking"}
    props = load_proposals_yaml(out)
    assert all(p.status == "proposed" and p.provenance == "ai"
               for p in props)
    by = {p.name: p for p in props}
    assert by["create_booking"].param_types == \
        {"guest_name": "string", "room_id": "string", "check_in": "string"}
    assert by["get_room"].operation == "__fetch__"
    assert by["get_room"].resource_type == "room"
    assert by["cancel_booking"].risk_class == "high"


def test_writer_refuses_invalid_drafts(tmp_path):
    bad = DraftResult(
        operations=({"operation": "x", "tool_name": "Bad Name!",
                     "description": "d", "params": [], "side_effects": False,
                     "risk_class": "read", "param_types": {},
                     "action": "x"},),
        notes=())
    with pytest.raises(ValueError, match="refusing to write invalid draft"):
        write_proposals_yaml(bad, tmp_path / "proposals.yaml")


def test_enrich_default_is_identity():
    result = parse_openapi(HOTEL_SPEC)
    assert enrich(result) is result


def test_enrich_may_only_touch_descriptions():
    result = parse_openapi(HOTEL_SPEC)

    def _enricher(ops):
        out = [{**op, "description": f"OWNER COPY: {op['description']}"}
               for op in ops]
        return out

    enriched = enrich(result, _enricher)
    assert enriched.operations[0]["description"].startswith("OWNER COPY")
    assert enriched.operations[0]["params"] == \
        result.operations[0]["params"]
    assert enriched.notes == result.notes


def test_enrich_changing_mechanics_is_refused():
    result = parse_openapi(HOTEL_SPEC)

    def _sneaky(ops):
        out = [dict(op) for op in ops]
        out[0]["risk_class"] = "high"  # NOT allowed — risk is deterministic
        return out

    with pytest.raises(ValueError, match="may only rewrite 'description'"):
        enrich(result, _sneaky)
