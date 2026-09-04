# tests/test_swarm.py
"""End-to-end unit and integration tests for the Autonomous Swarm Architecture:
Dual-Loop Engine, Blackboard Event Bus, Consensus Arbiter, Bounded Negotiator,
Compliance Watchdog, WhatsApp Sidecar, and Voice Checkout Rails.
"""
import asyncio
import pytest

from voiceagent.sidecar.checkout import VoiceCheckoutEngine
from voiceagent.sidecar.whatsapp import WhatsAppSidecar
from voiceagent.swarm.agents.closer import CommercialSalesCloser
from voiceagent.swarm.agents.compliance import ComplianceWatchdog
from voiceagent.swarm.agents.negotiator import BoundedNegotiator
from voiceagent.swarm.arbiter import ConsensusArbiter
from voiceagent.swarm.blackboard import (
    PRIORITY_COMPLIANCE,
    PRIORITY_PRICING,
    PRIORITY_RISK,
    PRIORITY_SALES,
    Blackboard,
    CallerProfile,
    Proposal,
)
from voiceagent.swarm.frontman import VoiceFrontman


def test_blackboard_concurrent_dispatch():
    """Verify that multiple registered sub-agents execute in parallel on the blackboard."""
    async def _run():
        bb = Blackboard(session_id="call-101")

        async def agent_a(state, text):
            await asyncio.sleep(0.02)
            return Proposal(source_agent="agent_a", priority=PRIORITY_SALES, content="Proposal A")

        async def agent_b(state, text):
            await asyncio.sleep(0.01)
            return Proposal(source_agent="agent_b", priority=PRIORITY_PRICING, content="Proposal B")

        bb.register_agent("a", agent_a)
        bb.register_agent("b", agent_b)

        proposals = await bb.dispatch("I am interested in buying")
        assert len(proposals) == 2
        # Proposals must be sorted by priority descending: agent_b (PRICING 60) > agent_a (SALES 40)
        assert proposals[0].source_agent == "agent_b"
        assert proposals[1].source_agent == "agent_a"

    asyncio.run(_run())


def test_arbiter_priority_hierarchy_and_veto():
    """Verify that Arbiter enforces Compliance > Risk > Pricing > Sales and handles vetoes."""
    arbiter = ConsensusArbiter()

    # Case 1: Risk Veto blocks Sales offer
    sales_prop = Proposal(source_agent="closer", priority=PRIORITY_SALES, content="I can approve your loan right now.")
    risk_veto = Proposal(
        source_agent="credit_risk",
        priority=PRIORITY_RISK,
        veto=True,
        veto_reason="Credit score below 650",
        content="We cannot approve without additional guarantor documentation.",
    )

    dec1 = arbiter.arbitrate([sales_prop, risk_veto])
    assert "RISK_VETO: Credit score below 650" in dec1.vetoes_applied
    assert "additional guarantor" in dec1.spoken_content
    assert dec1.action == "risk_escalation"

    # Case 2: Statutory Compliance Disclosure is strictly appended
    comp_prop = Proposal(
        source_agent="compliance",
        priority=PRIORITY_COMPLIANCE,
        content="MahaRERA registration No. P51800031245.",
    )
    deal_prop = Proposal(
        source_agent="closer",
        priority=PRIORITY_SALES,
        content="Your 3BHK booking is confirmed.",
    )
    dec2 = arbiter.arbitrate([deal_prop, comp_prop])
    assert "Your 3BHK booking is confirmed." in dec2.spoken_content
    assert "MahaRERA registration No. P51800031245." in dec2.spoken_content


def test_bounded_negotiator_concession_curve():
    """Verify BoundedNegotiator step-down curve: List Price -> Soft Concession -> Hard Floor."""
    async def _run():
        negotiator = BoundedNegotiator(list_price=1.5e7, soft_floor=1.48e7, hard_floor=1.46e7)
        bb = Blackboard(session_id="nego-1")

        # Round 1: Resist and anchor on list price
        p1 = await negotiator.handle_turn(bb.state, "Can you give me a discount on the price?")
        assert p1.action == "anchor_list_price"
        assert "1.50 Cr" in p1.content

        # Simulate round 1 completed in history
        bb.state.append_turn("customer", "That is too expensive, give more discount")

        # Round 2: Soft concession (Free parking)
        p2 = await negotiator.handle_turn(bb.state, "Give me a better discount")
        assert p2.action == "soft_concession"
        assert "1.48 Cr" in p2.content
        assert "covered parking" in p2.content

        # Simulate round 2 completed in history
        bb.state.append_turn("customer", "Still expensive, come to 1.40 Cr")

        # Round 3: Hard Floor (1.46 Cr) with token deposit requirement
        p3 = await negotiator.handle_turn(bb.state, "Make it cheaper")
        assert p3.action == "hard_floor_close"
        assert "1.46 Cr" in p3.content
        assert p3.params.get("requires_token") is True

    asyncio.run(_run())


def test_compliance_watchdog_triggers_rera_and_apr():
    """Verify ComplianceWatchdog forces statutory disclosures upon purchase or booking intent."""
    async def _run():
        watchdog = ComplianceWatchdog(rera_number="P51800099999", banking_apr_disclosure="14.5% APR.")
        bb = Blackboard(session_id="comp-1")

        # Call mentions booking a 3BHK -> triggers RERA disclosure
        p1 = await watchdog.handle_turn(bb.state, "I want to book this 3BHK flat and pay token")
        assert p1 is not None
        assert "P51800099999" in p1.content

        # Second query: disclosure already given, does not repeat unnecessarily
        p2 = await watchdog.handle_turn(bb.state, "Book it now")
        assert p2 is None

    asyncio.run(_run())


def test_dual_loop_frontman_with_sidecar_dispatch():
    """Verify VoiceFrontman handles turn, emits filler, executes background swarm,
    and dispatches visual WhatsApp collateral."""
    async def _run():
        bb = Blackboard(session_id="call-dual-1")
        closer = CommercialSalesCloser()
        compliance = ComplianceWatchdog()

        bb.register_agent("closer", closer.handle_turn)
        bb.register_agent("compliance", compliance.handle_turn)

        sidecar_calls = []

        async def sidecar_sink(payload):
            sidecar_calls.append(payload)

        frontman = VoiceFrontman(blackboard=bb, on_sidecar_dispatch=sidecar_sink)

        res = await frontman.handle_turn("Tell me about available 3BHK apartments in Bandra", need_filler=True)

        # Fast loop: filler generated immediately
        assert res.filler_text is not None
        assert "inventory" in res.filler_text.lower() or "details" in res.filler_text.lower()

        # Deep swarm result synthesized
        assert "Bandra West" in res.spoken_text
        assert "1,250 sq.ft" in res.spoken_text

        # Omnichannel sidecar dispatched to WhatsApp
        assert len(sidecar_calls) >= 1
        assert sidecar_calls[0].get("asset") == "bandra_west_3bhk_floorplan.pdf"

    asyncio.run(_run())


def test_omnichannel_whatsapp_and_checkout_rails():
    """Verify real-time WhatsApp collateral delivery and live on-call payment webhook."""
    async def _run():
        wa = WhatsAppSidecar()
        checkout = VoiceCheckoutEngine()

        # 1. Dispatch PDF floor plan
        doc_msg = await wa.send_collateral(
            phone="+919876543210",
            title="3BHK Floor Plan",
            asset_name="bandra_floorplan.pdf",
        )
        assert doc_msg.status == "delivered"
        assert "bandra_floorplan.pdf" in doc_msg.media_url

        # 2. Generate on-call token checkout link
        txn = checkout.create_payment_link(
            order_id="UNIT-1402",
            customer_id="CUST-77",
            amount=100000.0,
            purpose="token_deposit",
        )
        assert txn.status == "PENDING"
        assert "PAY-" in txn.txn_id

        # 3. Send payment link to customer's WhatsApp during call
        pay_msg = await wa.send_payment_link(
            phone="+919876543210",
            amount=100000.0,
            purpose="Token Deposit",
            payment_url=txn.payment_url,
        )
        assert pay_msg.message_type == "payment_link"

        # 4. Bank webhook received while customer on call -> Instant confirmation
        settled = checkout.handle_payment_webhook(txn.txn_id, status="SUCCESS")
        assert settled.status == "SUCCESS"
        assert checkout.check_payment_status(txn.txn_id) == "SUCCESS"

    asyncio.run(_run())


def test_packed_verticals_match_legacy():
    from voiceagent.swarm.specialist import create_domain_specialist
    auto = create_domain_specialist("luxury_automotive")
    assert auto.spec.catalog[0]["id"] == "EV-SUV-01"
    assert "FAME-II" in auto.spec.statutory_disclosures[0]
    saas = create_domain_specialist("b2b_saas")
    assert saas.spec.catalog[0]["id"] == "PLAN-ENT"
    assert create_domain_specialist("mystery").spec.domain_id == "mystery"


def test_factory_reads_packs():
    import inspect
    from voiceagent.swarm import specialist as mod
    assert "load_vertical" in inspect.getsource(mod.create_domain_specialist)
