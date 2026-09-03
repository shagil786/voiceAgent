# tests/test_specialist.py
"""Tests for Domain-Agnostic Pluggable Specialist Agents.
Proves the swarm can operate across ANY domain: Automotive, B2B SaaS, Insurance, or Custom.
"""
import asyncio
import pytest

from voiceagent.swarm.blackboard import Blackboard
from voiceagent.swarm.frontman import VoiceFrontman
from voiceagent.swarm.specialist import (
    DomainSpecialist,
    SpecialistSpec,
    create_domain_specialist,
)


def test_automotive_domain_specialist():
    """Verify Luxury Automotive specialist recommends EV and attaches brochure sidecar."""
    async def _run():
        bb = Blackboard(session_id="car-101")
        auto_agent = create_domain_specialist("luxury_automotive")
        bb.register_agent("auto", auto_agent.handle_turn)

        sidecars = []
        frontman = VoiceFrontman(blackboard=bb, on_sidecar_dispatch=lambda s: sidecars.append(s))

        res = await frontman.handle_turn("I want to know about your electric SUV and book a test drive")
        assert "Apex e-SUV GT" in res.spoken_text
        assert "620 km" in res.spoken_text
        assert len(sidecars) == 1
        assert sidecars[0].get("asset") == "apex_gt_specsheet.pdf"

    asyncio.run(_run())


def test_b2b_saas_domain_specialist():
    """Verify B2B SaaS specialist qualifies enterprise requirements."""
    async def _run():
        bb = Blackboard(session_id="saas-101")
        saas_agent = create_domain_specialist("b2b_saas")
        bb.register_agent("saas", saas_agent.handle_turn)

        frontman = VoiceFrontman(blackboard=bb)
        res = await frontman.handle_turn("We need an enterprise security plan with SSO for 500 seats")
        assert "Enterprise Shield Plan" in res.spoken_text
        assert "SOC monitoring" in res.spoken_text

    asyncio.run(_run())


def test_custom_user_trained_domain():
    """Verify any arbitrary user-trained domain (e.g. Healthcare / Dental clinic) plugs in cleanly."""
    async def _run():
        custom_spec = SpecialistSpec(
            domain_id="dental_clinic",
            name="Dental Concierge",
            role_description="Aligner consultations and appointment booking.",
            system_prompt="You are a dental clinic appointment specialist.",
            catalog=[
                {
                    "id": "ALIGN-01",
                    "name": "Invisible Clear Aligners",
                    "keywords": ["aligner", "braces", "teeth", "dental", "smile"],
                    "description": "3D digitally planned clear aligners with zero lifestyle disruption.",
                    "price": "₹65,000",
                    "sidecar": {"type": "whatsapp_doc", "title": "Smile Assessment Guide", "asset": "smile_guide.pdf"},
                }
            ],
        )
        dental_agent = DomainSpecialist(spec=custom_spec)

        bb = Blackboard(session_id="dental-1")
        bb.register_agent("dental", dental_agent.handle_turn)

        sidecars = []
        frontman = VoiceFrontman(blackboard=bb, on_sidecar_dispatch=lambda s: sidecars.append(s))

        res = await frontman.handle_turn("Tell me about clear aligners for teeth straightening")
        assert "Invisible Clear Aligners" in res.spoken_text
        assert "₹65,000" in res.spoken_text
        assert len(sidecars) == 1
        assert sidecars[0].get("asset") == "smile_guide.pdf"

    asyncio.run(_run())
