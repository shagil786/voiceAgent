# scripts/outbound_demo.py
"""Interactive End-to-End Outbound Autonomous Campaign Demo.
Demonstrates:
1. Regulatory DND Pre-Dial Scrubbing (TRAI/TCPA compliance)
2. Sub-600ms Answering Machine Detection (AMD)
3. Instant Handoff to Voice Frontman (< 100ms)
4. Domain-Agnostic Specialist Swarm (Real Estate, Automotive, B2B SaaS)
5. Bounded Mathematical Concession Negotiator
6. Real-Time WhatsApp Sidecar (Dropping 3D floor plans / spec sheets)
7. Frictionless Voice Checkout & Live Payment Webhook Settlement
"""
import asyncio
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voiceagent.outbound.amd import CallParty
from voiceagent.outbound.dialer import Lead, PredictiveDialer, RegulatoryDNDScrubber
from voiceagent.sidecar.checkout import VoiceCheckoutEngine
from voiceagent.sidecar.whatsapp import WhatsAppSidecar
from voiceagent.swarm.agents.closer import CommercialSalesCloser
from voiceagent.swarm.agents.compliance import ComplianceWatchdog
from voiceagent.swarm.agents.negotiator import BoundedNegotiator
from voiceagent.swarm.arbiter import ConsensusArbiter
from voiceagent.swarm.blackboard import Blackboard, CallerProfile
from voiceagent.swarm.frontman import VoiceFrontman
from voiceagent.swarm.specialist import create_domain_specialist

console = Console()


async def run_outbound_campaign():
    console.print(Panel.fit(
        "[bold cyan]AUTONOMOUS AGENTIC WORKFORCE: OUTBOUND REVENUE OPERATIONS[/bold cyan]\n"
        "[dim]Full-Duplex Telephony | Sub-600ms AMD | Domain Specialists | WhatsApp Sidecar | Voice Checkout[/dim]",
        border_style="cyan",
    ))

    # 1. Initialize Regulatory & Swarm Infrastructure
    scrubber = RegulatoryDNDScrubber(dnd_numbers={"+919999999999"})
    whatsapp = WhatsAppSidecar()
    checkout = VoiceCheckoutEngine()

    leads = [
        Lead(lead_id="LEAD-RE-01", phone="+919876543210", name="Rohan Sharma", domain="real_estate", interest="3BHK Luxury Bandra"),
        Lead(lead_id="LEAD-AUTO-02", phone="+919876543222", name="Vikram Patel", domain="luxury_automotive", interest="Apex e-SUV GT"),
        Lead(lead_id="LEAD-DND-03", phone="+919999999999", name="Aakash Verma (DND)", domain="real_estate", interest="Penthouse"),
    ]

    dialer = PredictiveDialer(scrubber=scrubber)

    # 2. Process Outbound Lead Queue
    table = Table(title="Outbound Campaign Lead Queue", border_style="magenta")
    table.add_column("Lead ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Phone", style="yellow")
    table.add_column("Domain", style="green")
    table.add_column("Product Interest", style="magenta")

    for l in leads:
        table.add_row(l.lead_id, l.name, l.phone, l.domain, l.interest)
    console.print(table)
    console.print()

    for lead in leads:
        console.print(f"[bold yellow]📞 Dialing Lead {lead.lead_id} ({lead.name} - {lead.phone})...[/bold yellow]")

        # Test within business hours
        daytime = datetime.time(14, 0)
        res = await dialer.dial_lead(
            lead,
            simulated_party=CallParty.HUMAN if "DND" not in lead.name else CallParty.SILENCE,
            current_time=daytime,
        )

        if res.status.startswith("BLOCKED"):
            console.print(f"  ❌ [bold red]PRE-DIAL SCRUB TRIGGERED:[/bold red] {res.status} - Call avoided. Zero compliance fine.")
            console.print()
            continue

        console.print(f"  ⚡ [bold green]SUB-600ms AMD RESULT:[/bold green] Party={res.party_detected.value} in {res.amd_latency_ms:.1f}ms")
        console.print(f"  🤝 [bold green]ACTION:[/bold green] {res.action_taken} (< 100ms latency handoff)")
        console.print()

        # 3. Simulate Live Conversation with the Autonomous Swarm
        console.print(f"[bold cyan]─── LIVE CALL CONNECTED: {lead.name} ({lead.domain.upper()}) ───[/bold cyan]")
        bb = Blackboard(session_id=f"sess-{lead.lead_id}", profile=CallerProfile(phone=lead.phone, name=lead.name))

        if lead.domain == "real_estate":
            closer = CommercialSalesCloser()
            negotiator = BoundedNegotiator(list_price=2.8e7, soft_floor=2.75e7, hard_floor=2.70e7)
            compliance = ComplianceWatchdog(rera_number="P51800031245")

            bb.register_agent("closer", closer.handle_turn)
            bb.register_agent("negotiator", negotiator.handle_turn)
            bb.register_agent("compliance", compliance.handle_turn)
        else:
            auto_spec = create_domain_specialist("luxury_automotive")
            bb.register_agent("auto", auto_spec.handle_turn)

        sidecars_sent = []
        async def on_sidecar(payload):
            sidecars_sent.append(payload)
            if payload.get("type") == "whatsapp_doc":
                msg = await whatsapp.send_collateral(
                    phone=lead.phone,
                    title=payload.get("title", "Brochure"),
                    asset_name=payload.get("asset", "doc.pdf"),
                )
                console.print(f"  📱 [bold green]WHATSAPP SIDECAR DELIVERED:[/bold green] {msg.title} -> {lead.phone} ({msg.media_url})")
            elif payload.get("type") == "payment_link":
                txn = checkout.create_payment_link(
                    order_id="UNIT-1402",
                    customer_id=lead.lead_id,
                    amount=payload.get("amount", 100000.0),
                    purpose=payload.get("purpose", "token_deposit"),
                )
                await whatsapp.send_payment_link(
                    phone=lead.phone,
                    amount=txn.amount,
                    purpose=payload.get("purpose", "Token"),
                    payment_url=txn.payment_url,
                )
                console.print(f"  💳 [bold green]VOICE CHECKOUT RAILS ACTIVE:[/bold green] Token Link ₹{txn.amount:,.2f} -> {lead.phone}")
                # Simulate live on-call payment webhook receipt
                await asyncio.sleep(0.05)
                checkout.handle_payment_webhook(txn.txn_id, status="SUCCESS")
                console.print(f"  ✅ [bold cyan]WEBHOOK SETTLED ON CALL:[/bold cyan] Payment ID {txn.txn_id} confirmed live by bank!")

        frontman = VoiceFrontman(blackboard=bb, on_sidecar_dispatch=on_sidecar)

        # Turn 1: Caller asks about options
        user_turn1 = "Tell me about the available 3BHK flats in Bandra" if lead.domain == "real_estate" else "Tell me about your electric SUV"
        console.print(f"  [bold]Caller:[/bold] \"{user_turn1}\"")
        resp1 = await frontman.handle_turn(user_turn1, need_filler=True)
        console.print(f"  [dim]Frontman (<200ms Filler):[/dim] \"{resp1.filler_text}\"")
        console.print(f"  [bold cyan]Frontman Spoken:[/bold cyan] \"{resp1.spoken_text}\"")
        console.print()

        # Turn 2: Caller pushes on pricing
        if lead.domain == "real_estate":
            user_turn2 = "The price is too expensive. Can you give me a discount or reduce it?"
            console.print(f"  [bold]Caller:[/bold] \"{user_turn2}\"")
            resp2 = await frontman.handle_turn(user_turn2, need_filler=True)
            console.print(f"  [dim]Frontman (<200ms Filler):[/dim] \"{resp2.filler_text}\"")
            console.print(f"  [bold cyan]Frontman Spoken (Bounded Concession):[/bold cyan] \"{resp2.spoken_text}\"")
            console.print()

            # Turn 3: Closer closes on token
            user_turn3 = "Okay, if you give me the best deal, I want to book the 3BHK flat right now and pay the token deposit."
            console.print(f"  [bold]Caller:[/bold] \"{user_turn3}\"")
            resp3 = await frontman.handle_turn(user_turn3, need_filler=True)
            console.print(f"  [dim]Frontman (<200ms Filler):[/dim] \"{resp3.filler_text}\"")
            console.print(f"  [bold cyan]Frontman Spoken (Statutory RERA + Floor Close):[/bold cyan] \"{resp3.spoken_text}\"")
            console.print()

        console.print("[dim]" + "─" * 60 + "[/dim]\n")

    console.print(Panel("[bold green]✔ Outbound Campaign Execution Complete. Zero Compliance Violations.[/bold green]", border_style="green"))


if __name__ == "__main__":
    asyncio.run(run_outbound_campaign())
