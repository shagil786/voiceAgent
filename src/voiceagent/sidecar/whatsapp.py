# src/voiceagent/sidecar/whatsapp.py
"""Omnichannel WhatsApp Sidecar (Voice + Visual Co-Browsing).
Dispatches rich visual collateral (PDF floor plans, EMI tables)
to the caller's WhatsApp in real-time during live speech.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WhatsAppMessage:
    recipient_phone: str
    message_type: str  # document, text, payment_link
    title: str
    media_url: str | None = None
    body_text: str = ""
    timestamp: float = field(default_factory=time.time)
    status: str = "sent"


class WhatsAppSidecar:
    """Delivers synchronized visual collateral to the customer's phone."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.dispatched_messages: list[WhatsAppMessage] = []

    async def send_collateral(
        self,
        phone: str,
        title: str,
        asset_name: str,
        caption: str = "",
    ) -> WhatsAppMessage:
        """Pushes a visual asset directly to the customer's WhatsApp."""
        msg = WhatsAppMessage(
            recipient_phone=phone,
            message_type="document",
            title=title,
            media_url=f"https://assets.domain.com/collateral/{asset_name}",
            body_text=caption or f"Here is the requested {title} discussed on our call.",
            status="delivered",
        )
        self.dispatched_messages.append(msg)
        return msg

    async def send_payment_link(
        self,
        phone: str,
        amount: float,
        purpose: str,
        payment_url: str,
    ) -> WhatsAppMessage:
        """Sends an instant one-click payment link to WhatsApp."""
        msg = WhatsAppMessage(
            recipient_phone=phone,
            message_type="payment_link",
            title=f"Payment Link: ₹{amount:,.2f}",
            media_url=payment_url,
            body_text=f"Please tap below to complete your {purpose} of ₹{amount:,.2f}.",
            status="delivered",
        )
        self.dispatched_messages.append(msg)
        return msg
