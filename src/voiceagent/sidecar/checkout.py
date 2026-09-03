# src/voiceagent/sidecar/checkout.py
"""Frictionless Voice Checkout & Instant Payment Rails.
Generates one-click UPI/payment links, detects webhooks in real-time,
and confirms transactions live on the call.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaymentTransaction:
    txn_id: str
    order_id: str
    customer_id: str
    amount: float
    currency: str = "INR"
    status: str = "PENDING"  # PENDING, SUCCESS, FAILED
    payment_url: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class VoiceCheckoutEngine:
    """Manages on-call payment link generation and live webhook settlement."""

    def __init__(self):
        self.transactions: dict[str, PaymentTransaction] = {}

    def create_payment_link(
        self,
        order_id: str,
        customer_id: str,
        amount: float,
        purpose: str = "token_deposit",
    ) -> PaymentTransaction:
        txn_id = f"PAY-{uuid.uuid4().hex[:10].upper()}"
        pay_url = f"https://pay.domain.com/checkout/{txn_id}"
        txn = PaymentTransaction(
            txn_id=txn_id,
            order_id=order_id,
            customer_id=customer_id,
            amount=amount,
            status="PENDING",
            payment_url=pay_url,
        )
        self.transactions[txn_id] = txn
        return txn

    def handle_payment_webhook(self, txn_id: str, status: str = "SUCCESS") -> PaymentTransaction:
        """Simulate or process incoming payment webhook from bank/gateway."""
        txn = self.transactions.get(txn_id)
        if not txn:
            raise KeyError(f"transaction not found: {txn_id}")
        txn.status = status
        txn.completed_at = time.time()
        return txn

    def check_payment_status(self, txn_id: str) -> str:
        txn = self.transactions.get(txn_id)
        return txn.status if txn else "NOT_FOUND"
