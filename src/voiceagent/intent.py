# src/voiceagent/intent.py
"""Deterministic intent classifier built on the same multilingual embeddings
used for RAG. The action decision is a nearest-neighbour match against curated
exemplar queries — no LLM involved, so it cannot drift format or reason itself
into the wrong action the way a small generative model does.

Output is constrained to the fixed intent vocabulary by construction.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "order_status": [
        "Where is my order?",
        "Mera order abhi tak nahi aaya",
        "मेरा ऑर्डर कहाँ है",
        "order delivery status",
        "has my order shipped",
    ],
    "refund": [
        "I need a refund for my order",
        "Can you refund my order",
        "मुझे रिफंड चाहिए",
        "refund my money",
        "mera paisa wapas karo",
    ],
    "cancel_order": [
        "cancel my order",
        "I want to cancel the order",
        "ऑर्डर कैंसिल करो",
        "cancel order before shipping",
    ],
    "address_change": [
        "change my delivery address",
        "update shipping address",
        "पता बदलना है",
        "my address is wrong, change it",
    ],
    "payment_declined": [
        "why was my payment declined",
        "payment fail ho gaya",
        "payment failed",
        "मेरा पेमेंट फेल हो गया",
    ],
    "recharge": [
        "my recharge failed",
        "recharge nahi hua",
        "रिचार्ज क्यों फेल हुआ",
        "top up failed",
    ],
    "billing": [
        "I don't understand my bill",
        "bill samajh nahi aaya",
        "बिल समझ नहीं आया",
        "why was I charged",
        "what is this charge on my bill",
    ],
    "return": [
        "I want to return an item",
        "product wapas karna hai",
        "return my product",
    ],
    "replacement": [
        "I got a damaged item, send a replacement",
        "replace my product",
        "product kharab aaya, replace karo",
    ],
    "otp": [
        "OTP nahi aaya mere phone pe",
        "resend the OTP",
        "मुझे OTP नहीं मिला",
        "did not receive OTP",
        "OTP not received",
    ],
    "fraud": [
        "someone used my account, block it",
        "mera account hack ho gaya",
        "fraud transaction on my account",
        "मेरे अकाउंट से पैसे कट गए बिना मेरी जानकारी के",
        "unauthorized transaction, block now",
    ],
    "account_closure": [
        "close my account",
        "delete my account",
        "खाता बंद करो",
        "how do I close my account",
    ],
    "delivery_delay": [
        "my order is late, where is it",
        "delivery bahut late ho rahi hai",
        "order delay hone par kya karein",
        "why is my delivery delayed",
    ],
    "product_info": [
        "tell me about this product",
        "product ki jankari do",
        "is this item in stock",
        "product specifications",
    ],
    "invoice": [
        "I need my invoice",
        "invoice kaise milega",
        "send me the bill receipt",
        "download my invoice",
    ],
    "plan_change": [
        "change my mobile plan",
        "plan badalna hai",
        "upgrade my plan",
        "switch to a cheaper plan",
    ],
    "roaming": [
        "international roaming not working",
        "roaming charges kya hain",
        "enable roaming",
        "roaming pack activate karo",
    ],
    "network_issue": [
        "network is down",
        "network nahi chal raha",
        "internet not working",
        "no signal on my phone",
    ],
    "complaint": [
        "I want to file a complaint",
        "shikayat karni hai",
        "complaint register karo",
        "I am unhappy with the service",
    ],
    "high_value_refund": [
        "I want a refund of 10000 rupees",
        "my 25000 refund is pending",
        "बड़ी रकम का रिफंड",
        "refund of 20000",
        "high value refund",
        "I need a large refund of 50000",
        "big amount ka refund chahiye",
        "मेरी बड़ी रकम वापस करो",
        "refund of 25000 rupees urgently",
        "my 100000 refund is stuck",
        "huge refund pending for 2 months",
    ],
}


class IntentClassifier:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model = SentenceTransformer(model_name)
        self._intents: list[str] = []
        self._exemplar_embs: np.ndarray | None = None
        self._exemplar_labels: list[str] = []
        self._build()

    def _build(self) -> None:
        queries: list[str] = []
        labels: list[str] = []
        for intent, exs in INTENT_EXEMPLARS.items():
            for ex in exs:
                queries.append(ex)
                labels.append(intent)
        emb = self._model.encode(queries, normalize_embeddings=True)
        self._exemplar_embs = np.asarray(emb, dtype=np.float32)
        self._exemplar_labels = labels
        self._intents = list(INTENT_EXEMPLARS.keys())

    def classify(self, text: str, k: int = 1) -> tuple[str, float]:
        """Return (best_intent, cosine_score)."""
        q = self._model.encode([text], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        scores = self._exemplar_embs @ q.T  # (n_exemplars, 1)
        scores = scores[:, 0]
        order = np.argsort(-scores)[:k]
        best = int(order[0])
        return self._exemplar_labels[best], float(scores[best])
