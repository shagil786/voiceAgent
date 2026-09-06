# src/voiceagent/demo_data.py — the platform's built-in DEMO TENANT fallback
# data, not core.
"""What still lives here, honestly (post Task E): the BASE-tier reply
mechanics tables (canned replies, empty-reply acknowledgement lines, empathy
prefixes) that voiceagent.reply_guards consumes, plus the demo deployment's
action vocabulary and tool contracts that the no-bundle Agent path
(voiceagent.agent) merges into its guard registries. Everything else that
used to be demo Python content moved to the COMMITTED default tenant bundle
data/tenants/default/ (identity + knowledge in tenant.json / knowledge/,
intent exemplars in intents/, the ERP fixture in erp_fixture.json) and loads
through the same Tenant machinery as named customer bundles.

These tables stay in Python deliberately: they are the reply-guard MECHANICS'
text tables — consumed only by the guard pipeline as the no-bundle fallback —
and carrying them in code keeps the guard module importable with zero file
I/O. A real tenant deployment never imports this module; its bundle declares
its own vocabulary and contracts (see voiceagent.tenant.Tenant).
"""
from __future__ import annotations

from voiceagent.tools import ToolSpec

# The demo deployment's action vocabulary (formerly agent.DEFAULT_ACTIONS,
# where it was core code). Consumed only when neither the policy engine
# (PolicyEngine.known_actions) nor the assembly seam declares a vocabulary.
DEMO_TENANT_ACTIONS = [
    "order_status", "refund", "cancel_order", "address_change",
    "payment_declined", "recharge", "billing", "return", "replacement",
    "otp", "fraud", "account_closure", "delivery_delay", "product_info",
    "invoice", "plan_change", "roaming", "network_issue", "complaint",
    "high_value_refund", "refund_info", "delivery_eta",
]

# Demo tool contracts for intent-actions that have NO code-bound tool (they
# are policy/guard data, never executed through the ToolGateway — hence
# params=()). `facts` are the customer-visible guarantees the echo guardrail
# enforces when the customer states them; formerly agent.KEYWORD_FACTS, where
# they were core code. A tenant declares its own contracts via tools.yaml
# `facts:` on real tools; these demo entries are the no-bundle fallback.
DEMO_TENANT_CONTRACT_SPECS: dict[str, ToolSpec] = {
    "fraud": ToolSpec(params=(), facts=("block",)),
    "otp": ToolSpec(params=(), facts=("otp",)),
    "billing": ToolSpec(params=(), facts=("bill",)),
    "payment_declined": ToolSpec(params=(), facts=("declined",)),
    "recharge": ToolSpec(params=(), facts=("fail", "recharge")),
    "refund_info": ToolSpec(params=(), facts=("refund",)),
    "delivery_eta": ToolSpec(params=(), facts=("order", "delivery")),
}

# ---------------------------------------------------------------------------
# Demo reply templates (Task D1: moved VERBATIM from voiceagent.agent, where
# they were core code). The deterministic canned replies the reply guards fall
# back to (voiceagent.reply_guards._canned_reply), the empty-reply
# acknowledgement lines, and the M6a empathy prefixes. DEMO TENANT DATA: a
# real tenant deployment overrides these; the demo tables are the no-bundle
# fallback. The reply-language guardrail rationale lives with the guard code
# in voiceagent.reply_guards.
# ---------------------------------------------------------------------------

REPLY_TEMPLATES: dict[str, dict[str, str]] = {
    "order_status": {
        "hi": "आपके ऑर्डर {ref} की स्थिति जाँच ली गई है। ताज़ा स्थिति जल्द ही आपके ऐप और एसएमएस पर अपडेट होगी।",
        "te": "మీ ఆర్డర్ {ref} స్థితి తనిఖీ చేయబడింది. తాజా స్థితి త్వరలో మీ యాప్‌లో మరియు ఎస్ఎంఎస్ ద్వారా అందుతుంది.",
        "hinglish": "Aapke order {ref} ka status check kar liya gaya hai. Latest update jald hi app aur SMS par milega.",
        "en": "Your order {ref} has been checked. The latest status will "
              "arrive in your app and by SMS shortly.",
        "es": "Su pedido {ref} ha sido verificado. El estado más reciente "
              "llegará pronto a su aplicación y por SMS.",
        "fr": "Votre commande {ref} a été vérifiée. Le statut le plus "
              "récent arrivera bientôt dans votre application et par SMS.",
        "de": "Ihre Bestellung {ref} wurde überprüft. Der aktuelle Status "
              "kommt in Kürze in Ihre App und per SMS.",
        "pt": "Seu pedido {ref} foi verificado. O status mais recente "
              "chegará em breve no seu aplicativo e por SMS.",
    },
    "refund": {
        "hi": "आपका रिफंड अनुरोध दर्ज हो गया है। प्रक्रिया पूरी होने पर स्थिति की जानकारी दी जाएगी।",
        "te": "మీ రీఫండ్ అభ్యర్థన నమోదైంది. ప్రక్రియ పూర్తయిన తర్వాత స్థితి తెలియజేయబడుతుంది.",
        "hinglish": "Aapka refund request note kar liya gaya hai. Process complete hone par status update mil jayega.",
        "en": "Your refund request has been recorded. You will be informed "
              "once the process is complete.",
        "es": "Su solicitud de reembolso ha sido registrada. Se le "
              "informará cuando el proceso esté completo.",
        "fr": "Votre demande de remboursement a été enregistrée. Vous "
              "serez informé une fois le processus terminé.",
        "de": "Ihre Rückerstattungsanfrage wurde aufgenommen. Sie werden "
              "informiert, sobald der Vorgang abgeschlossen ist.",
        "pt": "Sua solicitação de reembolso foi registrada. Você será "
              "informado quando o processo for concluído.",
    },
    "refund_info": {
        "hi": "रिफंड स्वीकृत होने के 5-7 कार्यदिवसों में आपके खाते में आ जाता है।",
        "te": "రీఫండ్ ఆమోదించబడిన 5-7 పనిదినాల్లో మీ ఖాతాలో జమ అవుతుంది.",
        "hinglish": "Refund approve hone ke 5-7 working days mein aapke account mein aa jata hai.",
        "en": "Refunds reach your account within 5-7 working days of "
              "approval.",
        "es": "El reembolso llega a su cuenta dentro de 5-7 días hábiles "
              "después de la aprobación.",
        "fr": "Le remboursement arrive sur votre compte dans les 5-7 jours "
              "ouvrés suivant l'approbation.",
        "de": "Die Rückerstattung trifft innerhalb von 5-7 Werktagen nach "
              "Genehmigung auf Ihrem Konto ein.",
        "pt": "O reembolso chega à sua conta em 5-7 dias úteis após a "
              "aprovação.",
    },
    "delivery_eta": {
        "hi": "आपका ऑर्डर 3-5 कार्यदिवसों में डिलीवर होने की उम्मीद है।",
        "te": "మీ ఆర్డర్ 3-5 పనిదినాల్లో డెలివరీ అవుతుందని భావిస్తున్నాము.",
        "hinglish": "Aapka order 3-5 working days mein deliver hone ki expectation hai.",
        "en": "Your order is expected to be delivered within 3-5 working "
              "days.",
        "es": "Se espera que su pedido llegue dentro de 3-5 días hábiles.",
        "fr": "Votre commande devrait être livrée dans les 3-5 jours "
              "ouvrés.",
        "de": "Ihre Bestellung wird voraussichtlich innerhalb von 3-5 "
              "Werktagen geliefert.",
        "pt": "Seu pedido deve ser entregue em 3-5 dias úteis.",
    },
    "default": {
        "hi": "आपका अनुरोध दर्ज कर लिया गया है। हमारी टीम जल्द ही आपकी सहायता करेगी।",
        "te": "మీ అభ్యర్థన నమోదు చేయబడింది. మా బృందం త్వరలో మీకు సహాయం చేస్తుంది.",
        "hinglish": "Aapka request note kar liya gaya hai. Hamari team jald hi aapki help karegi.",
        "en": "Your request has been recorded. Our team will assist you "
              "shortly.",
        "es": "Su solicitud ha sido registrada. Nuestro equipo le ayudará "
              "pronto.",
        "fr": "Votre demande a été enregistrée. Notre équipe vous aidera "
              "bientôt.",
        "de": "Ihre Anfrage wurde aufgenommen. Unser Team wird Ihnen in "
              "Kürze helfen.",
        "pt": "Sua solicitação foi registrada. Nossa equipe irá ajudá-lo "
              "em breve.",
    },
}



NOTED_REPLIES = {
    "hi": "आपका अनुरोध दर्ज कर लिया गया है।",
    "te": "మీ అభ్యర్థన నమోదు చేయబడింది.",
    "hinglish": "Aapka request note kar liya gaya hai.",
    "en": "Your request has been noted.",
    "es": "Su solicitud ha sido registrada.",
    "fr": "Votre demande a été enregistrée.",
    "de": "Ihre Anfrage wurde aufgenommen.",
    "pt": "Sua solicitação foi registrada.",
}

# M6a: empathy lines for HIGH frustration, in the customer's language
# (languages without an entry get no prefix — never a wrong-language one).

EMPATHY_PREFIXES = {
    "en": "I'm really sorry about the trouble. ",
    "hinglish": "Mujhe khed hai ki aapko pareshani hui. ",
    "hi": "मुझे खेद है कि आपको परेशानी हुई। ",
    "te": "ఇబ్బంది కోసం క్షమించండి. ",
    "es": "Lamento mucho las molestias. ",
    "fr": "Je suis vraiment désolé pour ce désagrément. ",
    "de": "Es tut mir wirklich leid für die Umstände. ",
    "pt": "Sinto muito pelo inconveniente. ",
}



