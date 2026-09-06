# src/voiceagent/demo_data.py — DEMO TENANT DATA, not core.
"""The built-in demo deployment's business data (the historical "Acme"
e-commerce demo). This module exists so orchestration core ships NO business
vocabulary: everything here is consumed ONLY as a fallback when no tenant
bundle / policy / deployment declares the corresponding data.

A real tenant never imports this module — their bundle (data/tenants/<name>/
with intents/*.yaml + tools.yaml + policies.yaml) declares its own action
vocabulary and tool contracts; see voiceagent.tenant.Tenant.action_vocabulary.
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


# --- default-tenant identity & knowledge (moved out of runtime.py) ----
DEMO_BUILTIN_IDENTITY = "You are Acme's voice support agent."
DEMO_BUILTIN_KNOWLEDGE: dict[str, str] = {
    "eta": "Deliveries occur between 9:00 and 19:00 local time.",
    "cancel_policy": "Orders that already shipped cannot be cancelled.",
}


# --- default-tenant intent exemplar seeds (moved out of intent.py) ---
DEMO_INTENT_EXEMPLARS: dict[str, list[str]] = {
    "order_status": [
        "Where is my order?",
        "Mera order abhi tak nahi aaya",
        "मेरा ऑर्डर कहाँ है",
        "order delivery status",
        "has my order shipped",
        # M5b: order-status queries that carry an ORD reference were being
        # out-pulled in the latin space by the refund exemplar "refund my
        # order ORD-12345" (70 en eval rows -> refund, en 0.859). These
        # mirror that reference pattern for order_status.
        "where is my order ORD-12345",
        "my order ORD-98765 has not arrived yet",
        "status of my order ORD-45678",
        "मेरा ऑर्डर कहां तक पहुंचा है?",
        "ऑर्डर का स्टेटस बताओ",
        "मेरा ऑर्डर कब शिप होगा",
        "என் ஆர்டர் எந்த நிலையில் உள்ளது?",
        "என் ஆர்டர் எப்போது அனுப்பப்படும்",
        "ஆர்டர் டிராக்கிங் தகவல் வேண்டும்",
        "నా ఆర్డర్ ఎక్కడిది వచ్చింది?",
        "నా ఆర్డర్ స్టేటస్ చెప్పండి",
        "నా ఆర్డర్ ఇంకా రాలేదు",
    ],
    "refund": [
        "I need a refund for my order",
        "Can you refund my order",
        "refund my order ORD-12345",
        "please refund my order, it arrived damaged",
        "मुझे रिफंड चाहिए",
        "refund my money",
        "mera paisa wapas karo",
        "मेरा पैसा वापस करो",
        "ऑर्डर का रिफंड चाहिए",
        "रिफंड की प्रोसेस शुरू करो",
        "என் பணத்தை திரும்பத் தாருங்கள்",
        "ஆர்டருக்கு ரீஃபண்ட் வேண்டும்",
        "ரீஃபண்ட் செய்யுங்கள்",
        "నా డబ్బు వాపస్ చేయండి",
        "ఆర్డర్‌కి రీఫండ్ కావాలి",
        "రీఫండ్ ప్రాసెస్ చేయండి",
    ],
    "cancel_order": [
        "cancel my order",
        "I want to cancel the order",
        "ऑर्डर कैंसिल करो",
        "cancel order before shipping",
        "मेरा ऑर्डर रद्द कर दो",
        "ऑर्डर कैंसिल करना है",
        "यह ऑर्डर अभी रद्द करो",
        "என் ஆர்டரை ரத்து செய்யுங்கள்",
        "ஆர்டரை கேன்சல் செய்ய வேண்டும்",
        "இந்த ஆர்டரை ரத்து செய்யவும்",
        "నా ఆర్డర్ రద్దు చేయండి",
        "ఆర్డర్ క్యాన్సిల్ చేయాలి",
        "ఈ ఆర్డర్ వెంటనే రద్దు చేయండి",
    ],
    "address_change": [
        "change my delivery address",
        "update shipping address",
        "पता बदलना है",
        "my address is wrong, change it",
        "मेरा डिलीवरी पता बदलना है",
        "डिलीवरी एड्रेस अपडेट करो",
        "गलत पते पर ऑर्डर गया है, पता बदलो",
        "என் டெலிவரி முகவரியை மாற்ற வேண்டும்",
        "டெலிவரி முகவரியை புதுப்பிக்கவும்",
        "தவறான முகவரிக்கு ஆர்டர் சென்றது, மாற்றவும்",
        "నా డెలివరీ చిరునామా మార్చాలి",
        "డెలివరీ అడ్రస్ అప్‌డేట్ చేయండి",
        "తప్పు చిరునామాకు ఆర్డర్ వెళ్ళింది, మార్చండి",
    ],
    "payment_declined": [
        "why was my payment declined",
        "payment fail ho gaya",
        "payment failed",
        "मेरा पेमेंट फेल हो गया",
        "मेरा पेमेंट डिक्लाइन हो गया",
        "पेमेंट क्यों रद्द हो गया",
        "कार्ड से पेमेंट नहीं हुआ",
        "என் பேமெண்ட் நிராகரிக்கப்பட்டது",
        "பேமெண்ட் ஏன் தோல்வியடைந்தது",
        "கார்டு மூலம் பணம் செலுத்த முடியவில்லை",
        "నా పేమెంట్ డిక్లైన్ అయింది",
        "పేమెంట్ ఎందుకు ఫెయిల్ అయింది",
        "కార్డుతో పేమెంట్ కాలేదు",
    ],
    "recharge": [
        "my recharge failed",
        "recharge nahi hua",
        "रिचार्ज क्यों फेल हुआ",
        "top up failed",
        "मेरा रिचार्ज नहीं हुआ",
        "रिचार्ज फेल हो गया",
        "पैसे कट गए पर रिचार्ज नहीं मिला",
        "என் ரீசார்ஜ் ஆகவில்லை",
        "ரீசார்ஜ் தோல்வியடைந்தது",
        "பணம் கழிந்தது ஆனால் ரீசார்ஜ் வரவில்லை",
        "నా రీఛార్జ్ కాలేదు",
        "రీఛార్జ్ ఫెయిల్ అయింది",
        "డబ్బులు కట్ అయ్యాయి కానీ రీఛార్జ్ రాలేదు",
    ],
    "billing": [
        "I don't understand my bill",
        "bill samajh nahi aaya",
        "बिल समझ नहीं आया",
        "why was I charged",
        "what is this charge on my bill",
        "मेरा बिल समझ नहीं आ रहा",
        "बिल में एक्स्ट्रा चार्ज क्यों है",
        "गलत बिल आया है",
        "என் பில் புரியவில்லை",
        "பில்லில் கூடுதல் கட்டணம் ஏன்",
        "தவறான பில் வந்துள்ளது",
        "నా బిల్లు అర్థం కాలేదు",
        "బిల్లులో అదనపు ఛార్జీ ఎందుకు",
        "తప్పు బిల్లు వచ్చింది",
    ],
    "return": [
        "I want to return an item",
        "product wapas karna hai",
        "return my product",
        "मुझे प्रोडक्ट वापस करना है",
        "रिटर्न कैसे करूं",
        "यह आइटम वापस करना चाहता हूं",
        "பொருளைத் திரும்பக் கொடுக்க வேண்டும்",
        "ரிட்டர்ன் எப்படி செய்வது",
        "இந்த பொருளை திருப்பி அனுப்ப விரும்புகிறேன்",
        "నాకు ప్రొడక్ట్ రిటర్న్ చేయాలి",
        "రిటర్న్ ఎలా చేయాలి",
        "ఈ ఐటమ్ తిరిగి ఇవ్వాలనుకుంటున్నాను",
    ],
    "replacement": [
        "I got a damaged item, send a replacement",
        "replace my product",
        "product kharab aaya, replace karo",
        "सामान टूटा हुआ आया है, रिप्लेसमेंट दो",
        "मुझे नया प्रोडक्ट भेजो",
        "रिप्लेसमेंट कब आएगा",
        "பொருள் உடைந்து வந்தது, மாற்று பொருள் அனுப்பவும்",
        "எனக்கு புதிய பொருள் அனுப்ப வேண்டும்",
        "ரீப்ளேஸ்மென்ட் எப்போது வரும்",
        "వస్తువు పగిలి వచ్చింది, రీప్లేస్‌మెంట్ ఇవ్వండి",
        "నాకు కొత్త ప్రొడక్ట్ పంపండి",
        "రీప్లేస్‌మెంట్ ఎప్పుడు వస్తుంది",
    ],
    "otp": [
        "OTP nahi aaya mere phone pe",
        "resend the OTP",
        "मुझे OTP नहीं मिला",
        "did not receive OTP",
        "OTP not received",
        "मुझे ओटीपी नहीं मिला",
        "ओटीपी दोबारा भेजो",
        "ओटीपी आया ही नहीं",
        "எனக்கு ஓடிபி வரவில்லை",
        "ஓடிபியை மீண்டும் அனுப்பவும்",
        "ஓடிபி இன்னும் வரவில்லை",
        "నాకు ఓటీపీ రాలేదు",
        "ఓటీపీ మళ్ళీ పంపండి",
        "ఓటీపీ ఇంకా రాలేదు",
    ],
    "fraud": [
        "someone used my account, block it",
        "mera account hack ho gaya",
        "fraud transaction on my account",
        "मेरे अकाउंट से पैसे कट गए बिना मेरी जानकारी के",
        "unauthorized transaction, block now",
        "मेरे खाते से बिना बताए पैसे कट गए",
        "किसी ने मेरा अकाउंट हैक कर लिया, ब्लॉक करो",
        "फ्रॉड ट्रांजैक्शन हुआ है, अकाउंट ब्लॉक करो",
        "என் கணக்கிலிருந்து தெரியாமல் பணம் கழிந்துள்ளது",
        "யாரோ என் கணக்கை ஹேக் செய்துள்ளனர், பிளாக் செய்யுங்கள்",
        "மோசடி பரிவர்த்தனை நடந்துள்ளது, உடனே பிளாக் செய்யவும்",
        "నా ఖాతా నుంచి తెలియకుండా డబ్బులు కట్ అయ్యాయి",
        "ఎవరో నా ఖాతా హ్యాక్ చేశారు, బ్లాక్ చేయండి",
        "మోసపూరిత లావాదేవీ జరిగింది, వెంటనే బ్లాక్ చేయండి",
    ],
    "account_closure": [
        "close my account",
        "delete my account",
        "खाता बंद करो",
        "how do I close my account",
        "मेरा अकाउंट बंद कर दो",
        "अकाउंट डिलीट कैसे करूं",
        "खाता हमेशा के लिए बंद करना है",
        "என் கணக்கை மூடவும்",
        "கணக்கை எப்படி நீக்குவது",
        "கணக்கை நிரந்தரமாக மூட வேண்டும்",
        "నా ఖాతా మూసివేయండి",
        "ఖాతా ఎలా డిలీట్ చేయాలి",
        "ఖాతా శాశ్వతంగా మూసివేయాలి",
    ],
    "delivery_delay": [
        "my order is late, where is it",
        "delivery bahut late ho rahi hai",
        "order delay hone par kya karein",
        "why is my delivery delayed",
        "डिलीवरी बहुत देर से है",
        "ऑर्डर लेट हो रहा है",
        "इतनी देर क्यों हो रही है",
        "டெலிவரி மிகவும் தாமதமாக உள்ளது",
        "ஆர்டர் தாமதமாகிறது",
        "இவ்வளவு நேரம் ஏன் ஆகிறது",
        "డెలివరీ చాలా ఆలస్యంగా ఉంది",
        "ఆర్డర్ లేట్ అవుతోంది",
        "ఇంత ఆలస్యం ఎందుకు",
    ],
    "product_info": [
        "tell me about this product",
        "product ki jankari do",
        "is this item in stock",
        "product specifications",
        "इस प्रोडक्ट की जानकारी दो",
        "यह आइटम स्टॉक में है क्या",
        "प्रोडक्ट की स्पेसिफिकेशन बताओ",
        "இந்த பொருள் பற்றிய தகவல் வேண்டும்",
        "இந்த பொருள் கடையில் உள்ளதா",
        "பொருளின் விவரக்குறிப்புகளைச் சொல்லுங்கள்",
        "ఈ ప్రొడక్ట్ గురించి సమాచారం ఇవ్వండి",
        "ఈ ఐటమ్ స్టాక్‌లో ఉందా",
        "ప్రొడక్ట్ స్పెసిఫికేషన్స్ చెప్పండి",
    ],
    "invoice": [
        "I need my invoice",
        "invoice kaise milega",
        "send me the bill receipt",
        "download my invoice",
        "मुझे इनवॉइस चाहिए",
        "बिल की रसीद भेजो",
        "इनवॉइस डाउनलोड कैसे करूं",
        "எனக்கு இன்வாய்ஸ் வேண்டும்",
        "பில் ரசீதை அனுப்பவும்",
        "இன்வாய்ஸை எப்படி டவுன்லோட் செய்வது",
        "నాకు ఇన్వాయిస్ కావాలి",
        "బిల్లు రశీదు పంపండి",
        "ఇన్వాయిస్ డౌన్‌లోడ్ ఎలా చేయాలి",
    ],
    "plan_change": [
        "change my mobile plan",
        "plan badalna hai",
        "upgrade my plan",
        "switch to a cheaper plan",
        "मेरा प्लान बदलना है",
        "सस्ते प्लान पर शिफ्ट करो",
        "प्लान अपग्रेड करना है",
        "என் திட்டத்தை மாற்ற வேண்டும்",
        "மலிவான திட்டத்திற்கு மாறவும்",
        "திட்டத்தை மேம்படுத்த வேண்டும்",
        "నా ప్లాన్ మార్చాలి",
        "తక్కువ ధర ప్లాన్‌కి మార్చండి",
        "ప్లాన్ అప్‌గ్రేడ్ చేయాలి",
    ],
    "roaming": [
        "international roaming not working",
        "roaming charges kya hain",
        "enable roaming",
        "roaming pack activate karo",
        "रोमिंग काम नहीं कर रहा",
        "रोमिंग चार्ज कितने हैं",
        "रोमिंग पैक चालू करो",
        "ரோமிங் வேலை செய்யவில்லை",
        "ரோமிங் கட்டணம் என்ன",
        "ரோமிங் தொகுப்பை செயல்படுத்தவும்",
        "రోమింగ్ పని చేయడం లేదు",
        "రోమింగ్ ఛార్జీలు ఎంత",
        "రోమింగ్ ప్యాక్ యాక్టివేట్ చేయండి",
    ],
    "network_issue": [
        "network is down",
        "network nahi chal raha",
        "internet not working",
        "no signal on my phone",
        "नेटवर्क आ नहीं रहा",
        "मेरे फोन में सिग्नल नहीं है",
        "इंटरनेट चल नहीं रहा है",
        "நெட்வொர்க் வரவில்லை",
        "என் போனில் சிக்னல் இல்லை",
        "இணையம் வேலை செய்யவில்லை",
        "నెట్‌వర్క్ రావడం లేదు",
        "నా ఫోన్‌లో సిగ్నల్ లేదు",
        "ఇంటర్నెట్ పని చేయడం లేదు",
    ],
    "complaint": [
        "I want to file a complaint",
        "shikayat karni hai",
        "complaint register karo",
        "I am unhappy with the service",
        "मैं शिकायत दर्ज करवाना चाहता हूं",
        "सर्विस ठीक नहीं है, शिकायत करनी है",
        "मुझे आपकी सर्विस पसंद नहीं आई",
        "எனக்கு புகார் பதிவு செய்ய வேண்டும்",
        "சேவை சரியில்லை, புகார் அளிக்கிறேன்",
        "உங்கள் சேவை எனக்கு பிடிக்கவில்லை",
        "నేను ఫిర్యాదు చేయాలనుకుంటున్నాను",
        "సర్వీస్ సరిగ్గా లేదు, ఫిర్యాదు చేయాలి",
        "మీ సర్వీస్ నాకు నచ్చలేదు",
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
        "मुझे 25000 रुपये का रिफंड चाहिए",
        "50000 का रिफंड अब तक नहीं मिला",
        "30000 की बड़ी रकम वापस करो",
        "எனக்கு 25000 ரூபாய் ரீஃபண்ட் வேண்டும்",
        "50000 ரீஃபண்ட் இன்னும் கிடைக்கவில்லை",
        "30000 பெரிய தொகையை திரும்பத் தாருங்கள்",
        "నాకు 25000 రూపాయల రీఫండ్ కావాలి",
        "50000 రీఫండ్ ఇంకా రాలేదు",
        "30000 పెద్ద మొత్తం వాపస్ చేయండి",
    ],
    # M5c: informational questions about refund timing — NOT refund requests.
    # Without these, "refund kitne din me aata hai?" misroutes to
    # high_value_refund -> ESCALATE.
    "refund_info": [
        "refund kitne din me aata hai",
        "when will I get my refund back",
        "paise kab wapas milenge",
        "how long does a refund take",
        "how many days for the refund",
        "refund kab tak aayega",
        "mera refund kab milega",
        "when is my money refunded",
        "रिफंड कितने दिन में मिलता है",
        "पैसा वापस आने में कितना समय लगता है",
        "रिफंड कब तक प्रोसेस होगा",
        "ரீஃபண்ட் எத்தனை நாட்களில் கிடைக்கும்",
        "பணம் திரும்ப எவ்வளவு நேரம் ஆகும்",
        "ரீஃபண்ட் எப்போது வரும்",
        "రీఫండ్ ఎన్ని రోజుల్లో వస్తుంది",
        "డబ్బు వాపస్ రావడానికి ఎంత సమయం పడుతుంది",
        "రీఫండ్ ఎప్పుడు ప్రాసెస్ అవుతుంది",
    ],
    # M5c: informational ETA questions about a pending delivery — distinct
    # from order_status ("where is it") and delivery_delay ("it is late").
    "delivery_eta": [
        "when will my order be delivered",
        "order kab tak aayega",
        "delivery kab hogi",
        "how many days will delivery take",
        "when will my order arrive",
        "mera order kab milega",
        "delivery date kya hai",
        "when can I expect my delivery",
        "मेरा ऑर्डर कितने दिन में पहुंचेगा",
        "डिलीवरी कब तक होगी",
        "ऑर्डर की डिलीवरी डेट क्या है",
        "என் ஆர்டர் எத்தனை நாட்களில் வரும்",
        "டெலிவரி எப்போது நடைபெறும்",
        "ஆர்டர் டெலிவரி தேதி என்ன",
        "నా ఆర్డర్ ఎన్ని రోజుల్లో వస్తుంది",
        "ఆర్డర్ ఎప్పుడు డెలివరీ అవుతుంది",
        "ఆర్డర్ డెలివరీ తేదీ ఏమిటి",
    ],
    "reschedule_delivery": [
        "I want to reschedule my delivery",
        "reschedule my order delivery",
        "can I change the delivery date",
        "deliver on another date",
        "reschedule my order",
        "change delivery date",
        "delivery reschedule karni hai",
        "order dusre din deliver karo",
        "delivery date change karni hai",
        "mera order kal nahi parso deliver karna",
        "डिलीवरी रिशेड्यूल करनी है",
        "डिलीवरी की तारीख बदलें",
        "ऑर्डर दूसरे दिन डिलीवर करें",
        "டெலிவரி தேதியை மாற்ற வேண்டும்",
        "எனது ஆர்டரை மற்றொரு நாள் டெலிவரி செய்யவும்",
        "டெலிவரி தேதியை மாற்றிக் கொள்ளலாமா",
        "డెలివరీ తేదీ మార్చండి",
        "నా ఆర్డర్ డెలివరీ మరో రోజుకు కావాలి",
        "డెలివరీ తేదీ మార్చుకోవచ్చా",
    ],
}


# --- demo ERP fixture (moved out of tools.py MockERP) -----------------
DEMO_ERP_ORDERS: dict = {
            "ORD-4821": {
                "order_id": "ORD-4821", "customer_id": "CUST-001",
                "status": "CONFIRMED", "amount": 1299.0,
                "items": ["Running Shoes"], "delivery_date": "2026-09-05",
                "address": "12 MG Road, Bangalore",
            },
            "ORD-7734": {
                "order_id": "ORD-7734", "customer_id": "CUST-001",
                "status": "SHIPPED", "amount": 6500.0,
                "items": ["Smart Watch"], "delivery_date": "2026-09-02",
                "tracking_url": "https://track.fake/7734",
            },
            # PizzaPal-flavored demo orders (CUST-002) so a phone lookup on
            # the demo customer's own number returns a coherent pizza order.
            "ORD-9021": {
                "order_id": "ORD-9021", "customer_id": "CUST-002",
                "status": "CONFIRMED", "amount": 749.0,
                "items": ["Large Pepperoni Pizza", "Garlic Bread"],
                "delivery_date": "2026-09-06",
                "address": "21 Koramangala 5th Block, Bangalore",
            },
            "ORD-9022": {
                "order_id": "ORD-9022", "customer_id": "CUST-002",
                "status": "DELIVERED", "amount": 429.0,
                "items": ["Medium Margherita"],
                "delivery_date": "2026-09-04",
                "address": "21 Koramangala 5th Block, Bangalore",
            },
        }
DEMO_ERP_CUSTOMERS: dict = {
            "CUST-001": {"name": "Shagil", "phone": "+91-9876543210",
                         "orders": ["ORD-4821", "ORD-7734"]},
            "CUST-002": {"name": "Demo PizzaPal Customer",
                         "phone": "+91-9828379313",
                         "orders": ["ORD-9021", "ORD-9022"]},
        }

# The default (demo) deployment name — domain data, not core config.
DEMO_DEPLOYMENT_NAME = "acme_support"
