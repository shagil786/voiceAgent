# src/voiceagent/intent.py
"""Deterministic intent classifier built on the same multilingual embeddings
used for RAG. The action decision is a nearest-neighbour match against curated
exemplar queries — no LLM involved, so it cannot drift format or reason itself
into the wrong action the way a small generative model does.

Output is constrained to the fixed intent vocabulary by construction.

M5a: each intent carries 3 Hindi (Devanagari), 3 Tamil and 3 Telugu
exemplars so native-script queries match same-language neighbours. M5a-2
swapped the default embedder to LaBSE (768-dim, 109 languages): native-script
resolution jumped (bn 0.300->1.000, te 0.700->1.000, gu 0.800->1.000,
ta 0.533->0.967) mostly via LaBSE cross-lingual transfer onto the existing
en/hinglish/hi/ta/te exemplars — the planned 3 Bengali, 3 Gujarati and 3
Marathi exemplars per intent were NOT added (mr/gu/bn eval rows resolve on
transfer alone). M5b: hybrid routing — the M5a-2 sweep also showed LaBSE
COLLAPSES on Romanized code-mixed Hindi (hinglish 0.993->0.700) while MiniLM
handles it well, so the classifier keeps TWO exemplar matrices over the SAME
exemplar strings: MiniLM-encoded for en/hinglish queries, LaBSE-encoded for
native-script queries, routed by script per classify() call. NOTE: the non-en
exemplars (Hinglish, hi/ta/te) are LLM-authored SYNTHETIC phrasings, not
transcripts of real customers — plausible support language, but real-traffic
validation is still pending (quality caveat).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from voiceagent.knowledge import (DEFAULT_EMBEDDER, LATIN_SPACE, NATIVE_SPACE,
                                  SPACE_EMBEDDERS, route_space)

INTENT_EXEMPLARS: dict[str, list[str]] = {
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


class IntentClassifier:
    """M5b hybrid: TWO exemplar matrices over the SAME exemplar strings —
    one LaBSE-encoded (native space) for native-script queries, one
    MiniLM-encoded (latin space) for en/hinglish queries. classify() routes
    by detect_language via knowledge.route_space (same routing rule as
    IndexHandle.search, so RAG and intent always agree on the space).

    Both encoders are constructed EAGERLY at init (~2-4s total, both models
    are small and already cached): no first-query latency cliff for either
    script family, and no lazy-init state to reason about in the voice
    server. The matrices themselves are tiny (~350 exemplars x dim)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER,
                 latin_model_name: str = SPACE_EMBEDDERS[LATIN_SPACE],
                 exemplars: dict[str, list[str]] | None = None):
        # model_name keeps its historical meaning: the native-script-space
        # encoder (primary). The latin-space encoder is latin_model_name.
        # exemplars: per-tenant intent exemplars (tenant bundle, M6b);
        # None -> the built-in INTENT_EXEMPLARS.
        self._exemplars = exemplars if exemplars is not None else INTENT_EXEMPLARS
        self._native_model = SentenceTransformer(model_name)
        self._latin_model = SentenceTransformer(latin_model_name)
        self._intents: list[str] = []
        # space -> (exemplar matrix, labels); both spaces cover every intent
        self._matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._build()

    def _build(self) -> None:
        queries: list[str] = []
        labels: list[str] = []
        for intent, exs in self._exemplars.items():
            for ex in exs:
                queries.append(ex)
                labels.append(intent)
        self._intents = list(self._exemplars.keys())
        for space, model in ((NATIVE_SPACE, self._native_model),
                             (LATIN_SPACE, self._latin_model)):
            emb = np.asarray(model.encode(queries, normalize_embeddings=True),
                             dtype=np.float32)
            self._matrices[space] = (emb, list(labels))

    def classify(self, text: str, k: int = 1) -> tuple[str, float]:
        """Return (best_intent, cosine_score), comparing the query against
        the exemplar matrix of the space matched to its script."""
        space = route_space(text)
        embs, labels = self._matrices[space]
        model = (self._native_model if space == NATIVE_SPACE
                 else self._latin_model)
        q = np.asarray(model.encode([text], normalize_embeddings=True),
                       dtype=np.float32)
        scores = embs @ q.T  # (n_exemplars, 1)
        scores = scores[:, 0]
        order = np.argsort(-scores)[:k]
        best = int(order[0])
        return labels[best], float(scores[best])
