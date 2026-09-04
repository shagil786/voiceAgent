# tests/test_livekit_audio.py
import math, struct
from voiceagent.telephony.audio import (
    chunk_frames, resample_16k_to_48k, resample_48k_to_16k)

def _sine_16k(ms=100, hz=440):
    n = 16 * ms
    return struct.pack(f"<{n}h", *[int(3000 * math.sin(2 * math.pi * hz * i / 16000)) for i in range(n)])

def test_roundtrip_preserves_tone():
    tone = _sine_16k()
    up = resample_16k_to_48k(tone)
    assert len(up) == len(tone) * 3
    back = resample_48k_to_16k(up)
    assert len(back) == len(tone)
    import struct as st
    amps = st.unpack(f"<{len(back)//2}h", back)
    assert max(abs(a) for a in amps) > 1000  # tone survived

def test_chunking_exact():
    assert len(chunk_frames(_sine_16k(ms=100), 20, 16000)) == 5

def test_config_livekit_fields():
    from voiceagent.config import load_config
    c = load_config(env={"LIVEKIT_URL": "wss://x", "LIVEKIT_ROOM_PREFIX": "call-"})
    assert c.livekit_url == "wss://x" and c.livekit_number is None
