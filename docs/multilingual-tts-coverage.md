# Multilingual TTS voice coverage (2026-09-15)

The voice registry is data-driven (`data/lang/*.yaml` `tts_voice:`); voices
download on demand from rhasspy/piper-voices. This audit pre-warmed every
declared voice and verified synthesis per language.

## Declared + verified speaking (14 languages)

| Lang | Voice | Verified |
|---|---|---|
| en | en_US-lessac-medium | (pre-existing) |
| hi | hi_IN-priyamvada-medium | (pre-existing) |
| es | es_ES-sharvard-medium | (pre-existing) |
| te | te_IN-maya-medium | (pre-existing) |
| th | th_TH-tsync2-medium | (pre-existing) |
| ar | ar_JO-kareem-medium | 81152 frames |
| bn | bn_BD-google-medium | 53248 frames |
| ml | ml_IN-meera-medium | 46336 frames |
| mr | mr_IN-google-medium | 46848 frames |
| ur | ur_PK-fasih-medium | 39424 frames |
| de | de_DE-thorsten-medium | 39424 frames |
| fr | fr_FR-siwis-medium | 35584 frames |
| pt | pt_BR-faber-medium | 30720 frames |
| hinglish | alias_of: hi | (by design) |

Sample WAVs: data/out/voice-multiling-smoke/

## MMS-TTS backend (2026-09-15): the piper gap is CLOSED

gu / kn / pa / ta now speak via Meta MMS-TTS (facebook/mms-tts-{tam,kan,pan,guj},
VITS via transformers — already a dependency, ungated). Declared as
`tts_voice: mms:<iso>` in the lang files; `TTSHandle._get_voice` routes
the prefix to the MMS loader, which adapts VITS to the piper voice
contract (synthesize_wav). Verified: ta/kn/pa/gu synthesize real speech
(~0.5-1.4s CPU for a sentence, 16kHz mono; first model load ~15s then
HF-cached). Samples: data/out/voice-multiling-smoke/{ta,kn,pa,gu}-mms.wav

## Known limits (remaining, upstream)

- **it**: upstream piper only ships it_IT-riccardo-x_low (no medium);
  declared policy is to wait for a medium voice rather than ship x_low
  quality (see data/lang/it.yaml). Could also route to mms:eng-style
  Italian (facebook/mms-tts-ita) if x_low quality stays unacceptable —
  one data line when decided.
- MMS voices are 16kHz mono (piper medium is 22kHz) — slightly lower
  fidelity, acceptable for telephony (8kHz trunk anyway).
