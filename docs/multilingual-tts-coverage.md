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

## Known limits (upstream gap, not platform)

- **gu / kn / pa / ta**: rhasspy/piper-voices ships NO voice for Gujarati,
  Kannada, Punjabi, Tamil (verified via HF repo tree API 2026-09-15). ASR
  fully works for these (IndicConformer); replies fall back to the en
  fallback voice with a warning. Adding a voice later = one `tts_voice:`
  data line, zero code.
- **it**: upstream only ships it_IT-riccardo-x_low (no medium); declared
  policy is to wait for a medium voice rather than ship x_low quality
  (see data/lang/it.yaml).
