# Voicefield — VoiceAgent dashboard background

The dashboard background **is the agent's state of mind**: a field of waveform
ribbons that breathes through the agent's actual rhythm —
**LISTEN → THINK → SPEAK → LISTEN** — with syllable bursts traveling through
the field, phoneme particles riding the waves, and sonar rings blooming where
things happen. Indigo → violet → cyan on an aurora-washed near-white (or
midnight) ground, quiet enough to sit under real dashboard cards.

Two halves, one visual:

| File | What it is |
|---|---|
| `render.py` | Offline renderer → `assets/voicefield-loop.mp4` (1920×1080@30), `assets/voicefield-loop.gif` (800×450@15), `assets/poster.png` |
| `voicefield.js` | The **live canvas twin** — same math every frame, plus interaction |
| `index.html` | Demo: mock VoiceAgent console over the live field |

## Why both video and canvas

A GIF/MP4 is a dead loop — you can't interact with it. `voicefield.js` is the
same scene evaluated live, so the background can *respond*:

- **pointer move** — the field bulges softly under the cursor
- **click / tap** — a sonar ring is born on the nearest ribbon
- **mic (opt-in)** — wave amplitude rides the *real* microphone level
- **`field.pulse(x)`** — programmatic sonar; fire it when a call lands, a
  decision is logged, a tool gets approved — the background becomes a
  notification channel

The loop math is identical in both (every time term is `sin(2π·k·t/T)`,
integer `k`), so frame 0 equals frame T exactly — the MP4/GIF loops seamlessly,
and live mode shares the same 8-second "breathing" rhythm.

## Use in the real console

```html
<script src="voicefield.js"></script>
<script>
  const field = VoiceField.mount(document.body, { theme: 'light' });

  // theme sync with your app
  field.setTheme('dark');

  // sonar on real dashboard events
  socket.on('decision.logged', () => field.pulse(Math.random()));

  // optional: the waves listen to the room
  field.enableMic(true);          // asks for mic permission

  // agent-state pill (LISTENING / THINKING / SPEAKING)
  setInterval(() => el.textContent = field.state, 400);
</script>
```

- The canvas is `position:fixed; inset:0; pointer-events:none` — it never
  blocks dashboard UI.
- Honors `prefers-reduced-motion` (renders one static hero frame).
- Pauses when the tab is hidden; caps DPR at 2.
- No dependencies, ~11 KB.

## Video/GIF usage

```html
<video autoplay muted loop playsinline poster="assets/poster.png"
       style="position:fixed; inset:0; width:100%; height:100%; object-fit:cover">
  <source src="assets/voicefield-loop.mp4" type="video/mp4">
</video>
```

MP4 ≈ 2–4 MB; GIF ≈ 6–10 MB (prefer the MP4 or the canvas everywhere except
email/docs that refuse `<video>`).

## Rebuild assets

```bash
pip install pillow numpy imageio imageio-ffmpeg
python3 render.py                # → assets/ (MP4 + GIF + poster)
python3 render.py --skip-gif     # MP4 + poster only
```

Design tokens live at the top of both files (`LIGHT` / `THEMES.light`) — keep
them in sync when retheming.
