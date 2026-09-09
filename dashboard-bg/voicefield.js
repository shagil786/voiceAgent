/**
 * voicefield.js — the live, interactive twin of dashboard-bg/render.py.
 *
 * Same math (envelope, ribbons, syllable bursts, sonar ripples) evaluated on a
 * <canvas> every frame. Drop-in dashboard background:
 *
 *   <script src="voicefield.js"></script>
 *   <script>
 *     const field = VoiceField.mount(document.body);   // behind your UI
 *     // field.setTheme('dark') | .pulse(0.7) | .enableMic(true) | .destroy()
 *   </script>
 *
 * Interactions
 *   - pointer move: field bends away/toward the cursor (soft gaussian force)
 *   - click/tap: sonar ring ripples out from the nearest ribbon
 *   - mic (opt-in): wave amplitude rides the REAL microphone level
 *   - VoiceField.pulse(x): programmatic sonar — call it when a decision,
 *     call, or approval lands in the dashboard
 *
 * Loop math is identical to the MP4/GIF (all terms periodic in T=8s), so live
 * mode and video mode stay visually in sync in style and rhythm.
 */
(function (global) {
  "use strict";

  var T = 8.0, RIBBONS = 7, TAU = Math.PI * 2;

  var THEMES = {
    light: {
      bgTop: [250, 251, 255], bgBottom: [242, 243, 252],
      vignette: [237, 239, 252],
      auroraA: [224, 231, 255], auroraB: [232, 225, 253], auroraA2: [219, 244, 253],
      lattice: [99, 102, 241], latticeA: 16,
      stops: [[99, 102, 241], [139, 92, 246], [34, 211, 238]],
      lineA: 0.69, glowA: 0.16, dotA: 0.82
    },
    dark: {
      bgTop: [12, 15, 30], bgBottom: [6, 8, 19],
      vignette: [22, 26, 52],
      auroraA: [38, 44, 96], auroraB: [52, 38, 100], auroraA2: [20, 60, 92],
      lattice: [148, 163, 255], latticeA: 22,
      stops: [[129, 140, 248], [167, 139, 250], [34, 211, 238]],
      lineA: 0.76, glowA: 0.22, dotA: 0.92
    }
  };

  // ------------------------------------------------------------ scene math --
  function envelope(t, phase) {
    var th = TAU * (t / T) + (phase || 0);
    var e = 0.42 + 0.30 * Math.sin(th) + 0.20 * Math.sin(2 * th + 1.7) +
            0.12 * Math.sin(3 * th + 4.2);
    return Math.max(0.12, Math.min(1.0, e));
  }
  function centerWeight(i) {
    var x = i / (RIBBONS - 1);
    return 0.35 + 0.65 * Math.pow(Math.sin(Math.PI * x), 1.5);
  }
  function xenv(u, t, seed) {
    var th = TAU * (t / T);
    var a = 0.55 + 0.45 * Math.sin(TAU * 2.0 * u - 2 * th + seed);
    var b = 0.70 + 0.30 * Math.sin(TAU * 3.0 * u + 3 * th + 1.3 * seed);
    var m = 0.62 * a + 0.38 * b;
    return Math.max(0.06, Math.min(1.55, 0.22 + 1.15 * Math.pow(m, 1.7)));
  }
  function ribbonY(i, u, t, W, H) {
    var th = TAU * (t / T);
    var seed = i * 2.399963;
    var yBase = H * (0.14 + 0.72 * i / (RIBBONS - 1)) + H * 0.018 * Math.sin(th + seed);
    var amp = H * (0.030 + 0.135 * centerWeight(i)) * (0.7 + 0.3 * Math.sin(seed * 1.9));
    var env = envelope(t, seed);
    var xe = xenv(u, t, seed);
    var wave =
      0.46 * Math.sin(TAU * (1.6 * u + 0.07 * i) + 3 * th + seed) +
      0.30 * Math.sin(TAU * (3.7 * u + 0.19 * i) - 5 * th + 2.1 * i) +
      0.24 * xe * Math.sin(TAU * (17.0 * u + 0.31 * i) + 9 * th + 4.7 * i);
    return yBase + amp * env * wave * xe;
  }
  function colorAt(u, stops) {
    var n = stops.length - 1;
    var s = Math.max(0, Math.min(1, u)) * n;
    var k = Math.min(Math.floor(s), n - 1), f = s - k;
    var a = stops[k], b = stops[k + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function rgba(c, a) {
    return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + a + ")";
  }

  // ---------------------------------------------------------------- mount --
  function mount(el, opts) {
    opts = opts || {};
    var host = (typeof el === "string") ? document.querySelector(el) : el;
    if (!host) throw new Error("VoiceField.mount: no host element");

    var canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");
    var s = canvas.style;
    s.position = "fixed"; s.inset = "0";
    s.width = "100%"; s.height = "100%";
    s.zIndex = opts.zIndex || "0";
    s.pointerEvents = "none";              // never blocks dashboard UI
    (host === document.body ? document.body : host).appendChild(canvas);
    if (host !== document.body && getComputedStyle(host).position === "static") {
      host.style.position = "relative";
    }

    var ctx = canvas.getContext("2d");
    var themeName = opts.theme ||
      (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark" : "light");
    var W = 0, H = 0, DPR = 1;
    var latticeCanvas = null;
    var start = performance.now();
    var reduced = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // interaction state
    var px = -1e4, py = -1e4, pEnergy = 0;          // pointer
    var clickRipples = [];                          // {x,y,birth}
    var mic = { enabled: false, stream: null, analyser: null, data: null, level: 0 };
    var paused = false, rafId = 0;

    function resize() {
      DPR = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.clientWidth || window.innerWidth;
      H = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.round(W * DPR);
      canvas.height = Math.round(H * DPR);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

      // dot lattice, pre-rendered
      latticeCanvas = document.createElement("canvas");
      latticeCanvas.width = canvas.width; latticeCanvas.height = canvas.height;
      var lc = latticeCanvas.getContext("2d");
      lc.setTransform(DPR, 0, 0, DPR, 0, 0);
      var th = THEMES[themeName];
      var step = Math.max(34, W / 46);
      lc.fillStyle = rgba(th.lattice, th.latticeA / 255);
      for (var gy = step / 2; gy < H; gy += step) {
        for (var gx = step / 2; gx < W; gx += step) {
          lc.beginPath(); lc.arc(gx, gy, 1, 0, TAU); lc.fill();
        }
      }
    }
    resize();
    window.addEventListener("resize", resize);

    // ------------------------------------------------------------ input -----
    function onMove(e) {
      var x = e.touches ? e.touches[0].clientX : e.clientX;
      var y = e.touches ? e.touches[0].clientY : e.clientY;
      px = x; py = y; pEnergy = Math.min(1, pEnergy + 0.25);
    }
    function onDown(e) {
      var x = e.touches ? e.touches[0].clientX : e.clientX;
      var y = e.touches ? e.touches[0].clientY : e.clientY;
      // snap to nearest ribbon's y at that x — ring is born "on" the wave
      var u = x / W, ri = Math.round(((y / H) - 0.14) / 0.72 * (RIBBONS - 1));
      ri = Math.max(0, Math.min(RIBBONS - 1, ri));
      var ry = ribbonY(ri, u, nowT(), W, H);
      clickRipples.push({ x: x, y: ry, birth: performance.now(), ribbon: ri });
      if (clickRipples.length > 8) clickRipples.shift();
    }
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("touchmove", onMove, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });

    // ------------------------------------------------------------- mic ------
    function enableMic(yes) {
      if (!yes) {
        if (mic.stream) mic.stream.getTracks().forEach(function (t) { t.stop(); });
        mic.stream = null; mic.analyser = null; mic.enabled = false;
        return Promise.resolve(false);
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return Promise.reject(new Error("getUserMedia unavailable"));
      }
      return navigator.mediaDevices.getUserMedia({ audio: true }).then(function (st) {
        var AC = window.AudioContext || window.webkitAudioContext;
        var ac = new AC();
        var src = ac.createMediaStreamSource(st);
        var an = ac.createAnalyser();
        an.fftSize = 512; an.smoothingTimeConstant = 0.75;
        src.connect(an);
        mic.stream = st; mic.analyser = an;
        mic.data = new Uint8Array(an.frequencyBinCount);
        mic.enabled = true;
        return true;
      });
    }
    function micLevel() {
      if (!mic.enabled || !mic.analyser) return 0;
      mic.analyser.getByteFrequencyData(mic.data);
      var sum = 0, n = mic.data.length;
      for (var i = 0; i < n; i++) sum += mic.data[i] * mic.data[i];
      var rms = Math.sqrt(sum / n) / 255;
      mic.level = mic.level * 0.7 + rms * 0.3;     // smooth
      return mic.level;
    }

    // ------------------------------------------------------------ paint -----
    function nowT() { return ((performance.now() - start) / 1000) % T; }

    function paint(t) {
      var th = THEMES[themeName];
      var th2 = TAU * (t / T);
      var mL = micLevel();

      // background
      var g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, rgba(th.bgTop, 1));
      g.addColorStop(1, rgba(th.bgBottom, 1));
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);

      // aurora blobs
      var blobs = [
        [0.28 + 0.04 * Math.sin(th2), 0.46 + 0.05 * Math.sin(2 * th2 + 1.0), 0.42, th.auroraA],
        [0.70 + 0.05 * Math.sin(2 * th2 + 2.5), 0.34 + 0.06 * Math.sin(th2 + 3.1), 0.36, th.auroraB],
        [0.52 + 0.06 * Math.sin(3 * th2 + 4.0), 0.68 + 0.04 * Math.sin(2 * th2 + 5.2), 0.34, th.auroraA2]
      ];
      for (var bi = 0; bi < blobs.length; bi++) {
        var bx = blobs[bi][0] * W, by = blobs[bi][1] * H, br = blobs[bi][2] * W;
        var rg = ctx.createRadialGradient(bx, by, 0, bx, by, br);
        rg.addColorStop(0, rgba(blobs[bi][3], 0.85));
        rg.addColorStop(1, rgba(blobs[bi][3], 0));
        ctx.fillStyle = rg;
        ctx.fillRect(0, 0, W, H);
      }

      // vignette
      var vg = ctx.createRadialGradient(W * 0.5, H * 0.46, 0, W * 0.5, H * 0.46,
        Math.max(W, H) * 0.62);
      vg.addColorStop(0, rgba(th.vignette, 0.9));
      vg.addColorStop(1, rgba(th.vignette, 0));
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, W, H);

      // lattice
      ctx.drawImage(latticeCanvas, 0, 0, W, H);

      // pointer influence decays when idle
      pEnergy *= 0.97;

      var step = Math.max(4, Math.floor(W / 240));   // ~240 segments
      var ri, u, x, y, c;

      // glow pass (wide, translucent) then core pass
      for (var pass = 0; pass < 2; pass++) {
        ctx.lineJoin = "round";
        ctx.lineWidth = pass === 0 ? 9 : 2.4;
        for (ri = 0; ri < RIBBONS; ri++) {
          ctx.beginPath();
          for (x = 0; x <= W + step; x += step) {
            u = x / W;
            y = ribbonY(ri, u, t, W, H);
            // live mic: swell the middle ribbons with actual voice level
            var boost = mL * (0.35 + 0.5 * centerWeight(ri)) * H * 0.16;
            y += boost * Math.sin(TAU * (6.0 * u) + 11 * th2 + ri) * xenv(u, t, ri * 2.399963);
            // pointer: a soft bulge under the cursor — "a hand parting the field"
            var dxn = (x - px) / 150;
            y -= pEnergy * H * 0.045 * Math.exp(-dxn * dxn) *
                 Math.exp(-Math.pow((y - py) / (H * 0.22), 2));
            if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          c = colorAt((ri + 0.5) / RIBBONS, th.stops);
          ctx.strokeStyle = rgba(c, pass === 0 ? th.glowA : th.lineA);
          ctx.stroke();
        }
      }

      // particles riding the ribbons
      var DOTS = 46;
      for (var d = 0; d < DOTS; d++) {
        var f = (d * 0.61803398875) % 1;
        var dri = d % RIBBONS;
        var laps = 1 + (d % 2);
        var du = ((f + laps * (t / T)) % 1);
        var dxp = du * W;
        var dyp = ribbonY(dri, du, t, W, H) +
          mL * (0.35 + 0.5 * centerWeight(dri)) * H * 0.16 *
          Math.sin(TAU * 6.0 * du + 11 * th2 + dri) * xenv(du, t, dri * 2.399963);
        var tw = 0.55 + 0.45 * Math.sin(th2 * (1 + d % 3) + d);
        var r = 2.0 + 1.8 * ((d * 7) % 5) / 4;
        c = colorAt(dxp / W, th.stops);
        var a = th.dotA * tw;
        ctx.beginPath(); ctx.arc(dxp, dyp, r * 3.2, 0, TAU);
        ctx.fillStyle = rgba(c, 0.22 * a); ctx.fill();
        ctx.beginPath(); ctx.arc(dxp, dyp, r, 0, TAU);
        ctx.fillStyle = rgba(c, a); ctx.fill();
      }

      // sonar rings: scripted + user clicks
      var LIFE = 2.1, now = performance.now(), k;
      var scripted = [[0.16 * T, 0.24, 1], [0.58 * T, 0.76, 5]];
      for (k = 0; k < scripted.length; k++) {
        var tb = scripted[k][0];
        if (t >= tb && t < tb + 0.26 * T) {
          var p = (t - tb) / (0.26 * T);
          var sx = scripted[k][1] * W;
          var sy = ribbonY(scripted[k][2], scripted[k][1], t, W, H);
          drawRing(sx, sy, (0.05 + 0.20 * p) * W, 1 - p, colorAt(scripted[k][1], th.stops));
        }
      }
      for (k = clickRipples.length - 1; k >= 0; k--) {
        var rp = clickRipples[k];
        var q = (now - rp.birth) / (LIFE * 1000);
        if (q >= 1) { clickRipples.splice(k, 1); continue; }
        var ease = 1 - Math.pow(1 - q, 2.2);          // ease-out
        drawRing(rp.x, rp.y, (0.03 + 0.30 * ease) * W, 1 - q,
                 colorAt(rp.x / W, th.stops), mL);
      }

      // pointer halo (subtle cue that the field sees you)
      if (pEnergy > 0.02 && px > 0) {
        var hg = ctx.createRadialGradient(px, py, 0, px, py, 90);
        c = colorAt(px / W, th.stops);
        hg.addColorStop(0, rgba(c, 0.10 * pEnergy));
        hg.addColorStop(1, rgba(c, 0));
        ctx.fillStyle = hg;
        ctx.fillRect(px - 90, py - 90, 180, 180);
      }

      return mL;
    }

    function drawRing(x, y, r, alpha, col, boost) {
      var ry = r * 0.62;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = rgba(col, 0.45 * alpha);
      ctx.beginPath(); ctx.ellipse(x, y, r, ry, 0, 0, TAU); ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = rgba(col, 0.22 * alpha);
      ctx.beginPath(); ctx.ellipse(x, y, r * 0.55, ry * 0.55, 0, 0, TAU); ctx.stroke();
      if (boost) {
        ctx.strokeStyle = rgba(col, 0.30 * alpha * boost * 3);
        ctx.beginPath(); ctx.ellipse(x, y, r * 1.35, ry * 1.35, 0, 0, TAU); ctx.stroke();
      }
    }

    // ------------------------------------------------------------- loop -----
    var lastLevel = 0;
    function frame() {
      rafId = 0;
      if (paused) return;
      var mL = paint(nowT());
      lastLevel = mL;
      rafId = requestAnimationFrame(frame);
    }
    function kick() { if (!rafId && !paused) rafId = requestAnimationFrame(frame); }
    document.addEventListener("visibilitychange", function () {
      paused = document.hidden;
      if (!paused && !reduced) kick();
    });

    if (reduced) {
      paint(3.05);                     // static hero frame, no motion
    } else {
      kick();
    }

    // -------------------------------------------------------------- API -----
    return {
      /** current theme: 'light' | 'dark' */
      get theme() { return themeName; },
      setTheme: function (name) {
        if (!THEMES[name]) throw new Error("unknown theme: " + name);
        themeName = name;
        resize();
        if (reduced) paint(3.05);
        return this;
      },
      /** fire a sonar ring from x-fraction (0..1) — hook dashboard events */
      pulse: function (xFrac) {
        var u = Math.max(0, Math.min(1, xFrac == null ? Math.random() : xFrac));
        var ri = Math.floor(Math.random() * RIBBONS);
        clickRipples.push({ x: u * W, y: ribbonY(ri, u, nowT(), W, H),
                            birth: performance.now(), ribbon: ri });
        return this;
      },
      /** let the waves listen to the room (asks for mic permission) */
      enableMic: enableMic,
      /** 0..1 smoothed microphone level (0 when mic off) */
      get micLevel() { return lastLevel; },
      /** agent-state guess from the envelope — LISTENING | THINKING | SPEAKING */
      get state() {
        var e = envelope(nowT(), 0);
        return e > 0.62 ? "SPEAKING" : (e > 0.40 ? "THINKING" : "LISTENING");
      },
      destroy: function () {
        if (rafId) cancelAnimationFrame(rafId);
        window.removeEventListener("resize", resize);
        if (mic.stream) mic.stream.getTracks().forEach(function (t) { t.stop(); });
        canvas.remove();
      }
    };
  }

  global.VoiceField = { mount: mount, THEMES: Object.keys(THEMES) };
})(window);
