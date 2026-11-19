"""
Soft ambient sound effects (wind, fire, crowd, night, rain, thunder, birds), made from synthesized
noise. No downloads and no copyright. Each scene gets at most one ambience, chosen from the words in
its picture prompt, and neighbouring scenes with the same ambience are joined into one smooth sound.

    kind = classify(picture_prompt)                      # 'fire', 'wind', ... or None
    build_track("work/sfx.wav", [(start, seconds, kind), ...], total_seconds, gain)
"""
import re
import wave

import numpy as np

SR = 44100

# first match wins
RULES = [
    ("thunder", r"thunder|lightning|storm"),
    ("rain", r"\brain|monsoon|downpour|drizzle"),
    ("fire", r"torch|bonfire|campfire|\bfire\b|flame|burning|lantern|candle"),
    ("crowd", r"market|bazaar|crowd|festival|gathering|villagers|townspeople|busy street|marketplace"),
    ("night", r"night|moonlit|moon\b|midnight|starry"),
    ("birds", r"forest|jungle|garden|dawn|sunrise|morning|meadow|riverbank|orchard"),
    ("wind", r"hill|fort\b|mountain|desert|plain|battlefield|landscape|cliff|open field|kingdom|ruins|village"),
]
# how loud each ambience is, compared with the others (1.0 = normal)
LEVEL = {"thunder": 1.0, "rain": 0.9, "fire": 1.0, "crowd": 0.9, "night": 0.9, "birds": 0.8, "wind": 1.0}


def classify(prompt):
    text = (prompt or "").lower()
    for kind, pattern in RULES:
        if re.search(pattern, text):
            return kind
    return None


def _band(x, lo, hi):
    """Keep only frequencies between lo and hi (Hz)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def _smooth_random(n, rate_hz, rng):
    """A slowly changing random curve between 0 and 1."""
    pts = max(4, int(n / SR * rate_hz) + 2)
    curve = rng.random(pts)
    return np.interp(np.linspace(0, pts - 1, n), np.arange(pts), curve)


def _pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1
    return np.fft.irfft(spec / np.sqrt(f), n)


def ambience(kind, seconds, rng):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if kind == "wind":
        x = _band(_pink(n, rng), 120, 1100)
        x *= 0.45 + 0.55 * _smooth_random(n, 0.25, rng)
    elif kind == "rain":
        x = _band(rng.standard_normal(n), 900, 9000) * (0.8 + 0.2 * _smooth_random(n, 0.5, rng))
    elif kind == "thunder":
        rain = _band(rng.standard_normal(n), 900, 9000) * 0.6
        rumble = np.zeros(n)
        for at in rng.uniform(0.15, 0.85, size=max(1, int(seconds / 14))) * seconds:
            i = int(at * SR)
            m = min(n - i, int(3.5 * SR))
            if m > 0:
                env = np.exp(-np.arange(m) / (1.1 * SR)) * (1 - np.exp(-np.arange(m) / (0.05 * SR)))
                rumble[i:i + m] += _band(rng.standard_normal(m), 25, 130) * env * 3.0
        x = rain + rumble
    elif kind == "fire":
        x = _band(_pink(n, rng), 40, 400) * 0.5
        pops = np.zeros(n)
        for at in rng.uniform(0, seconds, size=int(seconds * 14)):
            i = int(at * SR)
            m = min(n - i, int(rng.uniform(0.004, 0.03) * SR))
            if m > 8:
                pops[i:i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (m / 4)) * rng.uniform(0.2, 1.0)
        x = x + _band(pops, 700, 6000) * 1.4
    elif kind == "crowd":
        x = np.zeros(n)
        for lo, hi in ((250, 600), (500, 1200), (900, 2200)):
            band = _band(rng.standard_normal(n), lo, hi)
            syll = 0.3 + 0.7 * _smooth_random(n, rng.uniform(3, 6), rng)
            x += band * syll
        x *= 0.6 + 0.4 * _smooth_random(n, 0.3, rng)
    elif kind == "night":
        x = np.zeros(n)
        for f0 in (4300, 4700):
            burst = (np.sin(2 * np.pi * 2.2 * t + rng.uniform(0, 6)) > 0.3)
            pulse = (np.sin(2 * np.pi * 32 * t) > 0).astype(float)
            x += np.sin(2 * np.pi * f0 * t) * burst * pulse * 0.25
        x += _band(_pink(n, rng), 100, 600) * 0.25
    elif kind == "birds":
        x = np.zeros(n)
        for at in rng.uniform(0, max(0.1, seconds - 1), size=int(seconds * 0.9)):
            i = int(at * SR)
            m = int(rng.uniform(0.08, 0.2) * SR)
            if i + m >= n:
                continue
            f0 = rng.uniform(2200, 3800)
            ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.5 * np.linspace(0, 1, m))) / SR
            x[i:i + m] += np.sin(ph) * np.hanning(m) * rng.uniform(0.3, 1.0)
        x += _band(_pink(n, rng), 150, 900) * 0.15
    else:
        return np.zeros(n)
    rms = np.sqrt(np.mean(x ** 2)) or 1.0
    x = x / rms * 0.05 * LEVEL.get(kind, 1.0)          # about -26 dBFS
    return 0.15 * np.tanh(x / 0.15)                    # soften sharp peaks (fire crackle)


def build_track(path, items, total_seconds, gain=0.6, seed=7):
    """items = [(start_sec, seconds, kind or None)]. Returns True if the track has any sound."""
    rng = np.random.default_rng(seed)
    n = int((total_seconds + 1) * SR)
    out = np.zeros(n)
    # merge neighbours that have the same ambience
    merged = []
    for start, secs, kind in items:
        if merged and merged[-1][2] == kind and abs(merged[-1][0] + merged[-1][1] - start) < 0.05:
            merged[-1][1] += secs
        else:
            merged.append([start, secs, kind])
    used = False
    for start, secs, kind in merged:
        if not kind:
            continue
        used = True
        fade = min(1.5, secs / 3)
        a = ambience(kind, secs + 2 * fade, rng)
        env = np.ones(len(a))
        k = int(fade * SR)
        env[:k] = np.linspace(0, 1, k)
        env[-k:] = np.linspace(1, 0, k)
        i = max(0, int((start - fade) * SR))
        seg = (a * env)[: n - i]
        out[i:i + len(seg)] += seg
    out = np.clip(out * gain, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype(np.int16).tobytes())
    return used
