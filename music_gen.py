"""
Makes a soft background music track (no download, no copyright) from simple synthesized chords.

    from music_gen import make_music
    make_music("work/music.wav", seconds=300, mood="calm")      # mood: calm | dramatic
"""
import wave

import numpy as np

SR = 44100
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def freq(name, octave):
    """'A', 2 -> Hz. A trailing 'b' means flat."""
    base = NOTE[name[0]] + (-1 if name.endswith("b") else 0)
    midi = 12 * (octave + 1) + base
    return 440.0 * 2 ** ((midi - 69) / 12)


MOODS = {
    # each chord: list of (note, octave); every chord lasts `beat` seconds
    "calm": {
        "beat": 10.0,
        "chords": [[("C", 3), ("E", 3), ("G", 3), ("B", 3)],
                   [("A", 2), ("C", 3), ("E", 3), ("G", 3)],
                   [("F", 2), ("A", 2), ("C", 3), ("E", 3)],
                   [("G", 2), ("B", 2), ("D", 3), ("F", 3)]],
        "bass": [("C", 2), ("A", 1), ("F", 1), ("G", 1)],
        "swell": 0.0,
    },
    "dramatic": {
        "beat": 8.0,
        "chords": [[("D", 3), ("F", 3), ("A", 3), ("C", 4)],
                   [("Bb", 2), ("D", 3), ("F", 3), ("A", 3)],
                   [("G", 2), ("Bb", 2), ("D", 3), ("F", 3)],
                   [("A", 2), ("C", 3), ("E", 3), ("G", 3)]],
        "bass": [("D", 2), ("Bb", 1), ("G", 1), ("A", 1)],
        "swell": 0.12,
    },
}


def _voice(f, t):
    """One soft note: two slightly detuned sines plus a quiet octave above."""
    return (np.sin(2 * np.pi * f * t) + np.sin(2 * np.pi * f * 1.004 * t + 1.0)
            + 0.25 * np.sin(2 * np.pi * 2 * f * t + 0.5))


def make_music(path, seconds, mood="calm"):
    cfg = MOODS.get(mood, MOODS["calm"])
    n = int((seconds + 2) * SR)
    t = np.arange(n) / SR
    beat, cf = cfg["beat"], 3.0                      # chord length and cross-fade (seconds)
    out = np.zeros(n)
    k, start = 0, 0.0
    while start < seconds + 2:
        chord = cfg["chords"][k % len(cfg["chords"])]
        bass = cfg["bass"][k % len(cfg["bass"])]
        i0 = int(start * SR)
        i1 = min(n, int((start + beat + cf) * SR))
        tt = t[i0:i1] - start
        seg = sum(_voice(freq(nm, oc), tt) for nm, oc in chord) / len(chord)
        seg = seg + 0.9 * _voice(freq(*bass), tt)
        env = np.ones_like(tt)
        a = np.clip(tt / cf, 0, 1)                   # fade in
        d = np.clip((beat + cf - tt) / cf, 0, 1)     # fade out
        env = np.minimum(a, d) ** 1.5
        out[i0:i1] += seg * env
        start += beat
        k += 1
    # slow breathing, so the pad does not feel static
    out *= 0.85 + 0.15 * np.sin(2 * np.pi * (0.07 + cfg["swell"]) * t)
    # soft echo
    for delay, gain in ((0.45, 0.35), (0.9, 0.2)):
        d = int(delay * SR)
        echo = np.zeros_like(out)
        echo[d:] = out[:-d] * gain
        out = out + echo
    # gentle low-pass (moving average) to remove harshness
    k_len = 24
    out = np.convolve(out, np.ones(k_len) / k_len, mode="same")
    out = out[: int(seconds * SR)]
    out /= max(1e-6, np.abs(out).max())
    out *= 0.7
    data = (out * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    import sys
    make_music(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "calm")
