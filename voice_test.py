"""
VOICE TEST: makes the same Hindi sample in several voices and sends each one
to your Telegram, numbered, so you can listen and pick your favourite.

  1-4  Kokoro (open-source, runs here for free, safe for YouTube)
  5-7  Microsoft Edge voices (very natural, but an unofficial free service
       that Microsoft could stop at any time)

Optional input (from the GitHub form): SAMPLE_TEXT - your own Hindi text.
"""

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

import notify

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_TEXT = (
    "कुरुक्षेत्र का मैदान। सुबह की पहली किरण के साथ, दो विशाल सेनाएँ आमने-सामने खड़ी थीं। "
    "एक तरफ़ पांडव, दूसरी तरफ़ कौरव। शंखों की गूंज से पूरा आसमान काँप उठा। "
    "और तभी, अर्जुन का धनुष उसके हाथों से फिसलने लगा। "
    "क्योंकि सामने कोई दुश्मन नहीं, उसके अपने खड़े थे।"
)

KOKORO_VOICES = [("hm_omega", "Male"), ("hm_psi", "Male"), ("hf_alpha", "Female"), ("hf_beta", "Female")]
EDGE_VOICES = [("hi-IN-MadhurNeural", "Male"), ("hi-IN-AaravNeural", "Male"), ("hi-IN-SwaraNeural", "Female")]

OUT = Path("samples")
SAMPLE_RATE = 24000


def split_sentences(text):
    """Hindi sentences end with '।' (and sometimes . ! ?). Kokoro only splits on
    . ! ? by itself, so we split here to avoid words being cut in the middle."""
    parts = re.split(r"(?<=[।!?.])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def to_mp3(wav, mp3):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)


def make_kokoro(text):
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    print("Loading Kokoro (Hindi) ...")
    pipe = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M")
    lines = "\n".join(split_sentences(text))
    pause = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)   # short pause between sentences

    made = []
    for voice, gender in KOKORO_VOICES:
        try:
            print(f"  Kokoro {voice} ...")
            parts = []
            for result in pipe(lines, voice=voice, speed=1.0, split_pattern=r"\n+"):
                audio = result.audio
                if audio is None:
                    continue
                parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio, dtype=np.float32))
                parts.append(pause)
            if not parts:
                raise RuntimeError("no audio was produced")
            wav = OUT / f"kokoro_{voice}.wav"
            sf.write(wav, np.concatenate(parts), SAMPLE_RATE)
            mp3 = OUT / f"kokoro_{voice}.mp3"
            to_mp3(wav, mp3)
            made.append((mp3, f"Kokoro · {voice} · {gender}", "open-source, YouTube ke liye safe"))
        except Exception as e:
            print(f"  !! Kokoro {voice} failed: {type(e).__name__}: {e}")
    return made


def make_edge(text):
    import edge_tts

    async def one(voice, path):
        await edge_tts.Communicate(text, voice, rate="-5%").save(str(path))

    made = []
    for voice, gender in EDGE_VOICES:
        try:
            print(f"  Edge {voice} ...")
            mp3 = OUT / f"edge_{voice}.mp3"
            asyncio.run(one(voice, mp3))
            if not mp3.exists() or mp3.stat().st_size < 1000:
                raise RuntimeError("empty audio")
            short = voice.replace("hi-IN-", "").replace("Neural", "")
            made.append((mp3, f"Microsoft Edge · {short} · {gender}",
                         "bahut natural, par unofficial (kabhi bhi band ho sakta hai)"))
        except Exception as e:
            print(f"  !! Edge {voice} failed: {type(e).__name__}: {e}")
    return made


def main():
    text = (os.environ.get("SAMPLE_TEXT") or "").strip() or DEFAULT_TEXT
    OUT.mkdir(exist_ok=True)
    print(f"Text: {text}\n")

    samples = []
    try:
        samples += make_kokoro(text)
    except Exception as e:
        print(f"!! Kokoro could not start: {type(e).__name__}: {e}")
    try:
        samples += make_edge(text)
    except Exception as e:
        print(f"!! Edge voices could not start: {type(e).__name__}: {e}")

    if not samples:
        notify.send_message("❌ Koi bhi voice sample nahi ban paya. " + os.environ.get("RUN_URL", ""))
        sys.exit(1)

    notify.send_message(f"🎙 Voice samples ({len(samples)}) aa rahe hain. Sabko sunke batana "
                        f"kaunsa number pasand aaya.")
    for n, (path, label, note) in enumerate(samples, 1):
        notify.send_file(path, caption=f"{n}) {label}\n{note}", kind="audio", title=f"{n}) {label}")

    notify.send_message("👆 Bas itne hi samples the. Claude ko apna pasandida number bata do "
                        "(ya 2 numbers, ek male aur ek female).")
    print(f"\nDONE: sent {len(samples)} samples")


if __name__ == "__main__":
    main()
