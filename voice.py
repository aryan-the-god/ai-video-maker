"""
STAGE 2: turns every scene's narration into a voice-over file.

Reads  output/script.json
Writes output/voice/scene_001.wav ...   (one file per scene)
       output/voiceover.json            (scene id -> file, voice, duration in seconds)

Env:
  LANGUAGE    English | Hindi
  VOICE_MODE  male | female | alternate_chapter | alternate_scene
              (alternate = male and female voices take turns)
"""

import json
import os
import re
import sys
from pathlib import Path

import notify

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 24000
PAUSE_SECONDS = 0.3      # between sentences inside a scene
TAIL_SECONDS = 0.35      # small breath at the end of each scene

VOICES = {
    "Hindi":   {"lang_code": "h", "male": "hm_psi",    "female": "hf_alpha"},
    "English": {"lang_code": "a", "male": "am_michael", "female": "af_heart"},
}


def split_sentences(text):
    parts = re.split(r"(?<=[।!?.])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def pick_gender(mode, scene, index):
    if mode == "male":
        return "male"
    if mode == "female":
        return "female"
    key = scene["chapter"] if mode == "alternate_chapter" else index
    return "male" if key % 2 == 1 else "female"


def main():
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    language = os.environ.get("LANGUAGE", "Hindi")
    mode = os.environ.get("VOICE_MODE", "alternate_chapter")
    cfg = VOICES[language]

    script = json.loads(Path("output/script.json").read_text(encoding="utf-8"))
    scenes = script["scenes"]
    out = Path("output/voice")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Language: {language} | mode: {mode} | scenes: {len(scenes)}")
    pipe = KPipeline(lang_code=cfg["lang_code"], repo_id="hexgrad/Kokoro-82M")
    pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype=np.float32)
    tail = np.zeros(int(SAMPLE_RATE * TAIL_SECONDS), dtype=np.float32)

    info = []
    for n, scene in enumerate(scenes, 1):
        gender = pick_gender(mode, scene, n)
        voice = cfg[gender]
        lines = "\n".join(split_sentences(scene["narration"]))
        parts = []
        for result in pipe(lines, voice=voice, speed=1.0, split_pattern=r"\n+"):
            audio = result.audio
            if audio is None:
                continue
            parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio, dtype=np.float32))
            parts.append(pause)
        if not parts:
            raise RuntimeError(f"Scene {scene['id']}: no audio was produced")
        parts[-1] = tail
        audio = np.concatenate(parts)
        path = out / f"scene_{scene['id']:03d}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        info.append({"id": scene["id"], "chapter": scene["chapter"], "file": str(path),
                     "voice": voice, "gender": gender,
                     "seconds": round(len(audio) / SAMPLE_RATE, 2)})
        if n % 10 == 0 or n == len(scenes):
            print(f"  {n}/{len(scenes)} scenes done")

    total = sum(i["seconds"] for i in info)
    Path("output/voiceover.json").write_text(
        json.dumps({"mode": mode, "language": language, "total_seconds": round(total, 1),
                    "scenes": info}, indent=2), encoding="utf-8")

    males = sum(1 for i in info if i["gender"] == "male")
    notify.send_message(
        f"🎙 Voice-over ready!\nMode: {mode}\nTotal: {int(total // 60)} min {int(total % 60)} sec "
        f"({len(info)} scenes)\nMale scenes: {males} | Female scenes: {len(info) - males}")
    print(f"\nDONE: {total / 60:.1f} minutes of voice-over")


if __name__ == "__main__":
    main()
