"""Builds the list of clips for Kaggle from script.json + voiceover.json.
Each scene gets enough ~2-second clips to cover its voice-over length.
Usage: python prep_clips.py FIRST_SCENE LAST_SCENE"""
import json, math, sys
from pathlib import Path

CLIP_SECONDS = 2.0
first, last = int(sys.argv[1]), int(sys.argv[2])
script = json.loads(Path("output/script.json").read_text(encoding="utf-8"))
voice = json.loads(Path("output/voiceover.json").read_text(encoding="utf-8"))
secs = {s["id"]: s["seconds"] for s in voice["scenes"]}
style = script.get("visual_style", "cinematic")

jobs = []
for sc in script["scenes"]:
    if not (first <= sc["id"] <= last):
        continue
    n = max(1, math.ceil(secs[sc["id"]] / CLIP_SECONDS))
    for c in range(n):
        k = len(sc["shots"])
        shot = sc["shots"][min(k - 1, int(c * k / n))]     # shots in order, spread over the scene
        jobs.append({"clip": f"s{sc['id']:03d}_c{c+1:02d}", "seed": sc["id"] * 100 + c,
                     "prompt": f"{shot} Style: {style}. Cinematic film look, natural motion."})

src = Path("kaggle/gen_clips/gen_clips.py")
text = src.read_text()
assert "__JOBS__" in text
src.write_text(text.replace("__JOBS__", json.dumps(jobs, ensure_ascii=False)))
print(f"{len(jobs)} clips for scenes {first}-{last}  (~{len(jobs)*2.5/2:.0f} min on 2 GPUs)")
