"""
STAGE 4: joins clips + voice-over (+ subtitles) into one video with ffmpeg.

Folders it expects:
  output/script.json, output/voiceover.json, output/voice/scene_XXX.wav   (from Make Video)
  clips/s001_c01.mp4 ...                                                 (from Make Clips)
Usage: python assemble.py FIRST_SCENE LAST_SCENE [subtitles yes|no]
Result: final/video.mp4
"""
import json, math, re, subprocess, sys
from pathlib import Path

first, last = int(sys.argv[1]), int(sys.argv[2])
subtitles = (sys.argv[3] if len(sys.argv) > 3 else "yes").lower() == "yes"
FONT = "Noto Sans Devanagari"

script = json.loads(Path("output/script.json").read_text(encoding="utf-8"))
voice = {s["id"]: s for s in json.loads(Path("output/voiceover.json").read_text())["scenes"]}
work, final = Path("work"), Path("final")
work.mkdir(exist_ok=True); final.mkdir(exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(map(str, cmd))); print(r.stderr[-1500:]); sys.exit(1)


def stamp(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def make_srt(text, seconds, path):
    """Split the narration into short lines and spread them over the scene length."""
    parts = [p.strip() for p in re.split(r"(?<=[।!?.,])\s+", text) if p.strip()]
    lines, cur = [], ""
    for p in parts:                       # keep each caption under ~60 characters
        if cur and len(cur) + len(p) > 60:
            lines.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        lines.append(cur)
    total = sum(len(l) for l in lines) or 1
    t, out = 0.0, []
    for i, l in enumerate(lines, 1):
        d = seconds * len(l) / total
        out.append(f"{i}\n{stamp(t)} --> {stamp(t + d)}\n{l}\n")
        t += d
    path.write_text("\n".join(out), encoding="utf-8")


scene_files = []
for sc in script["scenes"]:
    sid = sc["id"]
    if not (first <= sid <= last):
        continue
    clips = sorted(Path("clips").glob(f"s{sid:03d}_c*.mp4"))
    if not clips:
        print(f"!! scene {sid}: no clips found, skipping"); continue
    secs = voice[sid]["seconds"]
    lst = work / f"s{sid:03d}.txt"
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    vf = "scale=1280:720:flags=lanczos,fps=24,format=yuv420p"
    if subtitles:
        srt = work / f"s{sid:03d}.srt"; make_srt(sc["narration"], secs, srt)
        vf += (f",subtitles={srt}:force_style='FontName={FONT},FontSize=20,Outline=2,Shadow=1,"
               f"MarginV=28,PrimaryColour=&H00FFFFFF'")
    out = work / f"scene_{sid:03d}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-i", voice[sid]["file"], "-vf", vf, "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
         "-ar", "44100", "-t", f"{secs}", str(out)])
    scene_files.append(out)
    print(f"scene {sid}: {len(clips)} clips -> {secs}s")

if not scene_files:
    print("nothing to join"); sys.exit(1)
(work / "all.txt").write_text("".join(f"file '{f.resolve()}'\n" for f in scene_files))
run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "all.txt"),
     "-c", "copy", str(final / "video.mp4")])
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(final / "video.mp4")], capture_output=True, text=True).stdout.strip())
print(f"DONE: final/video.mp4  {dur:.1f} sec, {len(scene_files)} scenes")
