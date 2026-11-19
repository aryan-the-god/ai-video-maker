"""
STEP 3: pictures + voice-over (+ subtitles) -> one video, with slow zoom and soft dissolves.

Folders it expects:
  output/script.json, output/voiceover.json, output/voice/scene_XXX.wav   (from Make Video)
  images/scene_001.jpg ...                                                 (from Make Images)
Usage: python assemble_images.py FIRST_SCENE LAST_SCENE [subtitles yes|no] [music none|calm|dramatic|file]
Result: final/video.mp4

How the timing works (T = dissolve length):
  every picture is shown for its scene's voice length + T, and the next picture starts T before
  the previous one ends, so each dissolve is centred on the moment the voice moves to the next scene.
  The voice starts T/2 late so the first picture is already visible.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

first, last = int(sys.argv[1]), int(sys.argv[2])
subtitles = (sys.argv[3] if len(sys.argv) > 3 else "yes").lower() == "yes"
music = (sys.argv[4] if len(sys.argv) > 4 else "none").lower()   # none | calm | dramatic | file

FONT = "Noto Sans Devanagari"
FPS = 24
T = 1.2                 # dissolve length in seconds
W, H = 1280, 720
GROUP = 10              # pictures joined per ffmpeg call (keeps memory small)

script = json.loads(Path("output/script.json").read_text(encoding="utf-8"))
voice = {s["id"]: s for s in json.loads(Path("output/voiceover.json").read_text())["scenes"]}
work, final = Path("work"), Path("final")
work.mkdir(exist_ok=True)
final.mkdir(exist_ok=True)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(map(str, cmd)))
        print(r.stderr[-2000:])
        sys.exit(1)


def stamp(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def caption_lines(text):
    parts = [p.strip() for p in re.split(r"(?<=[।!?.,])\s+", text) if p.strip()]
    lines, cur = [], ""
    for p in parts:                      # keep each caption under ~60 characters
        if cur and len(cur) + len(p) > 60:
            lines.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        lines.append(cur)
    return lines


def fit_canvas(img_path):
    """Load a picture as a 1280x720 frame. If it is not 16:9, the sides are filled with a blurred copy."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if abs(w / h - 16 / 9) < 0.06:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    bg = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    scale = H / h
    fg = cv2.resize(img, (max(2, round(w * scale)), H), interpolation=cv2.INTER_AREA)
    if fg.shape[1] > W:
        x0 = (fg.shape[1] - W) // 2
        return fg[:, x0:x0 + W]
    x0 = (W - fg.shape[1]) // 2
    bg[:, x0:x0 + fg.shape[1]] = fg
    return bg


def picture_segment(img_path, seconds, index, out):
    """One picture -> a video of `seconds` with a slow, smooth zoom.
    Each frame is drawn with sub-pixel accuracy (no stepping, so the picture does not vibrate)."""
    n = max(2, round(seconds * FPS))
    base = fit_canvas(img_path)
    mode = index % 3
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in range(n):
            p = f / (n - 1)
            e = 0.5 - 0.5 * math.cos(math.pi * p)          # smooth start and stop
            if mode == 0:                                  # slow zoom in
                zoom = 1.0 + 0.08 * e
                fx = 0.5
            elif mode == 1:                                # slow zoom out
                zoom = 1.08 - 0.08 * e
                fx = 0.5
            else:                                          # slow zoom in, drifting to the right
                zoom = 1.05 + 0.04 * e
                fx = 0.15 + 0.70 * e
            ww, wh = W / zoom, H / zoom
            x0 = (W - ww) * fx
            y0 = (H - wh) / 2
            s = zoom
            m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
            frame = cv2.warpAffine(base, m, (W, H), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REPLICATE)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode()
        if proc.wait() != 0:
            print(err[-1500:])
            sys.exit(1)
    except BrokenPipeError:
        print(proc.stderr.read().decode()[-1500:])
        sys.exit(1)


def dissolve_join(paths, nominal, out):
    """Join videos with dissolves. Each video is nominal[i] + T long; video k starts at the sum of
    nominal[:k]. The result is sum(nominal) + T long."""
    if len(paths) == 1:
        return paths[0]
    inputs, filt, offset = [], [], 0.0
    for p in paths:
        inputs += ["-i", p]
    prev = "[0:v]"
    for k in range(1, len(paths)):
        offset += nominal[k - 1]
        label = f"[v{k}]"
        filt.append(f"{prev}[{k}:v]xfade=transition=fade:duration={T}:offset={offset:.3f}{label}")
        prev = label
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt),
         "-map", prev, "-r", FPS, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", out])
    return out


# ---- 1) one moving picture per scene ---------------------------------------------------------
segments, secs_list, srt_entries, wavs = [], [], [], []
clock = T / 2          # the voice starts T/2 late
for sc in script["scenes"]:
    sid = sc["id"]
    if not (first <= sid <= last):
        continue
    img = Path("images") / f"scene_{sid:03d}.jpg"
    if not img.exists():
        print(f"!! scene {sid}: no picture found, skipping")
        continue
    secs = float(voice[sid]["seconds"])
    seg = work / f"seg_{sid:03d}.mp4"
    picture_segment(img, secs + T, len(segments), seg)
    segments.append(str(seg))
    secs_list.append(secs)
    wavs.append(voice[sid]["file"])

    lines = caption_lines(sc["narration"])
    total_chars = sum(len(x) for x in lines) or 1
    t = clock
    for line in lines:
        d = secs * len(line) / total_chars
        srt_entries.append((t, t + d, line))
        t += d
    clock += secs
    print(f"scene {sid}: {secs:.1f}s")

if not segments:
    print("nothing to join")
    sys.exit(1)

# ---- 2) join the pictures with dissolves (in groups, then the groups) --------------------------
level, nominal, gen = segments, secs_list, 0
while len(level) > 1:
    gen += 1
    next_level, next_nominal = [], []
    for g in range(0, len(level), GROUP):
        ps, ns = level[g:g + GROUP], nominal[g:g + GROUP]
        out = work / f"join_{gen}_{g // GROUP:03d}.mp4"
        next_level.append(str(dissolve_join(ps, ns, out)))
        next_nominal.append(sum(ns))
    level, nominal = next_level, next_nominal
video_only = level[0]

# ---- 3) voice-over, subtitles, final encode ---------------------------------------------------
total_voice = sum(secs_list)
total = total_voice + T                       # length of the final video
(work / "wavs.txt").write_text("".join(f"file '{Path(w).resolve()}'\n" for w in wavs))
run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", work / "wavs.txt",
     "-af", f"adelay={int(T / 2 * 1000)}:all=1,apad=pad_dur={T / 2}", "-ar", "44100", work / "voice.wav"])

vf = f"fade=t=in:st=0:d=0.8,fade=t=out:st={total - 1.2:.2f}:d=1.2"
if subtitles:
    srt = work / "subs.srt"
    srt.write_text("\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{txt}\n"
                             for i, (a, b, txt) in enumerate(srt_entries, 1)), encoding="utf-8")
    vf += (f",subtitles={srt}:force_style='FontName={FONT},FontSize=20,Outline=2,Shadow=1,"
           f"MarginV=28,PrimaryColour=&H00FFFFFF'")

# background music (optional): made here, or your own file music/bg.mp3 (or .wav)
music_file = None
if music in ("calm", "dramatic"):
    from music_gen import make_music
    music_file = work / "music.wav"
    make_music(music_file, total + 1, music)
    print(f"music: {music} (made here)")
elif music == "file":
    found = sorted(Path("music").glob("bg.*")) if Path("music").exists() else []
    if found:
        music_file = found[0]
        print(f"music: {music_file}")
    else:
        print("!! music/bg.mp3 not found, making the video without music")

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-i", work / "voice.wav"]
if music_file:
    cmd += ["-stream_loop", "-1", "-i", music_file]
    fc = (f"[0:v]{vf}[vout];"
          "[1:a]asplit[va][vb];"
          f"[2:a]atrim=0:{total:.2f},volume=0.35,afade=t=in:d=2,afade=t=out:st={total - 3:.2f}:d=3[m];"
          "[m][vb]sidechaincompress=threshold=0.02:ratio=8:attack=30:release=600[duck];"
          "[va][duck]amix=inputs=2:duration=first:normalize=0[aout]")
    cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
else:
    cmd += ["-vf", vf, "-map", "0:v", "-map", "1:a"]
cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.2f}", final / "video.mp4"]
run(cmd)

dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(final / "video.mp4")], capture_output=True, text=True).stdout.strip())
print(f"DONE: final/video.mp4  {dur:.1f} sec, {len(segments)} pictures")
