"""
STEP 2: makes one picture per scene (Cloudflare Workers AI, Flux model).

Usage:  python make_images.py FIRST LAST        (scene numbers, both included)
Reads   output/script.json
Writes  images/scene_001.jpg ...  and  images/images_manifest.json
Needs   env CF_ACCOUNT_ID, CF_API_TOKEN (GitHub Secrets)
"""
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

import notify

sys.stdout.reconfigure(line_buffering=True)

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "").strip()
TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/black-forest-labs/"
URL_KLEIN = BASE + "flux-2-klein-4b"      # lets us choose a landscape size (multipart form)
URL_SCHNELL = BASE + "flux-1-schnell"     # backup: always square
HEAD = {"Authorization": f"Bearer {TOKEN}"}
WIDTH, HEIGHT = 1280, 720

# Options to try, best first. The first one that works is remembered for the next pictures.
OPTIONS = [
    ("klein", None),
    ("schnell", {"steps": 6, "width": 1344, "height": 768}),
    ("schnell", {"steps": 6}),
    ("schnell", {}),
]
working = [None]
last_error = [""]
quota_hit = [False]
used_model = [""]


def one_request(kind, extra, prompt):
    if kind == "klein":
        form = {"prompt": (None, prompt), "width": (None, str(WIDTH)), "height": (None, str(HEIGHT))}
        return requests.post(URL_KLEIN, headers=HEAD, files=form, timeout=180)
    return requests.post(URL_SCHNELL, headers=HEAD, json=dict(extra, prompt=prompt), timeout=180)


def read_image(r):
    if r.headers.get("content-type", "").startswith("image"):
        return r.content
    b64 = (r.json().get("result") or {}).get("image")
    return base64.b64decode(b64) if b64 else None


def generate(prompt):
    order = [working[0]] if working[0] is not None else range(len(OPTIONS))
    for idx in order:
        kind, extra = OPTIONS[idx]
        for attempt in range(1, 5):
            try:
                r = one_request(kind, extra, prompt)
            except Exception as e:
                last_error[0] = f"network: {e}"
                time.sleep(8)
                continue
            if r.status_code == 200:
                try:
                    img = read_image(r)
                except Exception as e:
                    img, last_error[0] = None, f"bad answer: {e}"
                if img:
                    working[0] = idx
                    used_model[0] = kind
                    return img
                break
            text = r.text[:300]
            last_error[0] = f"{kind} HTTP {r.status_code}: {text[:160]}"
            print(last_error[0])
            low = text.lower()
            if r.status_code == 429 and ("neuron" in low or "daily" in low or "allocation" in low):
                quota_hit[0] = True
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * attempt)
                continue
            break  # 400 etc: try the next option
    return None


FRAMING = ("Wide establishing shot, cinematic 16:9 landscape composition, the whole environment "
           "visible with rich detailed background (architecture, sky, scenery, objects); people "
           "are shown small to medium-wide in the frame, full body, not a close-up.")


def widen(shot):
    """The pictures should be wide scenes, not face close-ups."""
    shot = re.sub(r"(?i)\b(extreme |tight |medium )?close[- ]?up( shot)?\b", "wide shot", shot)
    return shot


def main():
    if not ACCOUNT or not TOKEN:
        notify.send_message("❌ CF_ACCOUNT_ID ya CF_API_TOKEN secret missing hai.")
        sys.exit(1)

    first, last = int(sys.argv[1]), int(sys.argv[2])
    script = json.loads(Path("output/script.json").read_text(encoding="utf-8"))
    style = script.get("visual_style", "")
    scenes = [s for s in script["scenes"] if first <= s["id"] <= last]

    out = Path("images")
    out.mkdir(exist_ok=True)
    made, failed = [], []
    t_start = time.time()

    for s in scenes:
        path = out / f"scene_{s['id']:03d}.jpg"
        if path.exists() and path.stat().st_size > 5000:
            made.append(s["id"])
            continue
        prompt = (f"{FRAMING} Scene: {widen(s['shots'][0])}. Style: {style}. "
                  "No text, no letters, no signs, no watermark.")
        print(f"Scene {s['id']}/{scenes[-1]['id']} ...")
        img = generate(prompt)
        if img is None and not quota_hit[0]:
            time.sleep(10)
            img = generate(prompt)
        if img:
            path.write_bytes(img)
            made.append(s["id"])
        else:
            failed.append(s["id"])
            if quota_hit[0]:
                print("Daily free limit reached, stopping.")
                break

    Path("images/images_manifest.json").write_text(
        json.dumps({"made": made, "failed": failed}, indent=2), encoding="utf-8")

    mins = (time.time() - t_start) / 60
    msg = (f"🖼 Images ready: {len(made)}/{len(scenes)} (scenes {first}-{last}) in {mins:.1f} min"
           f" | model: {used_model[0] or '-'}")
    if failed:
        msg += f"\nFailed scenes: {failed}\nLast error: {last_error[0]}"
    if quota_hit[0]:
        msg += "\n⚠️ Cloudflare ki roz ki free limit khatam. Kal (UTC 00:00 ke baad) baaki scenes chalao."
    msg += "\n" + os.environ.get("RUN_URL", "")
    notify.send_message(msg)
    for sid in made[:4]:
        notify.send_file(out / f"scene_{sid:03d}.jpg", caption=f"Scene {sid}")
    if not made:
        sys.exit(1)


if __name__ == "__main__":
    main()
