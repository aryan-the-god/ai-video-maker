"""
Image test (Cloudflare Workers AI, Flux model).
Makes 3 story images (same king, 3 scenes) in the chosen style and sends them to Telegram.
"""
import base64
import os
import sys
import time
from pathlib import Path

import requests

import notify

sys.stdout.reconfigure(line_buffering=True)

STYLE = {
    "storybook": "flat 2D digital storybook illustration for an Indian Hindi story video, "
                 "clean bold outlines, smooth cel shading, natural human proportions, "
                 "expressive faces, detailed background painting, not photorealistic",
    "painterly": "traditional oil painting on canvas, visible brush strokes, dramatic warm "
                 "light, moody atmosphere, painterly textures, not photorealistic",
    "cartoon": "2D animated cartoon frame in the style of Indian animated story channels, "
               "clean outlines, flat colors, soft shadows, friendly expressive faces",
}
COLOR = {
    "colored": "rich natural colors",
    "vintage": "sepia tone like an aged parchment map, faded ink, old yellowed paper, "
               "muted warm brown colors",
}

KING = ("an Indian king in his fifties with a grey trimmed beard, a red and gold turban "
        "with a green jewel, a cream long coat with gold embroidery, a pearl necklace")
SCENES = [
    f"{KING}, sitting on a grand throne in his palace hall, calm and proud, medium shot",
    "a vast ancient Indian kingdom, a fort on a hill, green fields and a river, "
    "morning light, wide landscape view, no people",
    f"a scheming minister in a dark green robe and white turban whispering in the ear of "
    f"{KING}, dim palace corridor with torches, tense mood, medium shot",
]

style = os.environ.get("STYLE", "storybook")
color = os.environ.get("COLOR", "colored")
style_line = f"{STYLE.get(style, STYLE['storybook'])}, {COLOR.get(color, COLOR['colored'])}"

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "").strip()
TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
if not ACCOUNT or not TOKEN:
    notify.send_message("❌ CF_ACCOUNT_ID ya CF_API_TOKEN secret missing hai.")
    sys.exit(1)

URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/black-forest-labs/flux-1-schnell"
HEAD = {"Authorization": f"Bearer {TOKEN}"}

out = Path("output/img_test")
out.mkdir(parents=True, exist_ok=True)
notes = []


def generate(prompt, seed):
    """Returns image bytes or None. Tries a wide size first, then the default size."""
    for body in (
        {"prompt": prompt, "steps": 6, "seed": seed, "width": 1344, "height": 768},
        {"prompt": prompt, "steps": 6, "seed": seed},
    ):
        for attempt in range(1, 4):
            try:
                r = requests.post(URL, headers=HEAD, json=body, timeout=180)
                print(f"HTTP {r.status_code} (size={'wide' if 'width' in body else 'default'}, try {attempt})")
                if r.status_code == 200:
                    data = r.json()
                    b64 = (data.get("result") or {}).get("image")
                    if b64:
                        return base64.b64decode(b64)
                    print("No image in response:", str(data)[:300])
                elif r.status_code == 429:
                    time.sleep(10)
                    continue
                else:
                    print(r.text[:300])
                    break
            except Exception as e:
                print("error:", e)
                time.sleep(5)
    return None


for i, scene in enumerate(SCENES, 1):
    prompt = f"{scene}. Style: {style_line}. No text, no letters, no watermark."
    t0 = time.time()
    img = generate(prompt, seed=i * 11)
    secs = time.time() - t0
    if img:
        path = out / f"scene_{i}.jpg"
        path.write_bytes(img)
        notes.append(f"scene {i}: OK in {secs:.0f}s")
        notify.send_file(path, caption=f"Scene {i} | {style} + {color} | {secs:.0f}s")
    else:
        notes.append(f"scene {i}: FAILED in {secs:.0f}s")

notify.send_message("🖼 Image test done\n" + "\n".join(notes) + "\n" + os.environ.get("RUN_URL", ""))
