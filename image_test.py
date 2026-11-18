"""
Image test: makes 3 story images (same king, 3 scenes) in the chosen style
and sends them to Telegram, with the time each one took.
"""
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

import notify

sys.stdout.reconfigure(line_buffering=True)

STYLE = {
    "illustrated": "classic 2D storybook illustration, natural human proportions, "
                   "soft painted shading, detailed, not childish, not cartoonish",
    "book": "hand-painted book illustration, ink linework with watercolor texture, "
            "visible paper grain",
    "realistic": "photorealistic cinematic film still, natural light, 35mm",
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
    "a vast ancient Indian kingdom, fort on a hill, green fields and a river, "
    "morning light, wide landscape view",
    f"a scheming minister whispering in the ear of {KING}, dim palace corridor with torches, "
    "tense mood, medium shot",
]

style = os.environ.get("STYLE", "illustrated")
color = os.environ.get("COLOR", "vintage")
style_line = f"{STYLE.get(style, STYLE['illustrated'])}, {COLOR.get(color, COLOR['vintage'])}"

out = Path("output/img_test")
out.mkdir(parents=True, exist_ok=True)
notes = []

for i, scene in enumerate(SCENES, 1):
    prompt = f"{scene}. Style: {style_line}. No text, no letters, no watermark."
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
           + f"?width=1280&height=720&model=flux&nologo=true&seed={i * 7}")
    path = out / f"scene_{i}.jpg"
    t0 = time.time()
    ok = False
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=180)
            ctype = r.headers.get("content-type", "")
            print(f"scene {i} attempt {attempt}: HTTP {r.status_code} {ctype} {len(r.content)} bytes")
            if r.status_code == 200 and ctype.startswith("image") and len(r.content) > 5000:
                path.write_bytes(r.content)
                ok = True
                break
            print(r.text[:200] if not ctype.startswith("image") else "")
        except Exception as e:
            print(f"scene {i} attempt {attempt} error: {e}")
        time.sleep(8)
    secs = time.time() - t0
    notes.append(f"scene {i}: {'OK' if ok else 'FAILED'} in {secs:.0f}s")
    if ok:
        notify.send_file(path, caption=f"Scene {i} | {style} + {color} | {secs:.0f}s")

notify.send_message("🖼 Image test done\n" + "\n".join(notes) + "\n" + os.environ.get("RUN_URL", ""))
