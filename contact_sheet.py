"""Puts the pictures on numbered contact sheets (12 per sheet) and sends them to Telegram,
so you can spot bad pictures and note their scene numbers.
Usage: python contact_sheet.py FIRST LAST     (reads images/scene_XXX.jpg)"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import notify

first, last = int(sys.argv[1]), int(sys.argv[2])
files = [(i, Path("images") / f"scene_{i:03d}.jpg") for i in range(first, last + 1)]
files = [(i, p) for i, p in files if p.exists()]
if not files:
    sys.exit("no pictures found")

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
except Exception:
    font = ImageFont.load_default()

COLS, ROWS, TW, TH = 3, 4, 640, 360
for sheet_no, start in enumerate(range(0, len(files), COLS * ROWS), 1):
    part = files[start:start + COLS * ROWS]
    sheet = Image.new("RGB", (COLS * TW, ROWS * TH), "black")
    for k, (sid, p) in enumerate(part):
        im = Image.open(p).convert("RGB").resize((TW - 6, TH - 6))
        x, y = (k % COLS) * TW + 3, (k // COLS) * TH + 3
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x, y, x + 90, y + 54], fill=(0, 0, 0))
        d.text((x + 8, y + 2), str(sid), fill=(255, 255, 0), font=font)
    out = Path(f"sheet_{sheet_no}.jpg")
    sheet.save(out, quality=88)
    notify.send_file(out, caption=f"Sheet {sheet_no}: scenes {part[0][0]}-{part[-1][0]}")
if not os.environ.get("QUIET"):
    notify.send_message("🖼 Bad pictures ke scene numbers likh ke Claude ko bhejo (ya seedha 'Make Images' mein redo_scenes mein daalo).")
