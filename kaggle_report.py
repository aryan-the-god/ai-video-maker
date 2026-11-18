"""Reads Kaggle result.json and tells you on Telegram how fast each model was."""
import json, sys
from pathlib import Path
import notify

d = Path(sys.argv[1])
r = json.loads((d / "result.json").read_text())
lines = [f"🎬 Clip speed test\nGPU: {r.get('gpu') or 'none'}"]
for m in r["models"]:
    if m.get("ok"):
        lines.append(f"✅ {m['model']}: {m['seconds']//60} min {m['seconds']%60} sec "
                     f"for a {m['clip_seconds']} sec clip")
    else:
        lines.append(f"❌ {m['model']}: {m['error']}")
notify.send_message("\n".join(lines))
for m in r["models"]:
    if m.get("ok") and (d / m["file"]).exists():
        notify.send_file(str(d / m["file"]), caption=m["model"], kind="document")
