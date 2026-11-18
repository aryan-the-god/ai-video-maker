import json, sys
from pathlib import Path
import notify
d = Path(sys.argv[1])
r = json.loads((d / "result.json").read_text())
lines = [f"🔍 Quality test\nGPU: {r.get('gpu')}"]
for m in r["items"]:
    lines.append(f"✅ {m['name']}: {m['seconds']//60} min {m['seconds']%60} sec" if m.get("ok") else f"❌ {m['name']}: {m['error']}")
notify.send_message("\n".join(lines))
for m in r["items"]:
    if m.get("ok") and (d / m["file"]).exists():
        notify.send_file(str(d / m["file"]), caption=m["name"], kind="document")
