import json, sys
from pathlib import Path
import notify
d = Path(sys.argv[1])
m = json.loads((d / "clips_manifest.json").read_text())
notify.send_message(f"🎞 Clips ready: {m['made']}/{m['requested']} clips in {m['minutes']} min on {m['gpus']} GPU(s).")
for f in m["files"][:3]:
    notify.send_file(str(d / "clips" / f), caption=f, kind="document")
