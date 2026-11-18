import json, sys
from pathlib import Path
import notify
d = Path(sys.argv[1])
m = json.loads((d / "clips_manifest.json").read_text())
msg = f"🎞 Clips ready: {m['made']}/{m['requested']} clips in {m['minutes']} min on {m['gpus']} GPU(s)."

if m["made"] < m["requested"]:
    errs = []
    for log in sorted(d.glob("worker*.log")):
        for line in log.read_text(errors="ignore").splitlines():
            if "failed" in line and line not in errs:
                errs.append(line)
    msg += "\n\n⚠ Kuch clips nahi bane. Errors:\n" + ("\n".join(e[:230] for e in errs[:4]) or "(koi error line nahi mili)")
notify.send_message(msg)

for log in sorted(d.glob("worker*.log")):
    notify.send_file(str(log), caption=log.name, kind="document")
for f in m["files"][:3]:
    notify.send_file(str(d / "clips" / f), caption=f, kind="document")
