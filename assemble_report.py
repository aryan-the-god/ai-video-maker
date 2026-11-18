from pathlib import Path
import notify
v = Path("final/video.mp4")
size = v.stat().st_size / 1e6
if size < 49:
    notify.send_message(f"🎬 Video taiyar! ({size:.0f} MB)")
    notify.send_file(str(v), caption="Test video", kind="document")
else:
    notify.send_message(f"🎬 Video taiyar, par {size:.0f} MB hai (Telegram ki limit 50 MB). GitHub run ke Artifacts mein 'video' naam se download karo.")
