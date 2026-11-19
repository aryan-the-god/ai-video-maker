from pathlib import Path
import notify

full = Path("final/video.mp4")
preview = Path("final/preview.mp4")
full_mb = full.stat().st_size / 1e6
if preview.exists() and preview.stat().st_size < 49e6:
    notify.send_message(f"🎬 Video taiyar! Poori quality {full_mb:.0f} MB hai "
                        f"(GitHub run ke Artifacts mein 'video' se download karo). "
                        f"Neeche chhoti preview copy hai.")
    notify.send_file(str(preview), caption="Preview (kam quality)", kind="document")
elif full_mb < 49:
    notify.send_message(f"🎬 Video taiyar! ({full_mb:.0f} MB)")
    notify.send_file(str(full), caption="Video", kind="document")
else:
    notify.send_message(f"🎬 Video taiyar, par {full_mb:.0f} MB hai (Telegram ki limit 50 MB). "
                        f"GitHub run ke Artifacts mein 'video' naam se download karo.")
