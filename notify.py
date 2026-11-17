"""
Send messages and files to your Telegram.

Uses two GitHub Secrets:
    TELEGRAM_BOT_TOKEN  - from @BotFather
    TELEGRAM_CHAT_ID    - from @userinfobot

Ways to run it:
    python notify.py "any text"                 -> sends that text
    python notify.py --script-summary output/script.json
                                                -> sends a short summary of the script
                                                   + the full narration as a .txt file
"""

import json
import os
import sys
from pathlib import Path

import requests

sys.stdout.reconfigure(line_buffering=True)

API = "https://api.telegram.org/bot{token}/{method}"


def _creds():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram secrets are missing, so nothing was sent.")
        return None, None
    return token, chat_id


def _check(resp, what):
    if resp.status_code != 200:
        print(f"Telegram error {resp.status_code} while sending {what}: {resp.text[:300]}")
        if resp.status_code in (400, 403):
            print("Tip: open your bot in Telegram and press Start, and check TELEGRAM_CHAT_ID.")
        return False
    print(f"Telegram: sent {what}.")
    return True


def send_message(text):
    token, chat_id = _creds()
    if not token:
        return False
    resp = requests.post(
        API.format(token=token, method="sendMessage"),
        data={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": "true"},
        timeout=30,
    )
    return _check(resp, "a message")


def send_file(path, caption="", kind="document", title=None):
    """kind = 'document' (any file) or 'audio' (mp3 shown with a play button). Max 50 MB."""
    token, chat_id = _creds()
    if not token:
        return False
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return False
    method = {"document": "sendDocument", "audio": "sendAudio"}[kind]
    data = {"chat_id": chat_id, "caption": caption[:1000]}
    if kind == "audio" and title:
        data["title"] = title[:60]
    with open(path, "rb") as f:
        resp = requests.post(API.format(token=token, method=method), data=data,
                             files={kind: (path.name, f)}, timeout=180)
    return _check(resp, path.name)


def script_summary(path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)

    first_lines = "\n".join(f"• {sc['narration']}" for sc in s["scenes"][:3])
    return (
        f"✅ Script ready!\n\n"
        f"🎬 {s['title']}\n"
        f"🗣 Language: {s['language']}\n"
        f"🧩 Scenes: {len(s['scenes'])}\n"
        f"⏱ About {s['estimated_minutes']} minutes of voice-over\n"
        f"🤖 Written by: {s.get('model_used') or '-'}\n\n"
        f"First lines:\n{first_lines}\n\n"
        f"Full narration is in the file below 👇\n{os.environ.get('RUN_URL', '')}"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--script-summary":
        script_path = Path(sys.argv[2])
        send_message(script_summary(script_path))
        narration = script_path.parent / "narration.txt"
        if narration.exists():
            send_file(narration, caption="📜 Poori narration")
    elif len(sys.argv) >= 2:
        send_message(" ".join(sys.argv[1:]))
    else:
        print(__doc__)
