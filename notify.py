"""
Send messages to your Telegram.

Uses two GitHub Secrets:
    TELEGRAM_BOT_TOKEN  - from @BotFather
    TELEGRAM_CHAT_ID    - from @userinfobot

Ways to run it:
    python notify.py "any text"                 -> sends that text
    python notify.py --script-summary output/script.json
                                                -> sends a short summary of the script
"""

import json
import os
import sys

import requests


def send_message(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram secrets are missing, so no message was sent.")
        return False

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": "true"},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Telegram error {resp.status_code}: {resp.text[:300]}")
        if resp.status_code in (400, 403):
            print("Tip: open your bot in Telegram and press Start, and check TELEGRAM_CHAT_ID.")
        return False

    print("Telegram message sent.")
    return True


def script_summary(path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)

    first_lines = "\n".join(f"• {sc['narration']}" for sc in s["scenes"][:3])
    run_url = os.environ.get("RUN_URL", "")

    return (
        f"✅ Script ready!\n\n"
        f"🎬 {s['title']}\n"
        f"🗣 Language: {s['language']}\n"
        f"🧩 Scenes: {len(s['scenes'])}\n"
        f"⏱ About {s['estimated_minutes']} minutes of voice-over\n\n"
        f"First lines:\n{first_lines}\n\n"
        f"Download the full script here (bottom of the page, 'script'):\n{run_url}"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--script-summary":
        send_message(script_summary(sys.argv[2]))
    elif len(sys.argv) >= 2:
        send_message(" ".join(sys.argv[1:]))
    else:
        print(__doc__)
