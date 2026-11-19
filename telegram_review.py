"""Waits for your replies on Telegram.

python telegram_review.py approve    -> waits for 'ok' (continue) or 'stop' (cancel) after the script
python telegram_review.py pictures   -> sends numbered picture sheets, then waits for
                                         'change 3,7,12' (remake those pictures + new video) or 'done'
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

import notify

DEADLINE_HOURS = 5
OK_WORDS = {"ok", "okay", "agree", "agreed", "yes", "y", "haan", "han", "ha", "hanji", "theek", "thik",
            "sahi", "go", "start", "chalao", "chalu", "done", "final", "perfect", "good", "accha", "achha"}
STOP_WORDS = {"stop", "cancel", "no", "nahi", "nhi", "ruko", "band"}


def api(method, **params):
    token, _ = notify._creds()
    r = requests.get(notify.API.format(token=token, method=method), params=params, timeout=70)
    return r.json() if r.status_code == 200 else {"ok": False, "result": []}


class Inbox:
    def __init__(self):
        _, self.chat = notify._creds()
        self.offset = None
        self.drain()

    def drain(self):
        """Ignore everything sent before now."""
        data = api("getUpdates", timeout=0)
        for u in data.get("result", []):
            self.offset = u["update_id"] + 1

    def wait(self, deadline):
        """Returns the next text message from you, or None at the deadline."""
        while time.time() < deadline:
            params = {"timeout": 50}
            if self.offset is not None:
                params["offset"] = self.offset
            data = api("getUpdates", **params)
            if not data.get("ok"):
                time.sleep(10)
                continue
            for u in data.get("result", []):
                self.offset = u["update_id"] + 1
                m = u.get("message") or {}
                if str(m.get("chat", {}).get("id")) == self.chat and m.get("text"):
                    return m["text"].strip()
        return None


def words(text):
    return re.findall(r"[a-zA-Zऀ-ॿ]+", text.lower())


def parse_scenes(text, n):
    """'change 3, 7 and 12-14' -> [3, 7, 12, 13, 14] (only valid scene numbers)."""
    found = set()
    for a, b in re.findall(r"(\d+)\s*[-–to]+\s*(\d+)", text):
        found.update(range(int(a), int(b) + 1))
    text2 = re.sub(r"(\d+)\s*[-–]\s*(\d+)", " ", text)
    found.update(int(x) for x in re.findall(r"\d+", text2))
    return sorted(x for x in found if 1 <= x <= n)


def approve():
    inbox = Inbox()
    notify.send_message("📝 Script upar bheji hai, aaram se padh lo.\n"
                        "Reply karo:\n• ok  → voice, pictures aur video banna shuru\n• stop → yahin band")
    deadline = time.time() + DEADLINE_HOURS * 3600
    while True:
        text = inbox.wait(deadline)
        if text is None:
            notify.send_message("⏰ 5 ghante tak reply nahi aaya, isliye video nahi bana. "
                                "Dobara Make Full Video chalao.")
            sys.exit(1)
        w = set(words(text))
        if w & STOP_WORDS:
            notify.send_message("🛑 Theek hai, band kar diya. Topic badal ke dobara chalao.")
            sys.exit(1)
        if w & OK_WORDS:
            notify.send_message("✅ Script approved. Voice, pictures aur video banna shuru.")
            return
        notify.send_message("Samajh nahi aaya. 'ok' likho (aage badho) ya 'stop' (band karo).")


def run(cmd, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run(cmd, env=env).returncode == 0


def pictures():
    n = len(json.loads(Path("output/script.json").read_text(encoding="utf-8"))["scenes"])
    music = os.environ.get("MUSIC", "calm")
    vol = os.environ.get("MUSIC_VOLUME", "35")
    inbox = Inbox()
    run([sys.executable, "contact_sheet.py", "1", str(n)], {"QUIET": "1"})
    prompt = ("🖼 Upar numbered picture sheets hain.\n"
              "Koi picture pasand nahi? Likho:  change 3, 7, 12-14\n"
              "(Naya video bhi bana dunga.)\nSab theek hai to likho:  done")
    notify.send_message(prompt)
    deadline = time.time() + DEADLINE_HOURS * 3600
    changed_any = False
    while True:
        text = inbox.wait(deadline)
        if text is None:
            notify.send_message("⏰ 5 ghante se reply nahi aaya, review band. "
                                + ("Aakhri video Artifacts me 'video-final' naam se hai."
                                   if changed_any else "Original video Artifacts me 'video' naam se hai."))
            return
        w = set(words(text))
        nums = parse_scenes(text, n)
        asks_change = bool(w & {"change", "badlo", "badal", "redo", "fix", "remake", "replace"}) or (
            nums and not (w - {"and", "aur", "to"}))
        if nums and asks_change:
            notify.send_message(f"🔁 Scenes {nums} dobara bana raha hu, phir naya video. Thoda time lagega.")
            ok = run([sys.executable, "make_images.py", "1", str(n)],
                     {"REDO": ",".join(map(str, nums))})
            if not ok:
                notify.send_message("❌ Pictures dobara nahi ban payi (shayad free limit). Baad me try karo ya "
                                    "kuch aur scene number bhejo.")
                continue
            if run([sys.executable, "assemble_images.py", "1", "999", "yes", music, vol]) and \
               run([sys.executable, "assemble_report.py"]):
                changed_any = True
                notify.send_message("✅ Naya video upar hai (preview). Aur badlav? 'change 5, 9' likho, "
                                    "ya 'done' likho.")
            else:
                notify.send_message("❌ Naya video join nahi ho paya. Run ka log Claude ko bhejo.")
        elif w & OK_WORDS:
            notify.send_message("🎉 Final! " + (
                "Poori quality video Artifacts me 'video-final' naam se download karo."
                if changed_any else "Poori quality video Artifacts me 'video' naam se hai (pehle wali)."))
            return
        else:
            notify.send_message("Samajh nahi aaya. Aise likho:  change 3, 7  — ya —  done")


if __name__ == "__main__":
    {"approve": approve, "pictures": pictures}[sys.argv[1]]()
