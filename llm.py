"""
One simple function to talk to AI, with automatic backups:

    from llm import AI
    ai = AI()
    answer = ai.ask(prompt, SomePydanticModel)

Order it tries:  Gemini  ->  Groq  ->  OpenRouter
  - Each provider is used only if its key exists in GitHub Secrets:
        GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
  - Busy / slow / not-free models are skipped quickly.
  - Once a model works, later questions go to that same model first
    (so the whole script is written in one consistent style).
  - If everything is busy, it waits a bit and goes round again.
"""

import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

GEMINI_PREFERRED = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
GEMINI_MAX_MODELS = 4
GEMINI_SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "robotics",
                     "computer", "veo", "imagen")

# OpenAI-style providers (same request format)
OPENAI_STYLE = {
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # preferred models, best first (matched by text inside the model id)
        "rank": ["kimi-k2", "gpt-oss-120b", "qwen3", "llama-4-maverick", "llama-3.3-70b", "gpt-oss-20b"],
        "fallback": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b",
                     "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.3-70b-versatile"],
        "skip": ("whisper", "tts", "guard", "playai", "orpheus", "compound", "vision"),
        "max_models": 4,
        "max_tokens": 6000,   # Groq free tier counts this toward its per-minute token limit
    },
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "rank": ["openrouter/free", "deepseek", "qwen", "llama", "gemma", "nemotron", "mistral"],
        "fallback": ["openrouter/free"],
        "skip": ("vision", "-vl", "coder", "embed"),
        "only_free": True,
        "max_models": 3,
        "max_tokens": 8000,
    },
}

REQUEST_TIMEOUT = 120            # seconds for a single request
ROUNDS = 4                       # full passes over all models before giving up
WAIT_BETWEEN_ROUNDS = [60, 120, 180]
TOTAL_TIME_LIMIT = 24 * 60       # stop before GitHub's 30-min job limit


class Skip(Exception):
    """A model failed. `action` says what to do next:
       'dead'   - never use this model again in this run
       'busy'   - try the next model now, maybe this one again next round
       'wait'   - wait `wait` seconds, then retry the same model
       'retry'  - retry the same model straight away (settings were adjusted)
       'provider_dead' - key problem: drop every model of this provider"""
    def __init__(self, action, message, wait=0):
        super().__init__(message)
        self.action, self.message, self.wait = action, message, wait


def _version(name):
    nums = re.findall(r"\d+(?:\.\d+)?", name)
    return float(nums[0]) if nums else 0.0


def _extract_json(text):
    """Get the JSON object out of a model reply (removes ```json fences etc.)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# The main class
# ---------------------------------------------------------------------------

class AI:
    def __init__(self):
        self.t0 = time.time()
        self.last_good = None
        self.no_thinking = set()      # gemini models that refuse the thinking setting
        self.plain_json = set()       # openai-style models that refuse json mode
        self.candidates = []          # list of (provider, model)
        self.gemini_client = None

        print("\nChecking which AI providers you have keys for ...")
        self._setup_gemini()
        for provider in OPENAI_STYLE:
            self._setup_openai_style(provider)

        if not self.candidates:
            self.give_up("Koi bhi AI key nahi mili. GitHub Secrets mein GEMINI_API_KEY "
                         "(ya GROQ_API_KEY) daalo.")
        print("Order to try:")
        for p, m in self.candidates:
            print(f"   {p:<10} {m}")

    # ----- setup ------------------------------------------------------------

    def _setup_gemini(self):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            print("  gemini:     no key -> skipped")
            return
        from google import genai
        from google.genai import types
        self.gemini_client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000))

        found = []
        try:
            for m in self.gemini_client.models.list():
                name = (m.name or "").replace("models/", "")
                low = name.lower()
                if "generateContent" not in (m.supported_actions or []) or "flash" not in low:
                    continue
                if any(w in low for w in GEMINI_SKIP_WORDS):
                    continue
                found.append(name)
        except Exception as e:
            print(f"  gemini:     could not list models ({str(e)[:120]})")
        found.sort(key=lambda n: ("lite" in n, any(w in n for w in ("preview", "exp")), -_version(n)))
        names = []
        for n in GEMINI_PREFERRED + found:
            if n not in names:
                names.append(n)
        names = names[:GEMINI_MAX_MODELS]
        print(f"  gemini:     {len(names)} models")
        self.candidates += [("gemini", n) for n in names]

    def _setup_openai_style(self, provider):
        cfg = OPENAI_STYLE[provider]
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            print(f"  {provider + ':':<11} no key -> skipped")
            return
        names = []
        try:
            r = requests.get(f"{cfg['base']}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
            if r.status_code == 401:
                print(f"  {provider + ':':<11} key was rejected (401) -> skipped. Check {cfg['key_env']}.")
                return
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            if cfg.get("only_free"):
                ids = [i for i in ids if i.endswith(":free")] + ["openrouter/free"]
            ids = [i for i in ids if not any(s in i.lower() for s in cfg["skip"])]

            def rank(i):
                for pos, word in enumerate(cfg["rank"]):
                    if word in i.lower():
                        return pos
                return len(cfg["rank"])
            ids = sorted(set(ids), key=lambda i: (rank(i), i))
            names = [i for i in ids if rank(i) < len(cfg["rank"])]
        except Exception as e:
            print(f"  {provider + ':':<11} could not list models ({str(e)[:120]}), using defaults")
        if not names:
            names = list(cfg["fallback"])
        names = names[:cfg["max_models"]]
        print(f"  {provider + ':':<11} {len(names)} models")
        self.candidates += [(provider, n) for n in names]

    # ----- one call per provider type ---------------------------------------

    def _call_gemini(self, model, prompt, schema):
        from google.genai import errors, types
        kwargs = dict(response_mime_type="application/json", response_schema=schema, temperature=0.8)
        if _version(model) >= 3 and model not in self.no_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        try:
            resp = self.gemini_client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs))
        except errors.APIError as e:
            code, msg = getattr(e, "code", None), str(e)
            if code in (401, 403):
                raise Skip("provider_dead", "Gemini rejected the key (check GEMINI_API_KEY)")
            if code == 404 or (code == 429 and ("PerDay" in msg or "limit: 0" in msg)):
                raise Skip("dead", f"{code}: not available on your free key")
            if code == 429:
                m = re.search(r"retryDelay'?:\s*'?(\d+)", msg)
                raise Skip("wait", "per-minute limit", wait=min(int(m.group(1)) + 3, 65) if m else 30)
            if code == 400 and "think" in msg.lower() and model not in self.no_thinking:
                self.no_thinking.add(model)
                raise Skip("retry", "doesn't accept the thinking setting -> retrying without it")
            raise Skip("busy", f"{code}: {msg[:160]}")
        if resp.parsed is not None:
            return resp.parsed
        return schema.model_validate(_extract_json(resp.text))

    def _call_openai_style(self, provider, model, prompt, schema):
        cfg = OPENAI_STYLE[provider]
        key = os.environ.get(cfg["key_env"], "").strip()
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content":
                    "You are a JSON API. Reply with ONE valid JSON object and nothing else "
                    "(no markdown, no explanations). It must match this JSON Schema:\n" + schema_json},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "max_tokens": cfg["max_tokens"],
        }
        if (provider, model) not in self.plain_json:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["X-Title"] = "ai-video-maker"
        r = requests.post(f"{cfg['base']}/chat/completions", headers=headers, json=body,
                          timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            text = r.text[:300]
            low = text.lower()
            if r.status_code == 401:
                raise Skip("provider_dead", f"{provider} rejected the key (check {cfg['key_env']})")
            if r.status_code in (402, 403, 404, 413) or "per day" in low or "requests per day" in low \
                    or "decommissioned" in low or "not found" in low:
                raise Skip("dead", f"{r.status_code}: {text[:160]}")
            if r.status_code == 429:
                try:
                    wait = int(float(r.headers.get("retry-after", "20")))
                except ValueError:
                    wait = 20
                if wait > 70:
                    raise Skip("busy", f"429: limit resets in {wait}s")
                raise Skip("wait", "rate limit", wait=wait + 2)
            if r.status_code == 400 and ("response_format" in low or "json" in low) \
                    and (provider, model) not in self.plain_json:
                self.plain_json.add((provider, model))
                raise Skip("retry", "doesn't accept JSON mode -> retrying without it")
            raise Skip("busy", f"{r.status_code}: {text[:160]}")

        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise Skip("busy", "answer was cut off (too long)")
        content = (choice.get("message") or {}).get("content") or ""
        return schema.model_validate(_extract_json(content))

    # ----- the public method --------------------------------------------------

    def ask(self, prompt, schema):
        for round_no in range(1, ROUNDS + 1):
            if not self.candidates:
                self.give_up("Koi bhi AI model kaam nahi kar raha (keys check karo).")
            order = list(self.candidates)
            if self.last_good in order:            # start with what worked last time
                i = order.index(self.last_good)
                order = order[i:] + order[:i]

            for cand in order:
                if cand not in self.candidates:    # removed during this round
                    continue
                provider, model = cand
                for attempt in (1, 2, 3):
                    if time.time() - self.t0 > TOTAL_TIME_LIMIT:
                        self.give_up("Saare AI servers bahut der se busy hain. "
                                     "Thodi der baad dobara Run workflow karo.")
                    started = time.time()
                    print(f"  -> {provider}/{model} (try {attempt}, round {round_no}) at {time.strftime('%H:%M:%S')}")
                    try:
                        if provider == "gemini":
                            result = self._call_gemini(model, prompt, schema)
                        else:
                            result = self._call_openai_style(provider, model, prompt, schema)
                        print(f"     answer received in {time.time() - started:.0f}s")
                        self.last_good = cand
                        return result
                    except Skip as s:
                        print(f"     {s.action}: {s.message}")
                        if s.action == "provider_dead":
                            self.candidates = [c for c in self.candidates if c[0] != provider]
                            break
                        if s.action == "dead":
                            self.candidates.remove(cand)
                            break
                        if s.action == "wait":
                            print(f"     waiting {s.wait}s ...")
                            time.sleep(s.wait)
                            continue
                        if s.action == "retry":
                            continue
                        break          # busy -> next model
                    except Exception as e:   # time-out, bad JSON, network problem
                        print(f"     problem after {time.time() - started:.0f}s: {type(e).__name__}: {str(e)[:180]}")
                        if time.time() - started > 60 or attempt >= 2:
                            break
                        time.sleep(3)

            if round_no < ROUNDS:
                wait = WAIT_BETWEEN_ROUNDS[min(round_no - 1, len(WAIT_BETWEEN_ROUNDS) - 1)]
                print(f"\n  Every AI is busy right now. Waiting {wait}s, then round {round_no + 1}/{ROUNDS} ...\n")
                time.sleep(wait)

        self.give_up("Saare AI servers abhi busy hain. Thodi der baad dobara Run workflow karo.")

    def used(self):
        return f"{self.last_good[0]}/{self.last_good[1]}" if self.last_good else None

    @staticmethod
    def give_up(reason):
        print(f"\nERROR: {reason}")
        try:
            import notify
            notify.send_message(f"⚠️ Script nahi ban paya.\n\n{reason}\n\n{os.environ.get('RUN_URL', '')}")
        except Exception:
            pass
        sys.exit(1)
