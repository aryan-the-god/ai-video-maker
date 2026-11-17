"""
STEP 1 of the video pipeline: write the script as a SCENE LIST.

Input  (environment variables, set by the GitHub workflow):
    GEMINI_API_KEY  - your Gemini key (from GitHub Secrets)
    TOPIC           - e.g. "Battle of Kurukshetra"
    MINUTES         - target video length, e.g. "15"
    LANGUAGE        - "English" or "Hindi" (language of the voice-over)

Output (in the output/ folder):
    script.json     - title, characters, and every scene:
                      narration (what the voice says) + shots (what the screen shows)
    narration.txt   - the full narration as plain text, easy to read
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Gemini models to try first (names Google itself recommended in Nov 2026).
# On top of these, the script asks Google which models your key can use and
# tries those too, so it keeps working when Google renames models again.
PREFERRED_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "robotics", "computer", "veo", "imagen")
MAX_MODELS_TO_TRY = 8

WORDS_PER_MINUTE = {"English": 150, "Hindi": 130}  # normal narration speed
WORDS_PER_SCENE = 28          # about 10-12 seconds of voice per scene
SHOTS_PER_SCENE = 2           # 2 video clips per scene
PAUSE_BETWEEN_CALLS = 13      # seconds; keeps us under the free-tier per-minute limit

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# The shape of the answers we want from Gemini
# ---------------------------------------------------------------------------

class Character(BaseModel):
    name: str = Field(description="Character name, e.g. 'Arjuna'")
    look: str = Field(description="Fixed visual description used in EVERY shot: age, face, hair, "
                                  "clothing, armour, colours. 20-40 words. English.")


class Chapter(BaseModel):
    title: str
    summary: str = Field(description="What happens in this chapter, 2-4 sentences")
    key_facts: list[str] = Field(description="Important names, places, dates or events to mention")


class Outline(BaseModel):
    title: str = Field(description="Catchy YouTube title, max 90 characters")
    youtube_description: str = Field(description="YouTube description, 3-5 sentences")
    tags: list[str] = Field(description="8-15 YouTube tags")
    visual_style: str = Field(description="One visual style line added to every shot so the whole "
                                          "film looks consistent, e.g. 'cinematic, 35mm film, warm "
                                          "desaturated colours, dramatic lighting, epic scale'. English.")
    characters: list[Character]
    chapters: list[Chapter]


class Scene(BaseModel):
    narration: str = Field(description="What the narrator says in this scene")
    shots: list[str] = Field(description="Video prompts for what the screen shows while this "
                                         "narration plays. English.")


class ChapterScenes(BaseModel):
    scenes: list[Scene]


# ---------------------------------------------------------------------------
# Talking to Gemini (with retries and model fallback)
# ---------------------------------------------------------------------------

def find_models(client):
    """Build the list of models to try: preferred ones first, then whatever
    'flash' text models Google says this key can use (newest first)."""
    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            low = name.lower()
            actions = m.supported_actions or []
            if "generateContent" not in actions or "flash" not in low:
                continue
            if any(word in low for word in SKIP_WORDS):
                continue
            found.append(name)
    except Exception as e:
        print(f"  (could not get the model list from Google: {str(e)[:200]})")

    def sort_key(name):
        nums = re.findall(r"\d+(?:\.\d+)?", name)
        version = float(nums[0]) if nums else 0.0
        is_test = any(w in name for w in ("preview", "exp"))
        return ("lite" in name, is_test, -version)   # full > lite, stable > preview, newest first

    found.sort(key=sort_key)
    ordered = []
    for name in PREFERRED_MODELS + found:
        if name not in ordered:
            ordered.append(name)
    return ordered[:MAX_MODELS_TO_TRY]


def ask_gemini(client, prompt, schema, model_state):
    """Send one prompt, get back a parsed object of type `schema`."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.8,
    )
    models = model_state["models"]

    while model_state["index"] < len(models):
        model = models[model_state["index"]]
        for attempt in range(1, 5):
            try:
                print(f"  -> asking {model} (attempt {attempt})")
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                if resp.parsed is not None:
                    return resp.parsed
                # Fallback: parse the raw text ourselves
                return schema.model_validate(json.loads(resp.text))

            except errors.APIError as e:
                code = getattr(e, "code", None)
                print(f"     Gemini error {code}: {str(e)[:300]}")
                if code in (401, 403):
                    sys.exit("\nERROR: Gemini rejected the key. Check the GEMINI_API_KEY secret "
                             "in GitHub (Settings -> Secrets and variables -> Actions).")
                if code == 404:
                    break                      # model not available -> try next model
                if code == 429 or (code and code >= 500):
                    wait = 30 * attempt        # busy / rate limited -> wait and retry
                    print(f"     waiting {wait}s ...")
                    time.sleep(wait)
                    continue
                break                          # other error -> try next model

            except Exception as e:  # bad JSON, network hiccup, etc. -> retry
                print(f"     Problem reading Gemini's answer: {str(e)[:300]}")
                time.sleep(10)

        print(f"  !! giving up on {model}, trying the next model")
        model_state["index"] += 1

    sys.exit("\nERROR: none of the Gemini models worked. See the messages above.")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHOT_RULES = """
Rules for every shot prompt (these go to an AI video model that makes 5-second clips):
- Write in English, 30-60 words, as ONE continuous moment that can be filmed in 5 seconds.
- Describe: the subject, what it is doing, the setting, the camera (e.g. slow push-in,
  aerial pan, low-angle tracking shot), and the light/time of day.
- NEVER use a character's name alone - the video model doesn't know who that is.
  Instead, describe the person using their fixed 'look' from the character list.
- No text, letters, subtitles, logos or maps with writing on screen.
- Keep violence cinematic but non-graphic: no gore, no blood close-ups, no severed limbs.
  Prefer wide shots, silhouettes, smoke, dust, banners, faces and reactions.
- The shots must match what the narration is saying at that moment.
- Do NOT include the overall visual style line; it is added automatically later.
"""


def outline_prompt(topic, minutes, language, n_chapters):
    return f"""You are a top YouTube documentary writer and director.
Plan a {minutes}-minute narrated cinematic video about: "{topic}"

The video is a documentary-style story (history, battles, mythology or mysteries) with a
narrator speaking all the time and cinematic AI-generated video clips on screen.
The voice-over language will be {language}.

Accuracy rules:
- History/documentary: stick to well-established facts, names, places and dates.
  If something is disputed, say it is disputed.
- Mythology: follow the traditional, widely known version of the story.
- Mysteries: clearly separate known facts from theories.

Create:
- a catchy title, YouTube description and tags (in {language}; tags may be mixed)
- one visual_style line (English) that every shot will share
- the main characters (if any) with a fixed visual 'look' each (English)
- exactly {n_chapters} chapters that tell the story from a strong hook to a satisfying ending.
  Chapter 1 must open with a hook that grabs attention in the first 15 seconds.
"""


def chapter_prompt(topic, language, outline, chapter_no, words, n_scenes, previous_tail):
    chapter = outline.chapters[chapter_no - 1]
    characters = "\n".join(f"- {c.name}: {c.look}" for c in outline.characters) or "- (none)"
    plan = "\n".join(f"{i}. {c.title}: {c.summary}" for i, c in enumerate(outline.chapters, 1))
    facts = "\n".join(f"- {f}" for f in chapter.key_facts) or "- (none)"

    lang_note = ("Write the narration in natural, simple spoken Hindi using Devanagari script "
                 "(the way a good Hindi documentary narrator speaks)."
                 if language == "Hindi" else
                 "Write the narration in clear, simple spoken English.")

    return f"""You are writing the narration and shot list for a cinematic YouTube documentary
about "{topic}" titled "{outline.title}".

Full plan of the video:
{plan}

Characters and their FIXED looks (use these looks in shots, never the bare name):
{characters}

NOW WRITE CHAPTER {chapter_no}: "{chapter.title}"
What happens: {chapter.summary}
Key facts to include:
{facts}

The narration from the end of the previous chapter (continue smoothly from here):
{previous_tail or "(this is the start of the video)"}

Requirements:
- {lang_note}
- About {words} words of narration in total for this chapter, split into about {n_scenes} scenes.
- Each scene: 1-2 sentences of narration (about {WORDS_PER_SCENE} words) and exactly
  {SHOTS_PER_SCENE} shot prompts showing what the narration describes.
- Storytelling tone: dramatic, vivid, emotional, like a great documentary narrator.
  Use short sentences. No headings, no "in this video", no asking viewers to subscribe.
{SHOT_RULES}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    topic = os.environ.get("TOPIC", "").strip()
    minutes = int(os.environ.get("MINUTES", "15") or 15)
    language = os.environ.get("LANGUAGE", "English").strip() or "English"

    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY is missing. Add it in GitHub Secrets.")
    if not topic:
        sys.exit("ERROR: no topic given.")

    target_words = minutes * WORDS_PER_MINUTE.get(language, 150)
    n_chapters = max(3, round(minutes / 3))
    words_per_chapter = target_words // n_chapters
    scenes_per_chapter = max(3, round(words_per_chapter / WORDS_PER_SCENE))

    print(f"Topic: {topic}")
    print(f"Length: {minutes} min | language: {language} | ~{target_words} words | "
          f"{n_chapters} chapters x ~{scenes_per_chapter} scenes")

    client = genai.Client(api_key=api_key)
    models = find_models(client)
    print(f"Models to try: {', '.join(models)}")
    model_state = {"models": models, "index": 0}

    # 1) Outline
    print("\n[1] Planning the video ...")
    outline = ask_gemini(client, outline_prompt(topic, minutes, language, n_chapters),
                         Outline, model_state)
    print(f"  Title: {outline.title}")
    print(f"  Chapters: {len(outline.chapters)} | characters: {len(outline.characters)}")

    # 2) Scenes, chapter by chapter
    all_scenes = []
    previous_tail = ""
    for i in range(1, len(outline.chapters) + 1):
        time.sleep(PAUSE_BETWEEN_CALLS)
        print(f"\n[2.{i}] Writing chapter {i}/{len(outline.chapters)}: {outline.chapters[i-1].title}")
        result = ask_gemini(
            client,
            chapter_prompt(topic, language, outline, i, words_per_chapter,
                           scenes_per_chapter, previous_tail),
            ChapterScenes, model_state,
        )
        for s in result.scenes:
            narration = s.narration.strip()
            shots = [x.strip() for x in s.shots if x.strip()]
            if not narration:
                continue
            if not shots:
                shots = [narration]  # last-resort fallback
            all_scenes.append({
                "id": len(all_scenes) + 1,
                "chapter": i,
                "narration": narration,
                "shots": shots,
            })
        previous_tail = " ".join(s["narration"] for s in all_scenes[-2:])
        print(f"  -> {len(result.scenes)} scenes")

    # 3) Save
    total_words = sum(len(s["narration"].split()) for s in all_scenes)
    est_minutes = total_words / WORDS_PER_MINUTE.get(language, 150)

    script = {
        "topic": topic,
        "language": language,
        "target_minutes": minutes,
        "estimated_minutes": round(est_minutes, 1),
        "total_words": total_words,
        "title": outline.title,
        "youtube_description": outline.youtube_description,
        "tags": outline.tags,
        "visual_style": outline.visual_style,
        "characters": [c.model_dump() for c in outline.characters],
        "chapters": [c.model_dump() for c in outline.chapters],
        "scenes": all_scenes,
        "model_used": models[min(model_state["index"], len(models) - 1)],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (OUTPUT_DIR / "narration.txt").write_text(
        f"{outline.title}\n\n" + "\n\n".join(s["narration"] for s in all_scenes),
        encoding="utf-8",
    )

    print(f"\nDONE: {len(all_scenes)} scenes, {total_words} words (~{est_minutes:.1f} min of voice)")
    print("Saved output/script.json and output/narration.txt")


if __name__ == "__main__":
    main()
