"""
STEP 1 of the video pipeline: write the script as a SCENE LIST.

Input  (environment variables, set by the GitHub workflow):
    GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY - AI keys (GitHub Secrets);
                      Gemini is tried first, the others are automatic backups
    TOPIC           - a topic ("raja aur mantri ki sajish") OR a full pasted script
                      (long text: then the AI does not write a story, it only splits it into
                      scenes and plans one picture per scene)
    STYLE / COLOR   - picture look (see styles.py)
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

from pydantic import BaseModel, Field

from llm import AI
from styles import style_line as make_style_line

# Show every message in the GitHub log immediately (not only at the end).
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

WORDS_PER_MINUTE = {"English": 150, "Hindi": 130}  # normal narration speed
WORDS_PER_SCENE = 28          # about 12-14 seconds of voice per scene = one picture
SHOTS_PER_SCENE = 1           # one picture per scene
SCRIPT_MODE_MIN_CHARS = 400   # pasted text longer than this is treated as a finished script
SCENES_PER_CHAPTER_PASTED = 8
PAUSE_BETWEEN_CALLS = 8       # seconds between questions; gentler on free-tier limits

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# The shape of the answers we want from the AI
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
    shots: list[str] = Field(description="Exactly ONE image prompt: the picture shown while this "
                                         "narration plays. English.")


class ChapterScenes(BaseModel):
    scenes: list[Scene]


class ScriptPlan(BaseModel):
    title: str = Field(description="Catchy YouTube title, max 90 characters")
    youtube_description: str = Field(description="YouTube description, 3-5 sentences")
    tags: list[str] = Field(description="8-15 YouTube tags")
    characters: list[Character]


class ScenePrompts(BaseModel):
    prompts: list[str] = Field(description="One image prompt per numbered scene, in the same order")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHOT_RULES = """
Rules for every image prompt (each goes to an AI image model that paints ONE still picture):
- Write in English, 30-55 words, describing ONE still moment: who/what is in the picture, the
  pose or action, the facial expression, the place, the time of day, and the framing (wide shot,
  medium shot or close-up).
- The picture must show what the narration says at that moment, so it makes sense when heard.
- When a listed character appears, write the character as {Name} using the exact name from the
  character list, for example "{Raja} sits on his throne". Do NOT describe their face or clothes:
  the full fixed look is inserted automatically so the character looks the same in every picture.
  Never write a bare name. Other people (crowds, soldiers, villagers) are described normally.
- Keep the same place looking the same: while several scenes happen in one location, describe that
  location with the same key details each time (same hall, same colours, same time of day), and
  change only the framing and the action. When the story moves to a new place or a new character
  takes over, describe the new place fully.
- Vary the framing between neighbouring scenes (wide, medium, close-up) so the video feels alive.
- Big, clear subjects: people, buildings, landscapes, animals, fire, sky. Avoid tiny detailed
  objects, hands doing fine work, crowds of many faces, and anything with writing.
- No text, letters, numbers, signs, subtitles, logos or watermarks in the picture.
- Keep violence non-graphic: no gore, no blood. Show tension through faces, shadows and aftermath.
- Never show a real, named living person, and never use names, looks or places of copyrighted
  fictional worlds. Use original invented characters instead.
- Do NOT include the art style; it is added automatically later.
"""


def outline_prompt(topic, minutes, language, n_chapters, style_line=""):
    style_note = f" (it must be exactly this style: {style_line})" if style_line else ""
    return f"""You are a top YouTube storyteller and director.
Plan a {minutes}-minute narrated story video about: "{topic}"

The video is a story (folk tale, history, mythology or mystery) told by a warm, unhurried narrator
who speaks all the time, while one illustrated picture at a time is shown on screen (each picture
stays for about 12-14 seconds, then gently fades into the next).
The voice-over language will be {language}.

Accuracy rules:
- History/documentary: stick to well-established facts, names, places and dates.
  If something is disputed, say it is disputed.
- Mythology: follow the traditional, widely known version of the story.
- Mysteries: clearly separate known facts from theories.

If the topic is a fictional or folk story (for example a king and his minister's conspiracy), tell it
as a gripping story with invented characters, a clear beginning, middle and end, and a twist or
lesson. Keep violence non-graphic (no gore); show it through suspense, shadows and aftermath.

Create:
- a catchy title, YouTube description and tags (in {language}; tags may be mixed)
- one visual_style line (English) that every picture will share{style_note}
- the main characters (if any) with a fixed visual 'look' each (English): age, face, beard or
  hair, clothes and colours, so they can be drawn the same way every time
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

    return f"""You are writing the narration and picture list for a illustrated YouTube story video
about "{topic}" titled "{outline.title}".

Full plan of the video:
{plan}

Characters and their FIXED looks (in image prompts write only {Name}; the look is added automatically):
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
  {SHOTS_PER_SCENE} image prompt showing what the narration describes.
- Storytelling tone: warm, vivid, emotional, like a great storyteller speaking to listeners.
  Use short sentences. No headings, no "in this video", no asking viewers to subscribe.
{SHOT_RULES}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fill_looks(shot, characters):
    """Replace {Name} in a shot with that character's exact fixed look (keeps the face consistent)."""
    for c in characters:
        shot = re.sub(r"\{\s*" + re.escape(c.name) + r"\s*\}", c.look.rstrip(". "), shot, flags=re.I)
    return shot


def split_into_scenes(text):
    """Cut a finished script into narration pieces of about WORDS_PER_SCENE words."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [x.strip() for x in re.split(r"(?<=[\u0964!?.])\s+", text) if x.strip()]
    scenes, current = [], []
    for sent in sentences:
        current.append(sent)
        if sum(len(x.split()) for x in current) >= WORDS_PER_SCENE:
            scenes.append(" ".join(current))
            current = []
    if current:
        if scenes and sum(len(x.split()) for x in current) < WORDS_PER_SCENE // 3:
            scenes[-1] += " " + " ".join(current)
        else:
            scenes.append(" ".join(current))
    return scenes


def plan_prompt(script_text, language, style_line):
    return f"""Here is a finished narration script for a YouTube story video (voice-over language: {language}).

SCRIPT:
{script_text[:30000]}

Create: a catchy title and YouTube description (in {language}), 8-15 tags, and the main characters
with a fixed visual 'look' each (English, 20-40 words: age, face, beard or hair, clothes, colours),
so the pictures can draw them the same way every time. Art style (do not describe it): {style_line}
"""


def prompts_prompt(plan, scenes, first_no):
    characters = "\n".join(f"- {c.name}: {c.look}" for c in plan.characters) or "- (none)"
    numbered = "\n".join(f"{first_no + i}. {x}" for i, x in enumerate(scenes))
    return f"""You are planning the pictures for an illustrated story video titled "{plan.title}".
Each numbered narration below is spoken for about 13 seconds while ONE picture is shown.

Characters and their FIXED looks (in prompts write only {{Name}}; the look is added automatically):
{characters}

NARRATION (scenes {first_no} to {first_no + len(scenes) - 1}):
{numbered}

Write exactly {len(scenes)} image prompts, one per numbered scene, in the same order.
{SHOT_RULES}"""


def main():
    topic = os.environ.get("TOPIC", "").strip()
    minutes = int(os.environ.get("MINUTES", "15") or 15)
    language = os.environ.get("LANGUAGE", "English").strip() or "English"
    style_line = make_style_line(os.environ.get("STYLE", "").strip(),
                                 os.environ.get("COLOR", "").strip())

    if not topic:
        sys.exit("ERROR: no topic given.")

    ai = AI()
    pasted = len(topic) >= SCRIPT_MODE_MIN_CHARS

    if pasted:
        all_scenes, outline_info = pasted_script_scenes(ai, topic, language, style_line)
        title, desc, tags = outline_info["title"], outline_info["desc"], outline_info["tags"]
        characters, chapters = outline_info["characters"], []
        topic_label = title
    else:
        all_scenes, outline_info = written_story_scenes(ai, topic, minutes, language, style_line)
        title, desc, tags = outline_info["title"], outline_info["desc"], outline_info["tags"]
        characters, chapters = outline_info["characters"], outline_info["chapters"]
        topic_label = topic

    total_words = sum(len(s["narration"].split()) for s in all_scenes)
    est_minutes = total_words / WORDS_PER_MINUTE.get(language, 150)

    script = {
        "topic": topic_label,
        "language": language,
        "target_minutes": minutes,
        "estimated_minutes": round(est_minutes, 1),
        "total_words": total_words,
        "title": title,
        "youtube_description": desc,
        "tags": tags,
        "visual_style": style_line,
        "characters": [c.model_dump() for c in characters],
        "chapters": [c.model_dump() for c in chapters],
        "scenes": all_scenes,
        "model_used": ai.used(),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (OUTPUT_DIR / "narration.txt").write_text(
        f"{title}\n\n" + "\n\n".join(s["narration"] for s in all_scenes),
        encoding="utf-8",
    )

    print(f"\nDONE: {len(all_scenes)} scenes, {total_words} words (~{est_minutes:.1f} min of voice)")
    print("Saved output/script.json and output/narration.txt")


def pasted_script_scenes(ai, text, language, style_line):
    """The user pasted a finished script: keep the words, only plan the pictures."""
    print("Mode: pasted script (the AI will NOT rewrite your words)")
    pieces = split_into_scenes(text)
    print(f"  Cut into {len(pieces)} scenes")

    print("\n[1] Reading the script (title, characters) ...")
    plan = ai.ask(plan_prompt(text, language, style_line), ScriptPlan)
    print(f"  Title: {plan.title} | characters: {len(plan.characters)}")

    all_scenes = []
    chunk = 15
    for start in range(0, len(pieces), chunk):
        time.sleep(PAUSE_BETWEEN_CALLS)
        part = pieces[start:start + chunk]
        print(f"[2] Planning pictures for scenes {start + 1}-{start + len(part)} ...")
        result = ai.ask(prompts_prompt(plan, part, start + 1), ScenePrompts)
        prompts = [p.strip() for p in result.prompts]
        for k, narration in enumerate(part):
            prompt = prompts[k] if k < len(prompts) and prompts[k] else narration
            n = len(all_scenes) + 1
            all_scenes.append({
                "id": n,
                "chapter": (n - 1) // SCENES_PER_CHAPTER_PASTED + 1,
                "narration": narration,
                "shots": [fill_looks(prompt, plan.characters)],
            })
    info = {"title": plan.title, "desc": plan.youtube_description, "tags": plan.tags,
            "characters": plan.characters}
    return all_scenes, info


def written_story_scenes(ai, topic, minutes, language, style_line):
    """Only a topic was given: the AI writes the story and plans the pictures."""
    target_words = minutes * WORDS_PER_MINUTE.get(language, 150)
    n_chapters = max(3, round(minutes / 3))
    words_per_chapter = target_words // n_chapters
    scenes_per_chapter = max(3, round(words_per_chapter / WORDS_PER_SCENE))

    print(f"Topic: {topic}")
    print(f"Length: {minutes} min | language: {language} | ~{target_words} words | "
          f"{n_chapters} chapters x ~{scenes_per_chapter} scenes")

    print("\n[1] Planning the story ...")
    outline = ai.ask(outline_prompt(topic, minutes, language, n_chapters, style_line), Outline)
    print(f"  Title: {outline.title}")
    print(f"  Chapters: {len(outline.chapters)} | characters: {len(outline.characters)}")

    all_scenes = []
    previous_tail = ""
    for i in range(1, len(outline.chapters) + 1):
        time.sleep(PAUSE_BETWEEN_CALLS)
        print(f"\n[2.{i}] Writing chapter {i}/{len(outline.chapters)}: {outline.chapters[i-1].title}")
        result = ai.ask(
            chapter_prompt(topic, language, outline, i, words_per_chapter,
                           scenes_per_chapter, previous_tail),
            ChapterScenes,
        )
        for s in result.scenes:
            narration = s.narration.strip()
            shots = [x.strip() for x in s.shots if x.strip()][:1]
            if not narration:
                continue
            if not shots:
                shots = [narration]  # last-resort fallback
            shots = [fill_looks(x, outline.characters) for x in shots]
            all_scenes.append({
                "id": len(all_scenes) + 1,
                "chapter": i,
                "narration": narration,
                "shots": shots,
            })
        previous_tail = " ".join(s["narration"] for s in all_scenes[-2:])
        print(f"  -> {len(result.scenes)} scenes")

    info = {"title": outline.title, "desc": outline.youtube_description, "tags": outline.tags,
            "characters": outline.characters, "chapters": outline.chapters}
    return all_scenes, info


if __name__ == "__main__":
    main()
