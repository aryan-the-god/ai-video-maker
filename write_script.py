"""
STEP 1 of the video pipeline: write the script as a SCENE LIST.

Input  (environment variables, set by the GitHub workflow):
    GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY - AI keys (GitHub Secrets);
                      Gemini is tried first, the others are automatic backups
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

from pydantic import BaseModel, Field

from llm import AI

# Show every message in the GitHub log immediately (not only at the end).
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

WORDS_PER_MINUTE = {"English": 150, "Hindi": 130}  # normal narration speed
WORDS_PER_SCENE = 28          # about 10-12 seconds of voice per scene
SHOTS_PER_SCENE = 3           # 3 different shots per scene (each covers one part of the narration)
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
    shots: list[str] = Field(description="Video prompts for what the screen shows while this "
                                         "narration plays. English.")


class ChapterScenes(BaseModel):
    scenes: list[Scene]


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
- The shots must match what the narration is saying at that moment, IN THE SAME ORDER: shot 1 shows
  the first part of the narration, shot 2 the middle, shot 3 the last part.
- The AI video model is small. It draws big, clear, simple subjects well and small things badly.
  So prefer: landscapes, skies, crowds from far, silhouettes, buildings, fire, smoke, water, horses,
  armies, one person in a wide or medium shot with a clear action.
  AVOID: close-ups of small objects (conch shells, swords, rings, jewelry, coins, scrolls, weapons
  in hands), hands doing detailed things, anything with writing. If the narration mentions such an
  object, show the bigger scene around it instead (for example for a conch horn: warriors on a hill
  at dawn, a loud echo rolling over the valley).
- One shot = one simple subject and one simple movement. No lists of many things.
- Never show a real, named living person's face or body, and never use the names, looks, logos or
  places of copyrighted fictional worlds (film/book/game characters and locations). When the story
  is about such a person or franchise, show ORIGINAL generic visuals instead: places, objects,
  hands, silhouettes, crowds seen from far away, symbolic images, invented characters.
- Do NOT include the overall visual style line; it is added automatically later.
"""


STYLES = {
    "Cinematic realistic": "photorealistic cinematic film look, natural lighting, 35mm film, shallow depth of field",
    "Game cinematic (90s crime, open-world)": ("3D video game cinematic cutscene, open-world crime game, realistic "
        "game-engine rendering, 1990s city setting, moody golden-hour lighting, slight film grain, wide shots"),
    "Painterly animation": "painterly 2D animated film, rich saturated colors, hand-painted backgrounds, soft cinematic lighting",
}


def outline_prompt(topic, minutes, language, n_chapters, style_line=""):
    style_note = f" (it must be exactly this style: {style_line})" if style_line else ""
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

If the topic is a fictional story (for example a crime or gangster story) instead of history, tell it
as a gripping story with invented characters and a clear beginning, middle and end. Keep violence
non-graphic (no gore); show it through suspense, shadows and aftermath.

Create:
- a catchy title, YouTube description and tags (in {language}; tags may be mixed)
- one visual_style line (English) that every shot will share{style_note}
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
    topic = os.environ.get("TOPIC", "").strip()
    minutes = int(os.environ.get("MINUTES", "15") or 15)
    language = os.environ.get("LANGUAGE", "English").strip() or "English"
    style_line = STYLES.get(os.environ.get("STYLE", "").strip(), "")

    if not topic:
        sys.exit("ERROR: no topic given.")

    target_words = minutes * WORDS_PER_MINUTE.get(language, 150)
    n_chapters = max(3, round(minutes / 3))
    words_per_chapter = target_words // n_chapters
    scenes_per_chapter = max(3, round(words_per_chapter / WORDS_PER_SCENE))

    print(f"Topic: {topic}")
    print(f"Length: {minutes} min | language: {language} | ~{target_words} words | "
          f"{n_chapters} chapters x ~{scenes_per_chapter} scenes")

    ai = AI()

    # 1) Outline
    print("\n[1] Planning the video ...")
    outline = ai.ask(outline_prompt(topic, minutes, language, n_chapters, style_line), Outline)
    if style_line:
        outline.visual_style = style_line   # use the chosen style exactly
    print(f"  Title: {outline.title}")
    print(f"  Chapters: {len(outline.chapters)} | characters: {len(outline.characters)}")

    # 2) Scenes, chapter by chapter
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
        "model_used": ai.used(),
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
