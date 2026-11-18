"""Look of the pictures: one style + one colour choice become one line added to every image prompt."""

STYLE = {
    "storybook": "flat 2D digital storybook illustration for an Indian Hindi story video, "
                 "clean bold outlines, smooth cel shading, natural human proportions, "
                 "expressive faces, detailed background painting, not photorealistic",
    "inkwash": "vintage hand-drawn book illustration, detailed pen and ink linework with "
               "crosshatching, soft sepia watercolor wash, aged yellowed paper texture, "
               "historical storybook look, highly detailed, natural human proportions",
    "painterly": "traditional oil painting on canvas, visible brush strokes, dramatic warm "
                 "light, moody atmosphere, painterly textures, not photorealistic",
    "cartoon": "2D animated cartoon frame in the style of Indian animated story channels, "
               "clean outlines, flat colors, soft shadows, friendly expressive faces",
}

COLOR = {
    "colored": "rich natural colors",
    "vintage": "sepia tone like an aged parchment map, faded ink, old yellowed paper, "
               "muted warm brown colors",
}


def style_line(style, color):
    s = STYLE.get(style, STYLE["storybook"])
    c = COLOR.get(color, COLOR["colored"])
    return f"{s}, {c}"
