"""
ai_reader.prompt
================
Helpers to build prompts for **both** image and ambient‑audio generation,
leveraging extracted entities and detected mood.

Public API
----------
* ``build_image_prompt(text, style=None)`` – concise Stable Diffusion / DALLE‑3 prompt.
* ``build_audio_prompt(text)`` – mood‑mapped descriptor string for a 20‑30 s loop.
"""

from __future__ import annotations

from typing import List, Tuple

from extraction import perform_ner, extract_action
from mood import detect_mood

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_DEF_IMG_STYLE = (
    "highly detailed digital art, cinematic lighting, 8k resolution, trending on ArtStation"
)

MOOD_AUDIO_MAP = {
    "joy": "bright strings and upbeat tempo",
    "sadness": "slow piano with reverb",
    "fear": "dissonant drones and suspenseful pulses",
    "anger": "heavy percussion and distorted synths",
    "surprise": "sharp staccato notes",
    # fallback handled below
}

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _partition_entities(entities: List[dict]) -> Tuple[List[str], List[str]]:
    """Return (people, locations) lists from entity dicts."""
    people, places = [], []
    for ent in entities:
        if ent["type"] == "PER":
            people.append(ent["text"])
        elif ent["type"] == "LOC":
            places.append(ent["text"])
    return people, places

# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def build_image_prompt(
    text: str,
    *,
    style: str | None = None,
    include_context: bool = False,
) -> str:
    """Return a prompt string for Stable Diffusion / GPT‑Image‑1.

    * Extracts entities (PER, LOC)
    * Extracts main action phrase
    * Adds mood tag and default style block
    * Optionally prepends a CONTEXT section for GPT‑Image‑1
    """
    ents = perform_ner(text)
    mood = detect_mood(text)[0]["mood"]
    action = extract_action(text)
    people, places = _partition_entities(ents)

    parts: List[str] = []
    parts.extend(people or ["a figure"])
    if action:
        parts.append(action)
    if places:
        parts.append("in " + ", ".join(places))
    parts.append(f"mood: {mood}")
    parts.append(_DEF_IMG_STYLE)
    if style:
        parts.append(style)

    instructions = "; ".join(parts)

    if not include_context:
        return instructions

    context = text.strip()
    return f"### CONTEXT\n{context}\n\n### IMAGE INSTRUCTIONS\n{instructions}"


def build_audio_prompt(text: str) -> str:
    """Return a one‑line descriptor for an ambient loop based on mood."""
    mood = detect_mood(text)[0]["mood"]
    desc = MOOD_AUDIO_MAP.get(mood, "ambient soundscape")
    return f"{desc} loop"

# -----------------------------------------------------------------------------
# CLI demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "Breathing hard, Harry and Ron sprinted across the rain-slick platform toward the scarlet Hogwarts Express as its whistle shrieked, sparks cascading from the engine’s smokestack."
    print("IMAGE PROMPT:\n", build_image_prompt(sample, include_context=False))
    print("\nAUDIO PROMPT:\n", build_audio_prompt(sample))