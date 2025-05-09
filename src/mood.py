"""
ai_reader_app.core.mood
-----------------------
Sentiment / emotion detector for paragraphs, powered by Hugging Face.

Exported helpers
================
* ``detect_mood(text, k=1, threshold=0.0)`` – returns the *k* most probable
  emotions above *threshold* (probability).  Default ``k=1`` & ``threshold=0``
  reproduces the old behaviour of a single best‑guess label.

Returned schema
---------------
```
[
    {"mood": "joy", "confidence": 0.88},
    {"mood": "love", "confidence": 0.07},
]
```
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from transformers import pipeline

# --------------------------
# model + pipeline handling
# --------------------------

@lru_cache(maxsize=1)
def _sentiment_pipeline():
    """Load + cache an emotion‑classification pipeline (≈120 MB)."""
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,  # return full distribution per sample
    )


# --------------------------
# public API
# --------------------------

def detect_mood(text: str, *, k: int = 1, threshold: float = 0.0) -> List[Dict[str, float]]:
    """Return a ranked list of emotions for *text*.

    Parameters
    ----------
    text : str
        Passage to analyse (first 512 chars considered).
    k : int, optional
        Number of top labels to keep (after thresholding).  ``k <= 0`` means
        *all* labels above ``threshold``.
    threshold : float, optional
        Minimum probability a label must reach to be returned.

    Examples
    --------
    >>> detect_mood("I can't believe how wonderful this adventure is!", k=3)
    [{'mood': 'joy', 'confidence': 0.964},
     {'mood': 'love', 'confidence': 0.012},
     {'mood': 'surprise', 'confidence': 0.010}]
    """
    # 1. Run the model (cached) – returns [[{label, score}, …]]
    preds = _sentiment_pipeline()(text[:512])[0]

    # 2. Filter by threshold
    filtered = [p for p in preds if p["score"] >= threshold]

    # 3. Sort descending by probability
    filtered.sort(key=lambda p: p["score"], reverse=True)

    # 4. Trim to k labels unless k <= 0 (meaning no limit)
    if k > 0:
        filtered = filtered[:k]

    # 5. Normalise output schema
    return [
        {
            "mood": p["label"].lower(),
            "confidence": round(float(p["score"]), 3),
        }
        for p in filtered
    ]


if __name__ == "__main__":
    _sample = (
        "The narrow alley felt cold and oppressive as shadows danced on the walls, yet she felt a flicker of hope."
    )
    print(detect_mood(_sample))
