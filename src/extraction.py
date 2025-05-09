"""
ai_reader_app.core.extraction
---------------------------
Utility functions for Named Entity Recognition (NER) using
Hugging Face transformers.  Import `perform_ner` wherever you need
character / place / object extraction.
"""

from functools import lru_cache
from typing import Dict, List

from transformers import pipeline


@lru_cache(maxsize=1)
def _ner_pipeline():
    """Load + cache the NER pipeline so weights are fetched only once."""
    return pipeline(
        task="ner",  # instructs the pipeline what to do
        model="dslim/bert-base-NER",  # lightweight BERT fine‑tuned for NER
        aggregation_strategy="simple",  # merge sub‑tokens into whole words
    )


def perform_ner(text: str) -> List[Dict[str, str]]:
    """Return simplified entities for *text*.

    Parameters
    ----------
    text : str
        The passage to analyse.

    Returns
    -------
    List[Dict[str, str]]
        Each dict has `text`, `type` (PER/LOC/ORG/MISC) and `score`.
    """
    raw = _ner_pipeline()(text)
    # Reshape the pipeline output into a clean, predictable schema
    simplified = [
        {
            "word": ent["word"],
            "entity": ent["entity_group"],
            "score": float(ent["score"]),
        }
        for ent in raw
    ]
    # Reconstruct entities from subwords
    return reconstruct_entities(simplified)

def reconstruct_entities(ents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Reconstruct entities by merging subwords into full words.

    Parameters
    ----------
    ents : List[Dict[str, str]]
        List of token-level entities.

    Returns
    -------
    List[Dict[str, str]]
        List of reconstructed entities with `text`, `type`, and `score`.
    """
    reconstructed = []
    current_entity = None

    for e in ents:
        if e["word"].startswith("##"):
            # Append subword to the current entity
            current_entity["text"] += e["word"][2:]
            current_entity["score"] = min(current_entity["score"], e["score"])  # Use the lowest score
        else:
            # Start a new entity
            if current_entity:
                reconstructed.append(current_entity)
            current_entity = {"text": e["word"], "type": e["entity"], "score": e["score"]}

    # Add the last entity
    if current_entity:
        reconstructed.append(current_entity)

    return reconstructed

if __name__ == "__main__":
    _sample = "Harry met Hermione in London."
    for e in perform_ner(_sample):
        print(f"{e['text']:<10} {e['type']:<4} {e['score']:.3f}")
