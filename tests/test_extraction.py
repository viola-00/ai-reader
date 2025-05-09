import pytest
from ai_reader.src.extraction import perform_ner


def test_perform_ner_returns_list_of_dicts():
    text = "Alice visited Paris."
    ents = perform_ner(text)
    # Should be a non-empty list
    assert isinstance(ents, list) and len(ents) > 0
    # Each item should be a dict with expected keys
    first = ents[0]
    assert set(first.keys()) == {"text", "type", "score"}
    assert isinstance(first["text"], str)
    assert first["type"] in {"PER", "LOC", "ORG", "MISC"}
    assert 0.0 <= first["score"] <= 1.0


def test_perform_ner_detects_known_names():
    text = "Harry and Hermione walked to Hogwarts."
    ents = perform_ner(text)
    names = {e["text"] for e in ents if e["type"] == "PER"}
    assert {"Harry", "Hermione"}.issubset(names)