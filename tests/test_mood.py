from ai_reader.src.mood import detect_mood


def test_detect_mood_schema():
    output = detect_mood("I love sunny days and cheerful company.")
    # Should be a list with at least one element
    assert isinstance(output, list) and output
    # Check structure of first item
    mood_item = output[0]
    # print(mood_item)
    assert set(mood_item.keys()) == {"mood", "confidence"}
    assert isinstance(mood_item["mood"], str)
    assert isinstance(mood_item["confidence"], float)
    assert 0.0 <= mood_item["confidence"] <= 1.0


def test_detect_mood_threshold_and_k():
    text = "I love sunny days and cheerful company."
    # Request top-3 moods with no threshold
    results = detect_mood(text, k=3, threshold=0.0)
    assert len(results) == 3
    # Request all moods above a high threshold => possibly empty or single
    high = detect_mood(text, k=0, threshold=0.9)
    for item in high:
        # print(item)
        assert item["confidence"] >= 0.9