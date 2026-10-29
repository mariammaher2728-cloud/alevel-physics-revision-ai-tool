"""Reading the model's JSON replies (no AI model needed)."""
import pytest

from exam_predictor.generation.schemas import NewQuestion, Review
from exam_predictor.llm import drop_nulls, read_json


def test_drop_nulls():
    assert drop_nulls({"a": None, "b": [1, None, {"c": None, "d": 2}]}) == {"b": [1, {"d": 2}]}


def test_messy_reply_is_read():
    reply = ('{"$defs": {"Part": {}}}\nSure! Here it is:\n```json\n'
             '{"stem": "R = 5 \\Omega", "figure": null, "parts": [{"label": "(a)", "text": "t", "mark_scheme": "a; b"}]}\n```')
    qn = NewQuestion.model_validate(read_json(reply, NewQuestion))   # skips the echoed schema, repairs \Omega and null
    assert qn.stem == "R = 5 \\Omega" and qn.figure == "" and qn.parts[0].mark_scheme == ["a", "b"]


def test_unusable_replies_explain_why():
    with pytest.raises(ValueError, match="no JSON object"):
        read_json("Sorry, I cannot do that.", NewQuestion)
    with pytest.raises(ValueError, match="cut off"):
        read_json('{"stem": "x", "parts": [{"label": "(a)"', NewQuestion)


def test_review_reply():
    assert Review.model_validate(read_json('{"problems": []}', Review)).problems == []
