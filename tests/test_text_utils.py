from exam_predictor.text_utils import command_word, part_text, sitting_label, squash


def test_sitting_label():
    assert sitting_label(2024 * 3 + 0) == "FM 2024"
    assert sitting_label(2024 * 3 + 1) == "MJ 2024"
    assert sitting_label(2025 * 3 + 2) == "ON 2025"


def test_command_word():
    assert command_word("(a) State what is meant by a photon.") == "Define"     # longest phrase wins
    assert command_word("(b)(ii) Show that the speed is 4.2 m s^-1.") == "Show that"
    assert command_word("(i) the radius of the orbit", context="Calculate:") == "Calculate"
    assert command_word("(c) Something without a command word.") == "Other"


def test_squash_and_part_text():
    assert squash("a\n  b\tc", 3) == "a b"
    text = part_text("context " * 100, "(a) Define g.", "force per unit mass B1")
    assert text.startswith("Context: ") and "\nQuestion: (a) Define g.\nMark scheme: force per unit mass B1" in text
