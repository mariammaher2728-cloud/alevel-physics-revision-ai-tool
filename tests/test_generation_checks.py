"""The exact work Python does on each generated question (no AI model needed)."""
import math

from exam_predictor.generation.checks import finish_question, merge, plain
from exam_predictor.generation.numbers import nice, numbers_in, pretty, run, same_value, to_number, unknown_numbers
from exam_predictor.generation.schemas import NewQuestion


def question(parts, stem="A body of mass 2.0 kg hangs from a spring.", **extra):
    return NewQuestion.model_validate({"stem": stem, "parts": parts, **extra})


# ----- numbers -----
def test_reading_numbers():
    assert to_number("2.5 × 10^3") == 2500
    assert math.isclose(to_number("1.2 x 10^–4 m"), 1.2e-4)
    assert numbers_in("is 12 000 K and 2.5 × 10^3 Pa") == [12000, 2500]


def test_nice_numbers():
    assert nice(8943000) == "8.94 × 10^6"
    assert nice(0.0125) == "0.0125"
    assert nice(47.619) == "47.6"
    assert nice(1234.5) == "1230"
    assert nice(-1.419e7) == "-1.42 × 10^7"


def test_same_value_allows_unit_prefixes():
    assert same_value(75, 0.075)            # 75 mA and 0.075 A
    assert same_value(31, 31.4)
    assert not same_value(13, 31.4)


def test_run_cleans_common_slips():
    assert math.isclose(run("import math; math.sqrt(4.0) * g"), 19.62)
    assert run("2^3") == 8                  # ^ means power, not XOR
    assert run("ans_b_i / 2", {"ans_b_i": 10}) == 5


def test_pretty_and_unknown_numbers():
    assert pretty("(120*180)/(2.5*ans_b_i)") == "(120 × 180)/(2.5 × (b)(i))"
    assert unknown_numbers("3.90e-19 / h", given=[6.10e-19]) == [3.9e-19]      # a made-up value
    assert unknown_numbers("0.5 * 2.0 * 6.0**2", given=[2.0, 6.0]) == []       # data and everyday numbers are fine
    assert unknown_numbers("200e-6 * 160", given=[200, 160]) == []             # unit prefixes are fine


# ----- mark-scheme helpers -----
def test_plain_removes_mark_codes():
    assert plain("v = 2 pi r / T C1") == "v = 2 pi r / T"
    assert plain("(B1) force per unit mass") == "force per unit mass"
    assert plain("B1") == ""
    assert plain("charge on C1 is 4.0 μC") == "charge on C1 is 4.0 μC"   # a capacitor name is not a code


def test_merge_keeps_every_point():
    assert merge(["a", "b", "c", "d"], 2) == ["a; b; c", "d"]
    assert merge(["a"], 3) == ["a"]
    assert merge(["a", "b"], 0) == []


# ----- finish_question -----
SKELETON = [("(a)", 2, "Define"), ("(b)(i)", 3, "Calculate"), ("(b)(ii)", 2, "Explain")]


def test_messy_reply_is_finished_without_problems():
    qn = question(stem="A body of mass 2.0 kg hangs from a spring. (c) A copied part of a past paper.",
                  parts=[{"label": "(a)", "text": "Define weight.", "mark_scheme": ["gravitational force", "on a mass"]},
                         {"label": "(b)(i)", "text": "Calculate the weight of the body.",
                          "mark_scheme": ["W = m g", "C1", "19.6 N A1", "19.6 N A1"],
                          "unit": "N", "calculation": "import math; math.sqrt(4.0) * g"},
                         {"label": "(b)(ii)", "text": "Explain why the spring stretches.", "mark_scheme": ["p1", "p2"]},
                         {"label": "(z)", "text": "An extra part.", "mark_scheme": ["x"]}])
    assert finish_question(qn, SKELETON) == []
    assert qn.stem == "A body of mass 2.0 kg hangs from a spring."               # copied text cut off
    assert [p.label for p in qn.parts] == ["(a)", "(b)(i)", "(b)(ii)"]          # extra part dropped
    assert [p.marks for p in qn.parts] == [2, 3, 2]
    assert qn.parts[0].mark_scheme == ["gravitational force B1", "on a mass B1"]
    assert qn.parts[1].mark_scheme == ["W = m g C1", "= sqrt(4.0) × g C1", "19.6 N A1"]


def test_show_that_value_must_match():
    skeleton = [("(a)", 2, "Show that")]
    wrong = question([{"label": "(a)", "text": "Show that the weight is about 25 N.", "mark_scheme": ["W = m g"],
                       "unit": "N", "calculation": "2.0 * g"}])
    assert "does not match its calculation" in finish_question(wrong, skeleton)[0]
    prefix = question([{"label": "(a)", "text": "Show that the current is 75 mA.", "mark_scheme": ["I = V / R"],
                        "unit": "A", "calculation": "12.0 / 160"}], stem="12.0 V across 160 ohm.")
    assert finish_question(prefix, skeleton) == []
    algebra = question([{"label": "(a)", "text": "Show that M = 4π^2 r^3 / (G T^2).",
                         "mark_scheme": ["G M m / r^2 = m v^2 / r", "v = 2 pi r / T"]}])
    assert finish_question(algebra, skeleton) == []


def test_made_up_number_and_earlier_answers():
    skeleton = [("(a)", 2, "Calculate"), ("(b)", 2, "Calculate")]
    made_up = question([{"label": "(a)", "text": "Calculate the weight.", "mark_scheme": ["W = m g"], "unit": "N",
                         "calculation": "2.0 * g"},
                        {"label": "(b)", "text": "Calculate half the weight.", "mark_scheme": ["W / 2"], "unit": "N",
                         "calculation": "19.6 / 2.3"}])
    assert "not in the question's data" in finish_question(made_up, skeleton)[0]
    linked = question([{"label": "(a)", "text": "Calculate the weight.", "mark_scheme": ["W = m g"], "unit": "N",
                        "calculation": "2.0 * g"},
                       {"label": "(b)", "text": "Calculate half the weight.", "mark_scheme": ["W / 2"], "unit": "N",
                        "calculation": "ans_a / 2"}])            # the answer of part (a)
    assert finish_question(linked, skeleton) == []
    assert linked.parts[1].mark_scheme[-1] == "9.81 N A1"


def test_problems_the_model_must_fix():
    repeated = question([{"label": "(a)", "text": "Define gravitational field strength.",
                          "mark_scheme": ["force per unit mass", "force per unit mass"]}])
    assert "only 1 marking point" in finish_question(repeated, [("(a)", 2, "Define")])[0]
    missing = question([{"label": "(a)", "text": "Define g.", "mark_scheme": ["p1"]}])
    assert "exactly these 2 parts" in finish_question(missing, [("(a)", 1, "Define"), ("(b)", 1, "State")])[0]
    graph = question([{"label": "(a)", "text": "State the period.", "mark_scheme": ["0.02 s"]}],
                     graph_x_from=0, graph_x_to=0.04, graph_y_formula="12*sin(100*pi*x")
    assert "the graph does not work" in finish_question(graph, [("(a)", 1, "State")])[0]
    assert graph.graph_y_formula == ""
