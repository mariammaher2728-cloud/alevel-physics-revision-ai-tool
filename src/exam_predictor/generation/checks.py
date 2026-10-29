"""Python finishes each question and lists the problems the model still has to fix.

- labels and marks come from the blueprint (extra parts are dropped), so the marks are always right;
- each calculation is worked out by Python; a part can use an earlier answer, e.g. ans_b_i;
- Python writes the mark codes itself: C1 for each working step and A1 for the calculated answer,
  B1 for each marking point of an explanation (with "max." when there are more points than marks);
- checks: enough marking points, a "Show that" value that matches its calculation, no made-up numbers,
  a graph formula that works, and no near-copies of past parts.
"""
import re

from ..text_utils import command_word, part_text
from .numbers import (DATA_SHEET, NUMBER, STATED, nice, numbers_in, pretty, run, same_value, to_number,
                      unknown_numbers)

CALC_WORDS = {"Calculate", "Determine", "Show that", "Estimate", "Deduce"}
NO_NUMBER = {"Define", "Explain", "Describe", "Suggest", "Sketch", "Draw", "Outline", "Compare", "Discuss", "Identify"}


def plain(line):
    """A mark-scheme line without its mark code(s): 'v = 2 pi r / T C1' -> 'v = 2 pi r / T'."""
    code = r"[\(\[]?[ABCM]\d[\)\]]?[:.]?"
    return re.sub(rf"^\s*{code}\s*$|^(\s*{code}\s)+|(\s+{code})+\s*$", "", line).strip()


def merge(steps, n):
    """At most n steps: extra steps are joined into the first one, so no marking point is lost."""
    if n <= 0:
        return []
    k = max(len(steps) - n + 1, 1)
    return ["; ".join(steps[:k])] + steps[k:]


def finish_question(qn, skeleton):
    """Python does the exact work: labels and marks from the blueprint `skeleton` [(label, marks, command), ...],
    answers from the calculations, and the mark codes. Returns the problems that the model still has to fix."""
    qn.stem = re.split(r"\s\((?:[a-h]|i{1,3}|iv|v)\)\s", qn.stem)[0].strip()       # the stem must not contain parts
    labels = [label for label, _, _ in skeleton]
    given = {p.label.replace(" ", ""): p for p in qn.parts}
    if all(label in given for label in labels):
        qn.parts = [given[label] for label in labels]                               # extra parts are dropped
    elif len(qn.parts) != len(labels):
        return [f"the question must have exactly these {len(labels)} parts: "
                + ", ".join(f"{l} [{m}]" for l, m, _ in skeleton)]
    problems, answers = [], {}
    text = " ".join([qn.stem, qn.figure] + [p.text for p in qn.parts])
    given = [abs(x) for x in numbers_in(text) if x] + list(DATA_SHEET.values())
    if qn.graph_y_formula:
        try:
            if qn.graph_x_to <= qn.graph_x_from:
                raise ValueError("graph_x_to must be larger than graph_x_from")
            run(qn.graph_y_formula, {"x": qn.graph_x_from}), run(qn.graph_y_formula, {"x": qn.graph_x_to})
        except Exception as e:
            problems.append(f"the graph does not work ({e}); give graph_y_formula as one Python expression of x")
            qn.graph_y_formula = ""
    for p, (label, marks, _) in zip(qn.parts, skeleton):
        p.label, p.marks, p.answer = label, marks, None
        command = command_word(p.text)
        if p.calculation and command not in NO_NUMBER:
            try:
                p.answer = run(p.calculation, answers)
                answers["ans_" + "_".join(re.findall(r"\w+", label))] = p.answer
                for x in unknown_numbers(p.calculation, given):
                    problems.append(f"part {label}: the calculation uses {nice(x)}, which is not in the question's data; "
                                    "give it in the stem, or write ans_ and the label to use an earlier answer")
            except Exception as e:
                if command in CALC_WORDS:                               # elsewhere a broken calculation is just dropped
                    problems.append(f"part {label}: the calculation does not run ({e}). Use one expression with numbers, "
                                    "the data names, sqrt, exp, log, sin, cos, pi and ans_ names of earlier parts")
        steps = [plain(line) for line in p.mark_scheme]
        if p.answer is not None and command != "Show that":           # a calculation: C1 steps, then Python's answer A1
            finals = [i for i, line in enumerate(p.mark_scheme) if re.search(r"A1\W*$", line)]
            steps = [s for s in (steps[:finals[0]] if finals else steps) if s]
            steps = [s for s in steps if "=" in s or not any(same_value(to_number(n), p.answer) for n in NUMBER.findall(s))]
            if len(steps) < marks - 1 and "= " + pretty(p.calculation) not in steps:
                steps.append("= " + pretty(p.calculation))              # the substitution step
            steps = merge(steps, marks - 1)
            p.mark_scheme = [f"{s} C1" for s in steps] + [f"{nice(p.answer)} {p.unit} A1".replace("  ", " ")]
            if len(steps) < marks - 1:
                problems.append(f"part {label} has {marks} marks: give {marks - 1} working steps before the answer")
            continue
        steps = list(dict.fromkeys(s for s in steps if s))                 # empty and repeated points do not count
        if command == "Show that":                                      # working steps, the last one A1
            stated = [to_number(n) for n in STATED.findall(re.sub(r"(\d) (?=\d{3}\b)", r"\1", p.text))]
            if p.answer is not None and stated and not any(same_value(s, p.answer) for s in stated):
                problems.append(f"part {label}: the value to show in the question does not match its calculation, "
                                f"which gives {nice(p.answer)} {p.unit}; change the value in the text or the data")
            substitution = f"= {pretty(p.calculation)} = {nice(p.answer)} {p.unit}".strip() if p.answer is not None else ""
            if len(steps) < marks and substitution and substitution not in steps:
                steps.append(substitution)
            steps = merge(steps, marks)
            p.mark_scheme = [f"{s} C1" for s in steps[:-1]] + [f"{s} A1" for s in steps[-1:]]
        else:                                                           # marking points, B1 each
            p.mark_scheme = [f"{s} B1" for s in steps] + ([f"max. {marks}"] if len(steps) > marks else [])
        if len(steps) < marks:
            problems.append(f"part {label} has {marks} marks but only {len(steps)} marking point(s); "
                            f"write {marks} separate points")
    return problems


def copy_check(qn, embedder, part_vecs, copy_limit=0.95):
    """Calculation and explanation parts must not be near-copies of past parts (definitions may repeat).
    Returns (problems, similarity of the closest past part)."""
    texts = [part_text(qn.stem, p.text, " ".join(p.mark_scheme)) for p in qn.parts]
    closest = (embedder(texts) @ part_vecs.T).max(axis=1)
    problems = [f"part {p.label} is almost a copy of a past question (similarity {c:.2f}); change its situation"
                for p, c in zip(qn.parts, closest) if c > copy_limit and command_word(p.text) not in {"Define", "State"}]
    return problems, float(closest.max())
