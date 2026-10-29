"""Past-question examples (the RAG part) and the prompts sent to the model.

Keep the prompt wording unchanged if you want to reuse replies already saved in llm_cache.json:
the cache key includes the full prompt text.
"""
import json
import re

from ..analysis import question_table
from ..text_utils import squash
from .numbers import DATA_SHEET


def examples_for(pool, topic, outcomes, n=3):
    """Real past questions on this topic (with mark schemes), best matching the chosen outcomes, from different sittings."""
    qs = question_table(pool)
    qs = qs[qs.topic == topic].copy()
    hits = pool[pool.lo.isin(outcomes)].groupby(["paper_id", "question_number"]).size()
    qs["hits"] = [hits.get((p, q), 0) for p, q in zip(qs.paper_id, qs.question_number)]
    qs = qs.sort_values(["hits", "sitting_idx"], ascending=False).drop_duplicates("sitting_idx").head(n)
    examples = []
    for p, q in zip(qs.paper_id, qs.question_number):
        parts = pool[(pool.paper_id == p) & (pool.question_number == q)]
        first = parts.iloc[0]
        scheme = "\n".join(f"{r.part}: {squash(r.answer, 250)}" for r in parts.itertuples())
        question = re.sub(r"\n{2,}", "\n", first.full_question)[:2000]
        examples.append(f"### Past question ({first.sitting}, paper {first.component}, Q{q})\n{question}\n\nMark scheme:\n{scheme}")
    return examples


def reply_template(skeleton):
    """The exact JSON to fill in: the blueprint's parts, with one mark-scheme line per mark."""
    parts = []
    for label, marks, command in skeleton:
        numeric = command in {"Calculate", "Determine", "Estimate", "Deduce", "Show that"}
        steps = marks - 1 if numeric and command != "Show that" else marks
        word = command if command not in {"Other", "Complete", "Plot", "Draw"} else ""
        parts.append({"label": label, "text": f"{word} ...".strip(),
                      "mark_scheme": [f"working step {i + 1}" if numeric else f"marking point {i + 1}" for i in range(steps)],
                      "unit": "..." if numeric else "", "calculation": "Python expression" if numeric else ""})
    return json.dumps({"stem": "...", "figure": "", "graph_x_label": "", "graph_y_label": "", "graph_x_from": 0,
                       "graph_x_to": 0, "graph_y_formula": "", "parts": parts}, ensure_ascii=False)


def question_prompt(row, examples):
    """The prompt for one blueprint question (row has topic, marks, outcome_text and skeleton)."""
    outcomes = "\n".join(f"- {t}" for t in row.outcome_text)
    constants = ", ".join(f"{k} = {v:.3g}" for k, v in DATA_SHEET.items())
    return f"""You are an experienced Cambridge International A Level Physics (9702) examiner.
Write ONE new question for a Paper 4 (A Level Structured Questions) practice paper.

Topic: {row.topic}
Learning outcomes to assess:
{outcomes}

Rules:
1. Use a NEW situation and NEW numbers. The past questions below are only for style: do not copy their sentences.
2. The stem gives the situation and all the data the parts need. It never contains the part questions.
3. Cambridge data values you may use: {constants}.
4. Write powers as 10^5 and units as m s^-1. Do not use LaTeX.
5. Mark scheme: one line per marking point or working step, WITHOUT mark codes (Python adds B1, C1 and A1).
   A part with 2 marks needs 2 separate marking points.
6. Never work out numbers yourself. For a numerical answer write ONE Python expression in "calculation" (numbers, the
   data names above, sqrt, exp, log, sin, cos, pi, powers with **) and the unit in "unit". Its mark scheme lists only the
   working steps; Python adds the final answer line. To use the answer of an earlier part write ans_ and its label, e.g. ans_b_i.
7. In a "Show that" part, state the value in the text (e.g. "Show that the speed is about 4.2 m s^-1") and give the
   calculation that gives it.
8. Never refer to a table, graph or diagram that you do not give. Describe a needed diagram in "figure". Give a graph only
   if a part needs values read from it: graph_y_formula is a Python expression of x, drawn from graph_x_from to graph_x_to.
   Never draw the answer to a Sketch part.
9. Every part must be correct physics and answerable from the data given. Keep each part under 60 words.

Past questions on this topic (real exams with their mark schemes, for style only):

{chr(10).join(examples)}

Your question must have exactly these parts and marks ({row.marks} marks in total). Fill in this JSON:
{reply_template(row.skeleton)}
"""


def fix_prompt(qn, problems, skeleton):
    """Send the question back with its problems; the model corrects only those."""
    return ("You wrote this Cambridge International A Level Physics (9702) exam question. Python has already added the "
            "mark codes and the calculated answers:\n"
            + qn.model_dump_json(exclude={"parts": {"__all__": {"marks", "answer"}}}) +
            "\n\nIt has these problems:\n" + "\n".join(f"- {p}" for p in problems) +
            f"\n\nThe parts must be exactly: {', '.join(f'{l} [{m}]' for l, m, _ in skeleton)}. "
            "Return the whole question as corrected JSON in the same format. Change only what is needed to fix the problems.")


def as_text(qn):
    """The question and its mark scheme as plain text, for the examiner review."""
    lines = [qn.stem] + ([f"[Figure: {qn.figure}]"] if qn.figure else [])
    if qn.graph_y_formula:
        lines.append(f"[Graph of {qn.graph_y_label} against {qn.graph_x_label}: "
                     f"y = {qn.graph_y_formula} for x from {qn.graph_x_from} to {qn.graph_x_to}]")
    for p in qn.parts:
        lines += [f"{p.label} {p.text} [{p.marks}]", "    mark scheme: " + " | ".join(p.mark_scheme)]
    return "\n".join(lines)


def review_prompt(qn):
    """A senior examiner reads the finished question and lists serious errors only."""
    return ("You are a senior Cambridge International A Level Physics (9702) examiner checking a new question before "
            "it is printed:\n\n" + as_text(qn) +
            "\n\nList only SERIOUS errors, at most 3: wrong physics, a part that cannot be answered from the data given, "
            "a reference to a table or diagram that is not given, or a mark scheme that does not answer its part. "
            "The numbers in the mark scheme were calculated by Python from the question's own expressions. Ignore style. "
            'If there are no serious errors, reply {"problems": []}.\nReply format: {"problems": ["...", "..."]}')
