"""The writing loop for every blueprint question.

1. Retrieve up to 3 real past questions on the same topic (RAG) and ask for a new question in a fixed JSON template.
2. Python finishes it (checks.py). If Python finds nothing wrong, the model reviews it as a senior examiner.
3. Any problem is sent back with the question, and the model corrects only that ("fix, don't restart").
   There are up to max_tries attempts; the best version is kept, and anything still wrong is printed as CHECK.
Each attempt prints one line, e.g. "Q3 attempt 1: 1 problem(s) (70 s)", followed by its problems.
Good replies are cached, so re-running is instant and only retries what is still missing.
"""
import time

import pandas as pd
from tqdm.auto import tqdm

from .checks import copy_check, finish_question
from .prompts import examples_for, fix_prompt, question_prompt, review_prompt
from .schemas import NewQuestion, Review


def write_paper(blueprint, pool, llm, embedder, part_vecs, max_tries=3, copy_limit=0.95, review=True):
    """Write one question per blueprint row. Returns a list of dicts: row, question, problems, closest_past."""
    generated = []
    for row in tqdm(list(blueprint.itertuples()), desc="Writing questions"):
        prompt, best = question_prompt(row, examples_for(pool, row.topic, row.outcomes)), None
        for attempt in range(max_tries):
            start = time.time()
            try:
                qn = llm.ask(prompt, NewQuestion, temperature=0.6 if best is None else 0.4, attempt=attempt,
                             max_new_tokens=3000)
            except ValueError as e:
                print(f"Q{row.question} attempt {attempt + 1}: unusable reply ({time.time() - start:.0f} s) - {e}")
                continue
            problems = finish_question(qn, row.skeleton)
            copies, closest = copy_check(qn, embedder, part_vecs, copy_limit)
            problems += copies
            if not problems and review:                    # Python found nothing wrong: now an examiner reads it
                try:
                    problems = ["examiner: " + p for p in llm.ask(review_prompt(qn), Review, max_new_tokens=400).problems[:3]]
                except ValueError as e:
                    print("  (the review reply could not be read:", e, ")")
            print(f"Q{row.question} attempt {attempt + 1}: {len(problems)} problem(s) ({time.time() - start:.0f} s)"
                  + "".join(f"\n    - {p}" for p in problems))
            if best is None or len(problems) <= len(best["problems"]):     # ties: the newer version is kept
                best = {"row": row, "question": qn, "problems": problems, "closest_past": closest}
            if not problems:
                break
            prompt = fix_prompt(best["question"], best["problems"], row.skeleton)   # the next attempt corrects the best
        if best is None or not all(p.marks for p in best["question"].parts):
            print(f"Q{row.question} ({row.topic}) could not be generated - run again to retry it.")
        else:
            generated.append(best)
    return generated


def summary_table(generated):
    """One row per written question: topic, marks, problems left and the similarity of the closest past part."""
    return pd.DataFrame([{"question": g["row"].question, "topic": g["row"].topic,
                          "marks": sum(p.marks for p in g["question"].parts), "problems left": len(g["problems"]),
                          "closest past part": round(g["closest_past"], 2)} for g in generated])
