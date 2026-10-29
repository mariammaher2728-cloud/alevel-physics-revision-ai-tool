"""Step 8 (part 2) · Tag every part with the syllabus learning outcome it tests.

1. The similarity of every part to every outcome is one matrix multiplication: part_vecs @ lo_vecs.T.
2. Each part gets its most similar outcome. Papers from 2022 onward can only get current (non-legacy) outcomes.
3. Optional: when the two best outcomes score almost the same, the AI model picks the best of the top 3.
   It reads 10 parts at a time and replies with one letter (A, B or C) per part; a batch whose reply does not
   have the right number of letters keeps its embedding tags.
Check: look at a random sample of tags. Count how many you would change; that is the tag accuracy you can report.
"""
import re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .syllabus import outcome_info

LO_PROMPT = "Instruct: Find physics exam questions that assess this syllabus learning outcome\nQuery: "


def tag_parts(df, los, part_vecs, embedder, llm=None, work_dir=None):
    """Returns df with three new columns: lo (the outcome code), topic and lo_text.
    With an `llm`, the uncertain tags of current papers are double-checked by the AI model."""
    lo_vecs = embedder(los.outcome, prompt=LO_PROMPT)
    sims = part_vecs @ lo_vecs.T
    sims[np.ix_(df.current.values, los.code.str.startswith("L").values)] = -1    # no legacy tags for current papers
    top3 = np.argsort(-sims, axis=1)[:, :3]
    tags = los.code.values[top3[:, 0]].copy()
    if llm is not None:
        tags = check_uncertain_tags(df, los, sims, top3, tags, llm)

    info = outcome_info(los)
    df = df.assign(lo=tags, topic=info.topic.reindex(tags).values, lo_text=info.outcome.reindex(tags).values)
    if work_dir:
        df.drop(columns=["search_text"]).to_parquet(Path(work_dir) / "questions_tagged.parquet")
    return df


def check_uncertain_tags(df, los, sims, top3, tags, llm, margin=0.04, batch_size=10):
    """When the top two outcomes are close, the AI model picks the best of the top 3 (10 parts per request)."""
    rows = np.arange(len(df))
    unsure = sims[rows, top3[:, 0]] - sims[rows, top3[:, 1]] < margin
    todo = np.flatnonzero(df.current.values & unsure)
    print(f"{len(todo)} of {int(df.current.sum())} current parts have an uncertain tag; the AI checks those.")
    changed, kept = 0, 0
    for start in tqdm(range(0, len(todo), batch_size), desc="AI checks syllabus tags"):
        batch = todo[start:start + batch_size]
        prompt = ("For each Cambridge A Level Physics exam question part below, choose the syllabus learning outcome "
                  f"(A, B or C) that it mainly assesses.\nReply with ONLY {len(batch)} letters separated by spaces, "
                  "one per part in order, for example: A C B\n")
        for n, i in enumerate(batch, 1):
            options = "\n".join(f"{'ABC'[j]}) {los.outcome.iat[top3[i, j]]}" for j in range(3))
            prompt += f"\nPart {n}:\n{df.search_text.iat[i]}\n{options}\n"
        reply = llm.ask(prompt, max_new_tokens=60)
        picks = re.findall(r"\b[ABC]\b", reply)              # works for "A C B", "1. A 2. C" or ["A", "C"]
        if len(picks) == len(batch):
            new = [los.code.iat[top3[i, "ABC".index(c)]] for i, c in zip(batch, picks)]
            changed += int((tags[batch] != new).sum())
            tags[batch] = new
        else:
            kept += len(batch)
            print(f"A batch kept its embedding tags: expected {len(batch)} letters, the reply was {reply[:80]!r}")
    print(f"The AI changed {changed} tags; {kept} parts kept their embedding tag.")
    return tags
