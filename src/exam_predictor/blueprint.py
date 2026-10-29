"""Step 10 · Predict the next paper and test the method on past exams.

How the prediction works (simple rules you can explain):
1. Shape: copy the shape of the most typical real paper, i.e. the past paper with the usual number of questions whose
   marks per topic are closest to the average paper. The new paper gets the same questions, topics, marks and
   part structure, e.g. (a)[2] Define, (b)(i)[3] Calculate, so the AI only has to write the content.
2. Outcomes: for each question, choose 2-4 learning outcomes of its main topic, with the best score:
   score = share of sittings that tested the outcome + recency x (sittings since it was last tested, up to 3) / 3.
   An outcome is used only once in the paper, and each question stays on one topic.

Backtest: pretend each of the last 4 sittings has not happened yet, predict it from the earlier sittings only,
and check what share of the predicted outcomes really appeared on each paper, compared with random guessing.
"""
from itertools import chain

import numpy as np
import pandas as pd

from .analysis import lo_table
from .text_utils import sitting_label


def typical_paper(data):
    """The past paper (with the usual number of questions) whose marks per topic are closest to the average paper."""
    sizes = data.groupby("paper_id").question_number.nunique()
    usual = sizes.index[sizes == sizes.mode().iat[0]]
    topic_marks = data.pivot_table(index="paper_id", columns="topic", values="marks", aggfunc="sum", fill_value=0)
    return ((topic_marks.loc[usual] - topic_marks.mean()) ** 2).sum(axis=1).idxmin()


def make_blueprint(data, info, recency):
    """Copy the shape of the typical paper, then choose the learning outcomes each question tests."""
    lt = lo_table(data, info)
    lt["score"] = lt["sittings_%"] / 100 + recency * lt.gap.clip(upper=3) / 3
    shape = data[data.paper_id == typical_paper(data)]
    rows, used = [], set()
    for number, q in shape.groupby("question_number"):
        topic = q.groupby("topic").marks.sum().idxmax()                     # the question's main topic
        options = lt[lt.topic == topic]
        fresh = options[~options.index.isin(used)]                          # each outcome once per paper, if possible
        marks = int(q.marks.sum())
        chosen = (fresh if len(fresh) else options).nlargest(int(np.clip(round(marks / 3.5), 2, 4)), "score")
        used |= set(chosen.index)
        skeleton = [(r.part[len(str(number)):] or "(a)", int(r.marks), r.command) for r in q.itertuples()]
        rows.append({"question": number, "topic": topic, "marks": marks,
                     "outcomes": chosen.index.tolist(), "outcome_text": chosen.outcome.tolist(),
                     "skeleton": skeleton, "parts": " ".join(f"{l}[{m}]" for l, m, _ in skeleton)})
    return pd.DataFrame(rows)


def backtest(data, info, weights=(0.0, 0.3, 0.6), n_last=4):
    """% of predicted learning outcomes that really appeared on a paper of each of the last sittings."""
    rows = []
    for s in sorted(data.sitting_idx.unique())[-n_last:]:
        train, test = data[data.sitting_idx < s], data[data.sitting_idx == s]
        if train.sitting_idx.nunique() < 3:
            continue
        seen = set(train.lo)
        real = [set(p.lo) for _, p in test.groupby("paper_id")]   # outcomes on each real paper of that sitting
        row = {"sitting": sitting_label(s), "random guess": np.mean([len(r & seen) / len(seen) for r in real])}
        for w in weights:
            predicted = set(chain.from_iterable(make_blueprint(train, info, w).outcomes))
            row[f"recency {w}"] = np.mean([len(r & predicted) / len(predicted) for r in real])
        rows.append(row)
    return (pd.DataFrame(rows).set_index("sitting") * 100).round(1)


def best_recency(bt, default=0.3):
    """The recency weight with the best average backtest score."""
    return float(bt.drop(columns="random guess").mean().idxmax().split()[-1]) if len(bt) else default
