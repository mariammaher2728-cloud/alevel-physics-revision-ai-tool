"""Step 9 · Analysis: important topics, repeated tasks and trends.

Everything here is plain counting with pandas, on papers from the syllabus start year that pass the 100-mark check.

| Output                               | Question it answers                                                     |
|--------------------------------------|-------------------------------------------------------------------------|
| Marks per topic (bar chart, heatmap) | Which topics carry the most marks, and is that changing?                |
| Most tested learning outcomes        | Which tasks come back, how often, and when were they last seen?         |
| Repeated tasks (outcome + command)   | e.g. "Define gravitational potential" in 9 of 14 sittings               |
| Command words by year                | Is the paper moving towards more Explain or more Calculate?             |
| Paper template                       | How many questions, how many marks each, which topic usually comes where |
| Definitions bank                     | Every definition asked, with the mark-scheme wording                    |
| Same wording                         | Parts repeated word-for-word in different sittings                      |

There are only about 14 sittings since 2022, so a difference of one or two appearances is just noise.
"Sittings %" counts each exam session once; "papers %" is the chance a task appears on any single paper.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .display import finish_figure, show
from .text_utils import sitting_label, squash

BLUE = "#2a78d6"


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)


def question_table(data):
    """One row per question: its main topic (the topic with most marks), total marks and sitting."""
    t = data.groupby(["paper_id", "question_number", "topic"]).marks.sum().reset_index()
    main = t.sort_values("marks").groupby(["paper_id", "question_number"]).tail(1).drop(columns="marks")
    totals = data.groupby(["paper_id", "question_number"]).agg(marks=("marks", "sum"), sitting_idx=("sitting_idx", "first"))
    return main.merge(totals.reset_index(), on=["paper_id", "question_number"])


def lo_table(data, info):
    """How often each learning outcome is tested, and how long ago it was last tested."""
    g = data.groupby("lo")
    t = pd.DataFrame({"sittings_%": g.sitting_idx.nunique() / data.sitting_idx.nunique() * 100,
                      "papers_%": g.paper_id.nunique() / data.paper_id.nunique() * 100,
                      "marks_when_tested": g.marks.sum() / g.paper_id.nunique(),
                      "last_seen": g.sitting_idx.max(),
                      "usual_command": g.command.agg(lambda c: c.mode().iat[0])})
    t["gap"] = data.sitting_idx.max() - t.last_seen          # sittings since it was last tested
    return t.join(info[["topic", "outcome"]]).sort_values("sittings_%", ascending=False)


def topic_charts(pool, topic_order, reports_dir):
    """Average marks per paper by topic (bar chart) and by year (heatmap)."""
    paper_marks = pool.pivot_table(index=["paper_id", "year"], columns="topic", values="marks", aggfunc="sum", fill_value=0)
    topics = pd.DataFrame({"avg_marks_per_paper": paper_marks.mean(),
                           "papers_with_topic_%": (paper_marks > 0).mean() * 100}).round(1).sort_values("avg_marks_per_paper")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(topics.index, topics.avg_marks_per_paper, color=BLUE, height=0.6)
    for y, v in enumerate(topics.avg_marks_per_paper):
        ax.text(v + 0.2, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set(title="Average marks per paper by topic", xlabel="marks (out of 100)")
    tidy(ax)
    finish_figure(fig, reports_dir / "topic_marks.png")

    by_year = paper_marks.groupby(level="year").mean().T.reindex([t for t in topic_order if t in paper_marks.columns])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(by_year.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(by_year.columns)), by_year.columns)
    ax.set_yticks(range(len(by_year.index)), by_year.index)
    for (y, x), v in np.ndenumerate(by_year.values):
        ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > by_year.values.max() * 0.6 else "black")
    ax.set_title("Average marks per paper: topic by year")
    finish_figure(fig, reports_dir / "topic_by_year.png")
    return topics


def command_chart(pool, reports_dir):
    """Share of marks by command word, year by year."""
    cmd = pool.pivot_table(index="year", columns="command", values="marks", aggfunc="sum", fill_value=0)
    cmd = cmd.div(cmd.sum(axis=1), axis=0) * 100
    main_cmds = cmd.mean().nlargest(6).index
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c, colour in zip(main_cmds, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]):
        ax.plot(cmd.index, cmd[c], marker="o", lw=2, color=colour, label=c)
    ax.set(title="Share of marks by command word", ylabel="% of marks")
    ax.set_xticks(cmd.index)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    tidy(ax)
    finish_figure(fig, reports_dir / "command_words.png")
    return cmd


def run_analysis(pool, info, topic_order, reports_dir, syllabus_start=2022):
    """Run every analysis, show the results and save all tables and charts in reports_dir. Returns the tables."""
    n_papers, n_sittings = pool.paper_id.nunique(), pool.sitting_idx.nunique()
    print(f"Analysing {n_papers} papers from {n_sittings} sittings "
          f"({pool.sitting.iloc[pool.sitting_idx.argmin()]} to {pool.sitting.iloc[pool.sitting_idx.argmax()]})")

    # 1. Important topics
    topics = topic_charts(pool, topic_order, reports_dir)

    # 2. Most tested learning outcomes
    los_stats = lo_table(pool, info)
    outcomes = los_stats.assign(last_seen=los_stats.last_seen.map(sitting_label)).round(1)
    print("Most tested learning outcomes (current syllabus):")
    show(outcomes.head(25)[["topic", "outcome", "sittings_%", "papers_%", "marks_when_tested", "last_seen", "usual_command"]], 80)
    untested = sorted(set(info.index[~info.index.str.startswith("L")]) - set(los_stats.index))
    print("Never tested since", syllabus_start, ":", ", ".join(untested) or "none")

    # 3. Repeated tasks = learning outcome + command word
    tasks = (pool.groupby(["lo", "command"]).agg(sittings=("sitting_idx", "nunique"), papers=("paper_id", "nunique"),
                                                  marks=("marks", "sum"), example=("question_id", "last"))
                  .reset_index().sort_values(["sittings", "papers"], ascending=False))
    tasks["task"] = tasks.command + ": " + tasks.lo.map(info.outcome).str.slice(0, 70)
    print(f"\nMost repeated tasks (out of {n_sittings} sittings):")
    show(tasks.head(20)[["task", "sittings", "papers", "example"]], 90)

    # 4. Command words over time (share of marks)
    command_chart(pool, reports_dir)

    # 5. Paper template
    qs = question_table(pool)
    template = qs.groupby("question_number").agg(papers=("paper_id", "nunique"), avg_marks=("marks", "mean"),
                                                 usual_topic=("topic", lambda t: t.mode().iat[0]),
                                                 how_often=("topic", lambda t: f"{(t == t.mode().iat[0]).mean():.0%}"))
    print(f"\nA paper usually has {qs.groupby('paper_id').size().median():.0f} questions. Typical layout:")
    show(template.round(1))

    # 6. Definitions bank and identical wording
    defs = (pool[pool.command == "Define"].sort_values("sitting_idx")
                .groupby("lo").agg(sittings=("sitting_idx", "nunique"), wording=("text", "last"), mark_scheme=("answer", "last"))
                .join(info.outcome).sort_values("sittings", ascending=False))
    defs["wording"] = defs.wording.map(lambda t: squash(re.sub(r"\[(Total:)?\s*\d+\]", "", t)))
    defs["mark_scheme"] = defs.mark_scheme.map(squash)
    print("\nDefinitions bank (most asked first):")
    show(defs.head(15)[["sittings", "wording", "mark_scheme"]], 70)

    key = pool.text.str.replace(r"\[(Total:)?\s*\d+\s*\]|^\s*(\(\w+\)\s*)+", "", regex=True).map(squash).str.lower()
    same = (pool.assign(key=key)[key.str.len() > 25].groupby("key")
                .agg(sittings=("sitting_idx", "nunique"), ids=("question_id", list)).query("sittings > 1")
                .sort_values("sittings", ascending=False))
    print(f"\n{len(same)} parts were repeated word-for-word in different sittings:")
    show(same.head(10), 80)

    tables = {"topics": topics, "learning_outcomes": outcomes, "repeated_tasks": tasks, "template": template,
              "definitions_bank": defs, "same_wording": same}
    for name, table in tables.items():
        table.to_csv(reports_dir / f"{name}.csv")
    return tables
