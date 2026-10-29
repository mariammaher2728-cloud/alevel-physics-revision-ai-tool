"""Steps 5-6 · Read every paper into the question table `df`, and look at any question.

`df` has one row per scored part, e.g. 9702_s24_41_Q3(b)(ii), with its text, marks, context, mark-scheme answer,
sitting (e.g. MJ 2024, so variants 41/42/43 are not counted as repeats) and command word (Define, Calculate, ...).
A normal Paper 4 totals 100 marks; papers that fail this check are left out of the statistics but can still be searched.
"""
import pandas as pd
import pymupdf as fitz
from tqdm.auto import tqdm

from .config import SESSION_ORDER
from .display import show, show_markdown
from .extract import extract_pair
from .text_utils import command_word, part_text, sitting_label


def build_question_table(papers, syllabus_start, reports_dir):
    """Read all papers. Returns (df, quality): the question table and one row of checks per paper."""
    rows, checks = [], []
    for p in tqdm(papers.itertuples(), total=len(papers), desc="Reading papers"):
        try:
            records, check = extract_pair(p.qp, p.ms)
        except Exception as e:                                       # one broken PDF should not stop the run
            records, check = [], {"extraction_ok": False, "error": str(e)}
        rows += [{**r, "paper_id": p.paper_id, "year": p.year, "session": p.session, "component": p.component,
                  "qp": p.qp, "ms": p.ms, "extraction_ok": check["extraction_ok"]} for r in records]
        checks.append({"paper_id": p.paper_id, **check})

    df = pd.DataFrame(rows)
    quality = pd.DataFrame(checks)
    df["question_id"] = df.paper_id + "_Q" + df.part
    df["sitting_idx"] = df.year * 3 + df.session.map(SESSION_ORDER)
    df["sitting"] = df.sitting_idx.map(sitting_label)
    df["current"] = df.year >= syllabus_start
    df["command"] = [command_word(t, c) for t, c in zip(df.text, df.context)]
    df["search_text"] = [part_text(c, t, a) for c, t, a in zip(df.context, df.text, df.answer)]   # used in Step 8
    quality.to_csv(reports_dir / "extraction_quality.csv", index=False)
    print(f"{len(df):,} scored parts from {df.paper_id.nunique()} papers; "
          f"{quality.extraction_ok.sum()}/{len(quality)} papers pass the 100-mark check")
    return df, quality


def analysis_pool(df):
    """The parts used for statistics: current syllabus, from papers that pass the 100-mark check."""
    pool = df[df.current & df.extraction_ok].copy()
    if pool.empty:
        raise ValueError("No checked papers from the syllabus start year onward - look at extraction_quality.csv.")
    return pool


def show_question(df, question_id, images=True):
    """Print the extracted text and mark-scheme answer of one part, then show the real PDF pages.
    Example: show_question(df, "9702_s24_41_Q1(a)")"""
    r = df[df.question_id == question_id].iloc[0]
    show_markdown(f"**{question_id}** · {r.sitting} · {r.marks} marks · command: {r.command} · tag: {r.get('lo', '-')}")
    print("CONTEXT\n", r.context, "\n\nQUESTION\n", r.text, f"\n\nMARK SCHEME ({r.answer_match})\n", r.answer or "(not found)")
    if images:
        for path, pages in [(r.qp, r.context_pages), (r.ms, r.answer_pages)]:
            with fitz.open(path) as doc:
                for p in pages:
                    show(doc[p - 1].get_pixmap(dpi=80).pil_image())
