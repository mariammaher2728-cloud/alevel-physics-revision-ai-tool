"""Step 3 · Find the past papers and keep complete pairs.

Year, session and variant come from the filename: 9702_s24_qp_41.pdf is May/June 2024, question paper, variant 41.
A paper is kept only when both its question paper (qp) and mark scheme (ms) exist. Subfolders are fine.
"""
import re
from pathlib import Path

import pandas as pd

from .config import SESSION_CODE

NAME = re.compile(r"9702_([msw])(\d{2})_(qp|ms)_(4\d)\.pdf", re.I)


def find_papers(data_dir):
    """One row per paper, with the paths of its question paper (qp) and mark scheme (ms)."""
    files = []
    for pdf in sorted(Path(data_dir).rglob("*.pdf")):
        m = NAME.fullmatch(pdf.name)
        if m:
            session, yy, kind, comp = m.groups()
            files.append({"paper_id": f"9702_{session.lower()}{yy}_{comp}", "year": 2000 + int(yy),
                          "session": session.lower(), "component": comp, "kind": kind.lower(), "path": str(pdf)})
    files = pd.DataFrame(files)
    if files.empty:
        raise FileNotFoundError(f"No 9702 Paper 4 PDFs found in {data_dir}")
    papers = (files.drop_duplicates(["paper_id", "kind"])
                   .pivot(index=["paper_id", "year", "session", "component"], columns="kind", values="path")
                   .dropna().reset_index())
    print(f"{len(papers)} complete papers; {files.paper_id.nunique() - len(papers)} papers skipped (qp or ms missing)")
    return papers


def papers_per_session(papers):
    """How many papers there are per year and session (FM, MJ, ON)."""
    return papers.groupby(["year", "session"]).size().unstack(fill_value=0).rename(columns=SESSION_CODE)
