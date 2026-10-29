"""Step 13 · Ask the archive (RAG question answering).

ask_archive("...") finds the most relevant past parts with the search index, then lets the AI model answer ONLY from
them, listing which question IDs it used. Use show_question(df, id) to check the originals.
"""
from pydantic import BaseModel

from .display import show, show_markdown


class ArchiveAnswer(BaseModel):
    answer: str
    sources: list[str]


def ask_archive(question, index, llm, years=(2022, 2100), k=6):
    """Answer a question about past papers from the k best matching parts. Returns (answer, hits) or None."""
    hits = index.search(question, years, k)
    evidence = "\n\n".join(f"[{r.question_id}] ({r.sitting}, {r.marks} marks)\n{r.search_text}" for r in hits.itertuples())
    prompt = (f"Answer the question using ONLY the past exam parts below. Explain the method and how the marks are "
              f"awarded. In sources, list the IDs of the parts you used.\n\nQuestion: {question}\n\nPast exam parts:\n{evidence}"
              '\n\nReply format: {"answer": "...", "sources": ["9702_s24_41_Q1(a)", "..."]}')
    try:
        out = llm.ask(prompt, ArchiveAnswer)
    except ValueError as e:
        print("The model's reply could not be used:", e)
        return None
    show_markdown(out.answer)
    used = {s.strip("[] ") for s in out.sources}
    hits = hits.assign(used=hits.question_id.isin(used))
    show(hits[["question_id", "sitting", "marks", "lo", "used", "text"]], 80)
    return out, hits
