"""Step 8 (part 3) · Search the archive: the retrieval half of RAG.

- search(query) mixes meaning (embeddings) with exact words (BM25), combined by rank. This is hybrid search.
  Exact words such as "Hall voltage" help the BM25 half; descriptions help the embedding half.
- similar(question_id) finds the most alike parts from other sittings.
"""
import re

import numpy as np

QUERY_PROMPT = "Instruct: Given a question about physics exams, retrieve relevant exam question parts\nQuery: "


class SearchIndex:
    def __init__(self, df, part_vecs, embedder):
        from rank_bm25 import BM25Okapi

        self.df, self.part_vecs, self.embed = df, part_vecs, embedder
        self.bm25 = BM25Okapi([re.findall(r"\w+", t.lower()) for t in df.search_text])

    def search(self, query, years=(2022, 2100), k=6):
        """Hybrid search: meaning (embeddings) + exact words (BM25), combined by rank."""
        df = self.df
        ok = df.year.between(*years).values
        q = self.embed([query], prompt=QUERY_PROMPT)[0]
        score = np.zeros(len(df))
        for s in (self.part_vecs @ q, self.bm25.get_scores(re.findall(r"\w+", query.lower()))):
            top = np.argsort(-np.where(ok, s, -np.inf))[:100]
            score[top] += 1 / (60 + np.arange(1, len(top) + 1))   # a high rank in either list adds points
        best = [i for i in np.argsort(-score) if ok[i]][:k]
        return df.iloc[best]

    def similar(self, question_id, k=8):
        """The most similar parts from other sittings."""
        df = self.df
        i = int(np.flatnonzero(df.question_id.values == question_id)[0])
        s = self.part_vecs @ self.part_vecs[i]
        best = [j for j in np.argsort(-s) if df.sitting_idx.iat[j] != df.sitting_idx.iat[i]][:k]
        return df.iloc[best].assign(similarity=s[best].round(3))[["question_id", "sitting", "lo", "command", "marks",
                                                                   "similarity", "text"]]
