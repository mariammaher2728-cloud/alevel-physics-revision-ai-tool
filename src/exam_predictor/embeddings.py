"""Step 8 (part 1) · Embeddings: a list of 1024 numbers describing the meaning of a text.

Similar meanings give similar numbers. Every vector has length 1, so the dot product of two vectors
is their cosine similarity (1 = same meaning, around 0 = unrelated).
"""
import hashlib
from pathlib import Path

import numpy as np


class Embedder:
    """The small Qwen3 embedding model. Call it like a function: embedder(texts, prompt=None)."""

    def __init__(self, model_name, device=None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")   # a T4 GPU is about 10x faster
        if self.device == "cpu":
            print("No GPU found - choose Runtime > Change runtime type > T4 GPU, or embedding will be slow.")
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()                              # half precision: half the memory, same results
        self.model.max_seq_length = 512                    # longer texts are cut (our parts are short)

    def __call__(self, texts, prompt=None):
        """Meaning vectors of length 1, one row per text."""
        return np.asarray(self.model.encode(list(texts), prompt=prompt, batch_size=32, normalize_embeddings=True),
                          dtype=np.float32)


def part_vectors(df, embedder, work_dir):
    """Embeddings of all parts (context + question + mark scheme), saved in work_dir/part_vectors.npz.
    They are recomputed only when the extracted text changes."""
    vec_file = Path(work_dir) / "part_vectors.npz"
    text_hash = hashlib.sha256("\n".join(df.search_text).encode()).hexdigest()
    if vec_file.exists():
        with np.load(vec_file) as saved:
            if str(saved["hash"]) == text_hash:
                return saved["vecs"]
    vecs = embedder(df.search_text)
    np.savez(vec_file, vecs=vecs, hash=text_hash)
    return vecs
