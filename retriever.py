import time

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    """Embedding model plus a simple exact vector store: a NumPy matrix searched by cosine similarity."""

    def __init__(self, model_id, query_prefix=""):
        self.model = SentenceTransformer(model_id, device="cuda")
        self.query_prefix = query_prefix
        self.chunks = []
        self.vectors = None

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def build_index(self, chunks):
        self.chunks = chunks
        self.vectors = self.embed([c["text"] for c in chunks])

    def chunk_lengths(self):
        """Length of each chunk in the embedding model's own tokens, which it cuts off at max_seq_length."""
        return [len(self.model.tokenizer(c["text"], verbose=False)["input_ids"]) for c in self.chunks]

    def search(self, question, k):
        """Return the top k chunks and the time taken in milliseconds."""
        start = time.perf_counter()
        query = self.embed(self.query_prefix + question)
        scores = self.vectors @ query
        top = np.argsort(scores)[::-1][:k]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return [self.chunks[i] for i in top], elapsed_ms
