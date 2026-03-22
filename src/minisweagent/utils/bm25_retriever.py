"""BM25 retrieval for ranking short text documents (e.g. env knowledge snippets)."""

import math
import re
from collections import Counter


class BM25Retriever:
    """A simple implementation of the BM25 retrieval algorithm."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.n = len(documents)

        if self.n == 0:
            self.avgdl = 0
            self.doc_freqs = []
            self.idf = {}
            return

        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.avgdl = sum(len(d) for d in tokenized_docs) / self.n
        self.doc_freqs = []
        df = Counter()
        for tokens in tokenized_docs:
            self.doc_freqs.append(Counter(tokens))
            for token in set(tokens):
                df[token] += 1

        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r"\w+", text.lower())

    def get_scores(self, query: str) -> list[float]:
        if self.n == 0:
            return []
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.n
        for i in range(self.n):
            doc_len = sum(self.doc_freqs[i].values())
            if doc_len == 0:
                continue
            for token in query_tokens:
                if token in self.idf:
                    tf = self.doc_freqs[i].get(token, 0)
                    numerator = self.idf[token] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += numerator / denominator
        return scores

    def get_top_k(self, query: str, k: int) -> list[int]:
        if self.n == 0:
            return []
        scores = self.get_scores(query)
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in indexed_scores[:k]]
