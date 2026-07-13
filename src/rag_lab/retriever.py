"""A dependency-free TF-IDF retriever with cosine similarity.

This is deliberately simple and transparent so the lab runs anywhere. It is the
baseline you compare a local embedding model against — see the README for how to
swap in a local vector backend.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "that", "this", "be", "as", "at", "by", "it", "from", "how", "do",
    "i", "you", "we", "what", "should", "can", "my", "about",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    score: float
    text: str


class TfidfRetriever:
    def __init__(self) -> None:
        self._chunks: list = []
        self._vectors: List[Dict[str, float]] = []
        self._norms: List[float] = []
        self._idf: Dict[str, float] = {}

    def fit(self, chunks: Iterable) -> "TfidfRetriever":
        self._chunks = sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.chunk_id))
        chunk_ids = [chunk.chunk_id for chunk in self._chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_id values must be unique for deterministic retrieval")
        tokenized = [tokenize(c.text) for c in self._chunks]
        n = len(tokenized)

        df: Counter = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1
        self._idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

        self._vectors = []
        self._norms = []
        for tokens in tokenized:
            vec = self._vectorize(tokens)
            self._vectors.append(vec)
            self._norms.append(math.sqrt(sum(w * w for w in vec.values())) or 1.0)
        return self

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        length = len(tokens)
        vec: Dict[str, float] = {}
        for term, count in counts.items():
            idf = self._idf.get(term)
            if idf is not None:
                vec[term] = (count / length) * idf
        return vec

    def query(self, text: str, top_k: int = 3) -> List[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        qvec = self._vectorize(tokenize(text))
        qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0

        results: List[RetrievalResult] = []
        for chunk, vec, norm in zip(self._chunks, self._vectors, self._norms):
            # Dot product over the smaller of the two sparse vectors.
            small, large = (qvec, vec) if len(qvec) <= len(vec) else (vec, qvec)
            dot = 0.0
            for term, weight in small.items():
                other = large.get(term)
                if other is not None:
                    dot += weight * other
            if dot == 0.0:
                continue
            results.append(
                RetrievalResult(chunk.chunk_id, chunk.doc_id, dot / (qnorm * norm), chunk.text)
            )

        results.sort(key=lambda result: (-result.score, result.doc_id, result.chunk_id))
        return results[:top_k]
