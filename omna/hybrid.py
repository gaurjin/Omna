"""omna.hybrid — lexical retrieval (BM25) and rank fusion (RRF) for hybrid search.

Semantic (embedding) search is great at meaning but blurs rare exact tokens —
part codes, IDs, surnames, acronyms. BM25 nails those exact-term matches.
Hybrid search runs both and fuses their *rankings* with Reciprocal Rank Fusion,
so a row the embeddings ranked poorly but that lexically matches the query still
surfaces.

Everything here is pure Python + numpy — no new dependencies, no network, and
no relation to the proprietary cosine kernel. BM25 and RRF are standard,
published information-retrieval algorithms.
"""
from __future__ import annotations

import re

import numpy as np

# Split on any run of alphanumerics; lowercase. A code like "XJ-9000" tokenizes
# to ["xj", "9000"] in BOTH corpus and query, so exact-code queries still match.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# RRF's smoothing constant. 60 is the value from the original Cormack et al.
# (2009) paper and the de-facto industry default; it damps the influence of a
# single ranker's top positions so no one ranker dominates the fusion.
_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase *text* and split into alphanumeric tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """BM25 (Okapi) lexical scorer over a fixed corpus.

    Builds an inverted index once at construction; ``scores(query)`` is then
    O(number of postings for the query terms), not O(corpus). Standard
    parameters ``k1`` (term-frequency saturation) and ``b`` (length
    normalisation) use their conventional defaults.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)

        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)
        # term -> list of (doc_index, term_frequency)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        df: dict[str, int] = {}  # term -> document frequency

        for doc_idx, doc in enumerate(corpus):
            tokens = tokenize(doc if isinstance(doc, str) else "")
            self.doc_len[doc_idx] = len(tokens)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                self._postings.setdefault(tok, []).append((doc_idx, freq))
                df[tok] = df.get(tok, 0) + 1

        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # idf with the +0.5 smoothing and a +1 inside the log so it never goes
        # negative (a term in every doc gets idf≈0, not below).
        self._idf: dict[str, float] = {
            term: float(np.log(1.0 + (self.n_docs - dfi + 0.5) / (dfi + 0.5)))
            for term, dfi in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score per document for *query* (shape ``(n_docs,)``).

        Documents containing none of the query terms score exactly 0.0.
        """
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0 or self.avgdl == 0.0:
            return scores
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, freq in postings:
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
                )
                scores[doc_idx] += idf * (freq * (self.k1 + 1.0)) / denom
        return scores


# Process-lifetime cache of built BM25 indexes, keyed by "{index_path}::{column}".
# Each entry stores (df_identity, BM25). The df identity is id() of the loaded
# DataFrame: index.load() returns a fresh object whenever the index was
# re-saved, so a re-embed (even with the SAME row count but new content)
# produces a new identity and forces a rebuild. Keying by path keeps the cache
# bounded to one entry per index file.
_bm25_cache: dict[str, tuple[int, BM25]] = {}


def get_bm25(cache_key: str, df_id: int, corpus: list[str]) -> BM25:
    """Return a cached BM25 for *cache_key*, rebuilding from *corpus* on a miss
    or whenever *df_id* (the loaded DataFrame's identity) has changed — which is
    exactly when the indexed content may have changed."""
    entry = _bm25_cache.get(cache_key)
    if entry is None or entry[0] != df_id:
        bm = BM25(corpus)
        _bm25_cache[cache_key] = (df_id, bm)
        return bm
    return entry[1]


def rrf_fuse(rankings: list[list[int]], k: int = _RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion of several rankings.

    Each entry in *rankings* is a list of document indices ordered best-first.
    A document's fused score is ``sum(1 / (k + rank))`` over the rankings it
    appears in (rank is 0-based). A document present in multiple rankings, or
    ranked highly in any one, scores higher. Documents absent from a ranking
    simply contribute nothing for it.

    Returns ``{doc_index: fused_score}``.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return fused
