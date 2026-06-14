"""Unit tests for omna.hybrid — dependency-free BM25 lexical scoring and
Reciprocal Rank Fusion (RRF). These are the building blocks of hybrid search;
they touch no embeddings, no Rust, and no network."""
import numpy as np

from omna import hybrid


# ── tokenize ────────────────────────────────────────────────────────────────

def test_tokenize_lowercases_and_splits_on_nonalnum():
    assert hybrid.tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_splits_codes_on_punctuation():
    # A part code like "XJ-9000" splits identically in corpus and query, so an
    # exact code query still matches lexically.
    assert hybrid.tokenize("XJ-9000") == ["xj", "9000"]


def test_tokenize_empty_returns_empty():
    assert hybrid.tokenize("") == []
    assert hybrid.tokenize("   ,. !") == []


# ── BM25 ──────────────────────────────────────────────────────────────────--

def test_bm25_term_match_outscores_nonmatch():
    bm = hybrid.BM25(["the quick brown fox", "lazy sleeping dog", "green apple pie"])
    scores = bm.scores("fox")
    assert scores.shape == (3,)
    assert scores[0] > 0.0           # doc 0 contains "fox"
    assert scores[1] == 0.0          # doc 1 has no query term
    assert scores[2] == 0.0


def test_bm25_rare_term_outscores_common_term():
    # "fox" appears in 1 of 4 docs (rare → high idf); "the" in all 4 (idf≈0).
    corpus = [
        "the quick brown fox",
        "the lazy dog",
        "the green apple",
        "the blue sky",
    ]
    bm = hybrid.BM25(corpus)
    fox = bm.scores("fox")[0]
    the = bm.scores("the")[0]
    assert fox > the


def test_bm25_unknown_term_scores_all_zero():
    bm = hybrid.BM25(["alpha beta", "gamma delta"])
    assert list(bm.scores("zzz")) == [0.0, 0.0]


def test_bm25_empty_corpus_is_safe():
    bm = hybrid.BM25([])
    assert list(bm.scores("anything")) == []


# ── RRF ───────────────────────────────────────────────────────────────────--

def test_rrf_monotonic_for_identical_rankings():
    fused = hybrid.rrf_fuse([[0, 1, 2], [0, 1, 2]])
    assert fused[0] > fused[1] > fused[2]


def test_rrf_rewards_presence_in_both_rankings():
    # doc 0 is top of both rankings; doc 1 is only in the first. doc 0 wins.
    fused = hybrid.rrf_fuse([[0, 1], [0]])
    assert fused[0] > fused[1]


def test_rrf_lexical_only_doc_can_outrank_weak_semantic():
    # Semantic ranks doc 9 dead last; lexical ranks it first. With RRF it should
    # beat a doc that is merely mid-pack in semantic and absent from lexical.
    semantic = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # doc 9 worst semantically
    lexical = [9]                                # doc 9 is the exact lexical hit
    fused = hybrid.rrf_fuse([semantic, lexical])
    assert fused[9] > fused[4]


def test_rrf_empty_rankings_returns_empty():
    assert hybrid.rrf_fuse([]) == {}
    assert hybrid.rrf_fuse([[], []]) == {}


# ── get_bm25 cache ───────────────────────────────────────────────────────────

def test_get_bm25_caches_for_same_identity():
    bm1 = hybrid.get_bm25("idx::col", 5, ["alpha beta", "gamma delta"])
    bm2 = hybrid.get_bm25("idx::col", 5, ["alpha beta", "gamma delta"])
    assert bm1 is bm2


def test_get_bm25_rebuilds_when_identity_changes_even_at_same_length():
    # Regression for the stale-cache bug: re-embedding a column with NEW content
    # but the SAME row count must NOT serve the old lexical index. The df
    # identity (id) changes on reload, which forces a rebuild.
    bm_old = hybrid.get_bm25("idx2::col", 100, ["apple pie", "banana split"])
    assert bm_old.scores("apple")[0] > 0.0
    bm_new = hybrid.get_bm25("idx2::col", 200, ["rocket fuel", "turbine blade"])
    assert bm_new is not bm_old
    assert bm_new.scores("rocket")[0] > 0.0      # new content matches
    assert bm_new.scores("apple")[0] == 0.0      # old content is gone
