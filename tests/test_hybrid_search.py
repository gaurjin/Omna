"""End-to-end tests for hybrid search (BM25 + semantic, fused with RRF).

The win these prove: a rare *exact* token (a part code) that the embedding model
blurs is surfaced — and ranked first — by the lexical half of hybrid search,
while pure-semantic search may bury it. Hybrid is the default; hybrid=False is
the pure-semantic escape hatch (and preserves the old cosine-descending order).
"""
import pytest
import polars as pl

import omna  # noqa: F401


CORPUS = [
    "premium wireless headphones with active noise cancelling",
    "ergonomic mesh office chair with lumbar support",
    "insulated stainless steel water bottle keeps drinks cold",
    "replacement turbo module XJ9000 for industrial encabulators",
    "organic loose leaf green tea sampler",
    "waterproof bluetooth portable speaker with deep bass",
    "slim minimalist leather wallet with rfid blocking",
    "adjustable standing desk with electric height memory",
]
CODE_ROW = "replacement turbo module XJ9000 for industrial encabulators"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hyb")
    idx = tmp / "text.parquet"
    df = pl.DataFrame({"text": CORPUS, "sku": list(range(len(CORPUS)))})
    df.omna.embed("text", index_path=idx)
    return df, idx


def test_hybrid_ranks_exact_code_match_first(store):
    """The only row containing the exact code 'XJ9000' must rank #1 under hybrid.

    This is deterministic regardless of the embedding model: the code row is the
    sole lexical hit, so it receives the rank-0 lexical RRF boost that no other
    row gets, on top of a positive semantic contribution — its fused score
    therefore exceeds every semantic-only row's."""
    df, idx = store
    res = df.omna.search("XJ9000", on="text", k=8, index_path=idx)
    assert res["text"].to_list()[0] == CODE_ROW


def test_hybrid_surfaces_exact_match_at_small_k(store):
    df, idx = store
    res = df.omna.search("XJ9000", on="text", k=1, index_path=idx)
    assert res["text"].to_list() == [CODE_ROW]


def test_hybrid_promotes_exact_match_above_pure_semantic(store):
    """Hybrid ranks the exact lexical row at least as high as pure-semantic does
    (and here, strictly higher to position 0)."""
    df, idx = store
    n = len(CORPUS)
    hyb = df.omna.search("XJ9000", on="text", k=n, index_path=idx)["text"].to_list()
    sem = df.omna.search("XJ9000", on="text", k=n, index_path=idx, hybrid=False)["text"].to_list()
    assert hyb.index(CODE_ROW) <= sem.index(CODE_ROW)


def test_hybrid_off_preserves_cosine_descending(store):
    df, idx = store
    res = df.omna.search("water bottle", on="text", k=5, index_path=idx, hybrid=False)
    scores = res["_score"].to_list()
    assert scores == sorted(scores, reverse=True)


def test_hybrid_keeps_score_column_and_k(store):
    df, idx = store
    res = df.omna.search("speaker", on="text", k=3, index_path=idx)
    assert "_score" in res.columns
    assert len(res) == 3


def test_search_negative_k_returns_empty(store):
    df, idx = store
    res = df.omna.search("speaker", on="text", k=-1, index_path=idx)
    assert len(res) == 0


def test_search_rebuilt_index_is_not_stale(tmp_path):
    """Re-embedding a column with new content of the SAME row count must search
    the NEW content lexically, not a cached BM25 of the old content."""
    idx = tmp_path / "t.parquet"
    pl.DataFrame({"text": ["apple pie recipe", "banana bread"]}).omna.embed("text", index_path=idx)
    df1 = pl.DataFrame({"text": ["apple pie recipe", "banana bread"]})
    df1.omna.search("apple", on="text", k=1, index_path=idx)  # warms the BM25 cache
    # Overwrite the index with different content, same row count.
    df2 = pl.DataFrame({"text": ["rocket fuel mix", "turbine blade"]})
    df2.omna.embed("text", index_path=idx)
    res = df2.omna.search("rocket", on="text", k=1, index_path=idx)
    assert res["text"].to_list() == ["rocket fuel mix"]


def test_hybrid_no_lexical_overlap_falls_back_to_semantic(store):
    """A query with no token in the corpus (BM25 all-zero) must behave exactly
    like pure-semantic search — hybrid never hurts the meaning-only case."""
    df, idx = store
    hyb = df.omna.search("comfortable seating", on="text", k=4, index_path=idx)["text"].to_list()
    sem = df.omna.search("comfortable seating", on="text", k=4, index_path=idx, hybrid=False)["text"].to_list()
    assert hyb == sem
