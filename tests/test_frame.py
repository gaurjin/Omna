"""Day 4 tests — df.omna.embed(), search(), and filter() end-to-end."""
import pytest
import polars as pl
import omna  # noqa: F401


ANIMALS = ["cat", "kitten", "tabby cat", "dog", "puppy", "golden retriever"]
FOOD = ["pizza", "pasta", "burger", "sushi", "tacos"]
ALL_TEXTS = ANIMALS + FOOD


@pytest.fixture(scope="module")
def indexed_df(tmp_path_factory):
    """Build a DataFrame and write its index once for the whole module."""
    tmp = tmp_path_factory.mktemp("index")
    idx = tmp / "text.parquet"
    df = pl.DataFrame({"text": ALL_TEXTS, "id": list(range(len(ALL_TEXTS)))})
    df.omna.embed("text", index_path=idx)
    return df, idx


def test_embed_creates_index_file(tmp_path):
    idx = tmp_path / "text.parquet"
    df = pl.DataFrame({"text": ["hello", "world"]})
    df.omna.embed("text", index_path=idx)
    assert idx.exists()


def test_embed_returns_original_dataframe(tmp_path):
    idx = tmp_path / "text.parquet"
    df = pl.DataFrame({"text": ["a", "b"], "n": [1, 2]})
    result = df.omna.embed("text", index_path=idx)
    assert result.columns == ["text", "n"]
    assert result.shape == (2, 2)


def test_embed_missing_column_raises(tmp_path):
    df = pl.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError, match="'missing'"):
        df.omna.embed("missing", index_path=tmp_path / "x.parquet")


def test_search_returns_dataframe(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("feline", on="text", k=3, index_path=idx)
    assert isinstance(results, pl.DataFrame)


def test_search_respects_k(indexed_df):
    df, idx = indexed_df
    for k in (1, 3, 5):
        results = df.omna.search("animal", on="text", k=k, index_path=idx)
        assert len(results) == k


def test_search_has_score_column(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("cat", on="text", k=3, index_path=idx)
    assert "_score" in results.columns


def test_search_scores_descending(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("cat", on="text", k=5, index_path=idx)
    scores = results["_score"].to_list()
    assert scores == sorted(scores, reverse=True)


def test_search_feline_returns_cats_before_dogs(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("feline", on="text", k=3, index_path=idx)
    top_texts = results["text"].to_list()
    cat_hits = sum(1 for t in top_texts if "cat" in t or "kitten" in t or "tabby" in t)
    assert cat_hits >= 2


def test_search_food_query_returns_food(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("Italian food", on="text", k=2, index_path=idx)
    top_texts = results["text"].to_list()
    assert any(t in ("pizza", "pasta") for t in top_texts)


def test_search_preserves_all_columns(indexed_df):
    df, idx = indexed_df
    results = df.omna.search("cat", on="text", k=2, index_path=idx)
    assert "id" in results.columns
    assert "text" in results.columns


def test_search_no_index_raises(tmp_path):
    df = pl.DataFrame({"text": ["hello"]})
    with pytest.raises(FileNotFoundError, match="embed"):
        df.omna.search("query", on="text", index_path=tmp_path / "missing.parquet")


def test_filter_returns_dataframe(indexed_df):
    df, idx = indexed_df
    result = df.omna.filter("animal", on="text", index_path=idx)
    assert isinstance(result, pl.DataFrame)


def test_filter_no_score_column(indexed_df):
    df, idx = indexed_df
    result = df.omna.filter("cat", on="text", index_path=idx)
    assert "_score" not in result.columns


def test_filter_high_threshold_excludes_unrelated(indexed_df):
    df, idx = indexed_df
    result = df.omna.filter("cat", on="text", threshold=0.7, index_path=idx)
    texts = result["text"].to_list()
    assert all(t not in ("pizza", "pasta", "burger", "sushi", "tacos") for t in texts)


def test_filter_low_threshold_returns_everything(indexed_df):
    df, idx = indexed_df
    result = df.omna.filter("thing", on="text", threshold=0.0, index_path=idx)
    assert len(result) == len(ALL_TEXTS)


def test_filter_no_index_raises(tmp_path):
    df = pl.DataFrame({"text": ["hello"]})
    with pytest.raises(FileNotFoundError):
        df.omna.filter("concept", on="text", index_path=tmp_path / "missing.parquet")
