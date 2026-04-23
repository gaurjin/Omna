"""Day 3 tests — index.py."""
import pytest
import polars as pl
from omna.index import save, load, EMBEDDING_COL


@pytest.fixture
def tmp_index(tmp_path):
    return tmp_path / "test.parquet"


def test_round_trip_preserves_rows(tmp_index):
    df = pl.DataFrame({"text": ["hello", "world"]})
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    save(df, embeddings, tmp_index)
    df2, embs2 = load(tmp_index)
    assert df2.shape == df.shape
    assert len(embs2) == 2


def test_round_trip_preserves_text_column(tmp_index):
    df = pl.DataFrame({"text": ["alpha", "beta"]})
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    save(df, embeddings, tmp_index)
    df2, _ = load(tmp_index)
    assert df2["text"].to_list() == ["alpha", "beta"]


def test_round_trip_preserves_vectors(tmp_index):
    df = pl.DataFrame({"text": ["x"]})
    vec = [0.1, 0.2, 0.3, 0.4]
    save(df, [vec], tmp_index)
    _, embs = load(tmp_index)
    assert [round(v, 5) for v in embs[0]] == [round(v, 5) for v in vec]


def test_embedding_column_stripped_from_returned_df(tmp_index):
    df = pl.DataFrame({"text": ["x"]})
    save(df, [[1.0, 0.0]], tmp_index)
    df2, _ = load(tmp_index)
    assert EMBEDDING_COL not in df2.columns


def test_mismatched_lengths_raises(tmp_index):
    df = pl.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError, match="2 rows"):
        save(df, [[1.0]], tmp_index)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load("/tmp/omna_nonexistent_file.parquet")


def test_multi_column_dataframe_preserved(tmp_index):
    df = pl.DataFrame({"id": [1, 2], "text": ["foo", "bar"], "score": [0.9, 0.5]})
    save(df, [[0.1, 0.2], [0.3, 0.4]], tmp_index)
    df2, _ = load(tmp_index)
    assert df2.columns == ["id", "text", "score"]


def test_parent_dirs_created_automatically(tmp_path):
    deep_path = tmp_path / "a" / "b" / "c" / "index.parquet"
    df = pl.DataFrame({"text": ["x"]})
    save(df, [[1.0]], deep_path)
    assert deep_path.exists()
