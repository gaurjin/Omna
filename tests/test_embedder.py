"""Day 3 tests — embedder.py."""
import math
import pytest
from omna.embedder import embed, embedding_dim, DEFAULT_MODEL
from _omna import cosine_similarity


def test_returns_one_vector_per_text():
    texts = ["hello", "world", "foo"]
    vecs = embed(texts)
    assert len(vecs) == 3


def test_vector_length_matches_model_dim():
    vecs = embed(["test"])
    assert len(vecs[0]) == embedding_dim()


def test_all_elements_are_floats():
    vecs = embed(["check types"])
    assert all(isinstance(x, float) for x in vecs[0])


def test_identical_texts_produce_identical_vectors():
    a, b = embed(["same text", "same text"])
    assert a == b


def test_similar_texts_score_higher_than_dissimilar():
    vecs = embed(["cat", "kitten", "database"])
    sim_close = cosine_similarity(vecs[0], vecs[1])   # cat vs kitten
    sim_far = cosine_similarity(vecs[0], vecs[2])      # cat vs database
    assert sim_close > sim_far


def test_embed_single_text():
    vecs = embed(["solo"])
    assert len(vecs) == 1
    assert len(vecs[0]) > 0


def test_model_is_cached(monkeypatch):
    import omna.embedder as emb_module
    original_cache = dict(emb_module._cache)
    embed(["warm up"])
    first_model = emb_module._cache.get(DEFAULT_MODEL)
    embed(["again"])
    assert emb_module._cache.get(DEFAULT_MODEL) is first_model
