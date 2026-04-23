"""Day 2 tests — Rust cosine similarity kernel."""
import math
import pytest
from _omna import cosine_similarity, top_k


def test_identical_vectors_score_one():
    v = [1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v, v), 1.0)


def test_orthogonal_vectors_score_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_opposite_vectors_score_minus_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_top_k_returns_correct_count():
    query = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7]]
    results = top_k(query, embeddings, 2)
    assert len(results) == 2


def test_top_k_sorted_descending():
    query = [1.0, 0.0]
    embeddings = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    results = top_k(query, embeddings, 3)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_best_index_first():
    query = [1.0, 0.0]
    embeddings = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    results = top_k(query, embeddings, 1)
    assert results[0][0] == 1  # index 1 is the perfect match


def test_top_k_k_larger_than_embeddings():
    query = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    results = top_k(query, embeddings, 100)
    assert len(results) == 2
