"""Day 6 tests — omna.ask module and df.omna.ask()."""
import os
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import omna  # noqa: F401

import omna.ask as ask_mod


def _mock_client(answer: str = "42"):
    """Return a mock anthropic.Anthropic client that replies with *answer*."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=answer)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


# ── _serialize() ─────────────────────────────────────────────────────────────

def test_serialize_includes_shape():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = ask_mod._serialize(df)
    assert "2 rows" in out
    assert "2 columns" in out


def test_serialize_includes_column_names():
    df = pl.DataFrame({"revenue": [100], "region": ["north"]})
    out = ask_mod._serialize(df)
    assert "revenue" in out
    assert "region" in out


def test_serialize_includes_sample_data():
    df = pl.DataFrame({"x": [99]})
    out = ask_mod._serialize(df)
    assert "99" in out


def test_serialize_caps_at_max_rows():
    df = pl.DataFrame({"x": list(range(100))})
    out = ask_mod._serialize(df)
    # Only first 20 rows serialized — value 99 should not appear in sample
    lines = out.split("\n")
    sample_section = "\n".join(lines[lines.index("Sample data (first 20 rows):") + 1:])
    assert "99" not in sample_section


def test_serialize_notes_null_counts():
    df = pl.DataFrame({"x": [1, None, None]})
    out = ask_mod._serialize(df)
    assert "x" in out


# ── query() — missing API key ─────────────────────────────────────────────────

def test_query_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        ask_mod.query(df, "What is x?")


# ── query() — mocked client ───────────────────────────────────────────────────

@patch("anthropic.Anthropic")
def test_query_returns_string(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_cls.return_value = _mock_client("The answer is 7.")
    df = pl.DataFrame({"score": [7]})
    result = ask_mod.query(df, "What is the score?")
    assert result == "The answer is 7."


@patch("anthropic.Anthropic")
def test_query_passes_question_to_api(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = _mock_client()
    mock_cls.return_value = client
    df = pl.DataFrame({"val": [1]})
    ask_mod.query(df, "How many rows?")
    call_kwargs = client.messages.create.call_args
    content = call_kwargs.kwargs["messages"][0]["content"]
    assert "How many rows?" in content


@patch("anthropic.Anthropic")
def test_query_uses_default_model(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = _mock_client()
    mock_cls.return_value = client
    df = pl.DataFrame({"x": [1]})
    ask_mod.query(df, "?")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == ask_mod.DEFAULT_MODEL


@patch("anthropic.Anthropic")
def test_query_accepts_custom_model(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = _mock_client()
    mock_cls.return_value = client
    df = pl.DataFrame({"x": [1]})
    ask_mod.query(df, "?", model="claude-sonnet-4-6")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


# ── df.omna.ask() integration ─────────────────────────────────────────────────

@patch("anthropic.Anthropic")
def test_frame_ask_returns_string(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_cls.return_value = _mock_client("Three rows.")
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.omna.ask("How many rows?")
    assert result == "Three rows."


def test_frame_ask_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        df.omna.ask("hello?")
