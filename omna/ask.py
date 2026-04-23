"""omna.ask — LLM-powered natural-language queries over DataFrames."""
from __future__ import annotations

import os

import polars as pl

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_SAMPLE_ROWS = 20


def _serialize(df: pl.DataFrame) -> str:
    """Build a compact text representation of *df* for the prompt."""
    n_rows, n_cols = df.shape
    schema_lines = [f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    null_info = {
        col: df[col].null_count()
        for col in df.columns
        if df[col].null_count() > 0
    }
    null_lines = (
        [f"  {col}: {cnt}" for col, cnt in null_info.items()]
        if null_info else ["  none"]
    )
    sample_rows = min(n_rows, _MAX_SAMPLE_ROWS)
    return (
        f"Shape: {n_rows} rows × {n_cols} columns\n\n"
        f"Schema:\n" + "\n".join(schema_lines) + "\n\n"
        f"Null counts:\n" + "\n".join(null_lines) + "\n\n"
        f"Sample data (first {sample_rows} rows):\n"
        + df.head(sample_rows).write_csv()
    )


def query(df: pl.DataFrame, question: str, model: str = DEFAULT_MODEL) -> str:
    """Answer a natural-language *question* about *df* using Claude.

    Sends the DataFrame schema and a sample of rows to Claude and returns the
    answer as a plain string. Only the first 20 rows are sent; statistics
    (shape, null counts) cover the full DataFrame.

    Requires the ANTHROPIC_API_KEY environment variable.

    Args:
        df: The DataFrame to query.
        question: Any natural-language question about the data.
        model: Claude model ID. Defaults to claude-haiku-4-5-20251001.

    Returns:
        Claude's answer as a string.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
        ImportError: If the anthropic package is not installed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it with:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The anthropic package is required for ask(). "
            "Install it with:  pip install omna[ask]\n"
            "Then set your key:  export ANTHROPIC_API_KEY=sk-ant-..."
        ) from None
    client = anthropic.Anthropic(api_key=api_key)
    data_context = _serialize(df)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "You are a data analyst assistant. "
            "Answer questions about the provided DataFrame accurately and concisely. "
            "If the answer cannot be determined from the given data, say so clearly."
        ),
        messages=[{
            "role": "user",
            "content": f"DataFrame:\n\n{data_context}\n\nQuestion: {question}",
        }],
    )
    return message.content[0].text
