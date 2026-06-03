"""Shared utilities for the three Graph RAG notebooks.

Mirrors the shape of 01-intro-to-rag/helpers.py: a typed Config, a loud
env loader, single-object client constructors. Every notebook opens with:

    from helpers import load_env, get_openai_client, get_kuzu_conn
    cfg = load_env()
    conn = get_kuzu_conn()
    client = get_openai_client(cfg)  # NB3 only

Kuzu is embedded, so there's no API key for it. The database is just a
directory on disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KUZU_PATH = Path(__file__).resolve().parent / "data" / "movies.kuzu"


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_chat_model: str


_REQUIRED = {
    "OPENAI_API_KEY": "openai_api_key",
}

_OPTIONAL_DEFAULTS = {
    "OPENAI_CHAT_MODEL": ("openai_chat_model", "gpt-4o-mini"),
}


def load_env(dotenv_path: str | Path | None = None) -> Config:
    """Load a .env file and return a typed Config.

    Fails loudly if required keys are missing so the reader doesn't chase a
    401 when the real problem is that .env is in the wrong directory.
    """
    from dotenv import load_dotenv

    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent / ".env"
    loaded = load_dotenv(dotenv_path, override=False)

    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        hint = (
            f"Missing required env vars: {missing}. "
            f"Looked for .env at {dotenv_path} (loaded={loaded}). "
            "Copy .env.example to .env and fill in your keys."
        )
        raise RuntimeError(hint)

    values = {field: os.environ[env_key] for env_key, field in _REQUIRED.items()}
    for env_key, (field, default) in _OPTIONAL_DEFAULTS.items():
        values[field] = os.getenv(env_key, default)
    return Config(**values)


def get_openai_client(cfg: Config):
    from openai import OpenAI

    return OpenAI(api_key=cfg.openai_api_key)


def get_kuzu_conn(path: str | Path | None = None):
    """Open or create a Kuzu database at `path` and return a Connection.

    Kuzu stores everything in a single directory. If it doesn't exist, Kuzu
    creates it on first open. We hold no global state: each call returns a
    fresh Connection. Cheap, so it's fine to call once per notebook.
    """
    import kuzu

    if path is None:
        path = DEFAULT_KUZU_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(path))
    return kuzu.Connection(db)


def cypher_to_df(query_result):
    """Turn a Kuzu QueryResult into a tidy pandas DataFrame for display.

    Analog of module 01's format_matches_table. Notebooks display the
    result directly; this is the one-liner that goes between
    conn.execute(...) and the cell output.
    """
    import pandas as pd

    return query_result.get_as_df() if hasattr(query_result, "get_as_df") else pd.DataFrame(query_result)
