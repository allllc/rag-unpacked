"""Shared utilities for the three RAG notebooks.

Keeps env loading, client construction, and small reusable helpers in one
place so each notebook's cells can focus on the concept being taught, not
boilerplate. Every notebook opens with:

    from helpers import load_env, get_pinecone_client, get_openai_client
    cfg = load_env()
    pc = get_pinecone_client(cfg)
"""
from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Config:
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index_name: str
    openai_api_key: str
    openai_embed_model: str
    openai_chat_model: str


_REQUIRED = {
    "PINECONE_API_KEY": "pinecone_api_key",
    "OPENAI_API_KEY": "openai_api_key",
}

_OPTIONAL_DEFAULTS = {
    "PINECONE_CLOUD": ("pinecone_cloud", "aws"),
    "PINECONE_REGION": ("pinecone_region", "us-east-1"),
    "PINECONE_INDEX_NAME": ("pinecone_index_name", "rag-unpacked-intro"),
    "OPENAI_EMBED_MODEL": ("openai_embed_model", "text-embedding-3-small"),
    "OPENAI_CHAT_MODEL": ("openai_chat_model", "gpt-4o-mini"),
}


def load_env(dotenv_path: str | Path | None = None) -> Config:
    """Load a .env file and return a typed Config.

    Fails loudly if required keys are missing so the reader doesn't chase a
    401 when the real problem is that `.env` is in the wrong directory.
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


def get_pinecone_client(cfg: Config, pool_threads: int | None = None):
    from pinecone import Pinecone

    if pool_threads is None:
        return Pinecone(api_key=cfg.pinecone_api_key)
    return Pinecone(api_key=cfg.pinecone_api_key, pool_threads=pool_threads)


def get_openai_client(cfg: Config):
    from openai import OpenAI

    return OpenAI(api_key=cfg.openai_api_key)


def chunks(iterable: Iterable, batch_size: int = 100) -> Iterator[tuple]:
    """Yield successive tuples of size `batch_size` from `iterable`."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def ensure_index(
    pc,
    name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    wait: bool = True,
):
    """Create the index if it doesn't exist. Safe to re-run.

    Returns the index handle (`pc.Index(name)`).
    """
    from pinecone import ServerlessSpec

    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        if wait:
            wait_for_index_ready(pc, name)
    return pc.Index(name)


def wait_for_index_ready(pc, name: str, timeout: float = 60.0, poll: float = 1.0):
    """Block until `describe_index(name).status.ready` is True."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = pc.describe_index(name)
        ready = getattr(getattr(desc, "status", None), "ready", None)
        if ready is None and isinstance(desc, dict):
            ready = desc.get("status", {}).get("ready")
        if ready:
            return
        time.sleep(poll)
    raise TimeoutError(f"Index {name!r} not ready within {timeout}s")


def make_toy_movies(seed: int = 42) -> list[dict]:
    """Seeded 20-movie catalog with 16-dim vectors + rich metadata.

    Vectors are random-but-deterministic; metadata is hand-authored so
    filter demos have something meaningful to filter on.
    """
    rng = np.random.default_rng(seed)
    catalog = [
        ("m_01", "The Vanishing Signal", "thriller", 2014, 7.8),
        ("m_02", "Quiet Orbit", "sci-fi", 2019, 8.1),
        ("m_03", "Paper Kingdom", "drama", 2012, 7.2),
        ("m_04", "Last Train to Osaka", "thriller", 2017, 8.3),
        ("m_05", "Bright Hollow", "drama", 2021, 6.9),
        ("m_06", "Neon Cartography", "sci-fi", 2023, 8.6),
        ("m_07", "The Glass Hour", "thriller", 2016, 7.5),
        ("m_08", "Parallel Mothers", "drama", 2021, 7.8),
        ("m_09", "Moonwake", "sci-fi", 2015, 7.1),
        ("m_10", "A Quiet Grammar", "drama", 2018, 7.9),
        ("m_11", "Northbound", "thriller", 2024, 8.0),
        ("m_12", "Static Gardens", "sci-fi", 2013, 6.8),
        ("m_13", "The Cold Room", "horror", 2019, 7.4),
        ("m_14", "Soft Machines", "sci-fi", 2022, 8.4),
        ("m_15", "Hollow Tide", "horror", 2016, 6.5),
        ("m_16", "Lantern East", "drama", 2020, 7.6),
        ("m_17", "Feral Static", "horror", 2023, 7.7),
        ("m_18", "The Depot", "thriller", 2015, 7.0),
        ("m_19", "Tidewalker", "drama", 2011, 7.3),
        ("m_20", "Iron Archive", "sci-fi", 2025, 8.2),
    ]
    records = []
    for mid, title, genre, year, rating in catalog:
        vec = rng.normal(size=16).astype(np.float32)
        vec /= np.linalg.norm(vec)
        records.append(
            {
                "id": mid,
                "values": vec.tolist(),
                "metadata": {
                    "title": title,
                    "genre": genre,
                    "year": year,
                    "rating": rating,
                },
            }
        )
    return records


def format_matches_table(query_result) -> "pd.DataFrame":
    """Turn a Pinecone query result into a tidy pandas table for display."""
    import pandas as pd

    rows = []
    matches = query_result.get("matches") if isinstance(query_result, dict) else getattr(query_result, "matches", [])
    for m in matches:
        md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        rows.append({"id": mid, "score": round(float(score), 4) if score is not None else None, **(md or {})})
    return pd.DataFrame(rows)
