"""Shared utilities for the three Vectorless RAG notebooks.

Same shape as 02-graph-rag/helpers.py: a typed Config, a loud env loader,
single-object client constructors. The only twist is that PINECONE keys
are optional: NB3's head-to-head against module 01's vector index runs
if you have the keys, and gracefully skips that arm if you don't.

NB1 and NB2 don't call load_env() at all. They're entirely local: pure
BM25 over the seed corpus, no LLM calls.

NB3 calls load_env() once at the top:

    from helpers import load_env, get_openai_client
    cfg = load_env()
    client = get_openai_client(cfg)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_chat_model: str
    openai_embed_model: str
    pinecone_api_key: str | None
    pinecone_index_name: str


_REQUIRED = {
    "OPENAI_API_KEY": "openai_api_key",
}

_OPTIONAL_DEFAULTS = {
    "OPENAI_CHAT_MODEL": ("openai_chat_model", "gpt-4o-mini"),
    "OPENAI_EMBED_MODEL": ("openai_embed_model", "text-embedding-3-small"),
    "PINECONE_INDEX_NAME": ("pinecone_index_name", "rag-unpacked-intro"),
}


def load_env(dotenv_path: str | Path | None = None) -> Config:
    """Load a .env file and return a typed Config.

    Fails loudly if required keys (just OPENAI_API_KEY) are missing.
    PINECONE_API_KEY is optional: if absent, Config.pinecone_api_key is
    None and NB3's vector arm prints a graceful skip note.
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

    values: dict = {field: os.environ[env_key] for env_key, field in _REQUIRED.items()}
    for env_key, (field, default) in _OPTIONAL_DEFAULTS.items():
        values[field] = os.getenv(env_key, default)
    # Pinecone API key: optional, None if absent
    values["pinecone_api_key"] = os.getenv("PINECONE_API_KEY") or None
    return Config(**values)


def get_openai_client(cfg: Config):
    from openai import OpenAI

    return OpenAI(api_key=cfg.openai_api_key)


def get_pinecone_client(cfg: Config):
    """Return a Pinecone client or None if the API key isn't set.

    Lets NB3 Act II's head-to-head gracefully degrade to BM25-only when
    the user hasn't shipped module 01 yet (or just doesn't want to use
    their free-tier quota again).
    """
    if not cfg.pinecone_api_key:
        return None
    from pinecone import Pinecone

    return Pinecone(api_key=cfg.pinecone_api_key)


def build_bm25_index(docs: list[str], **bm25_kwargs):
    """Tokenize + index a list of documents in one call.

    Wraps the three-line dance from NB1 so NB2 and NB3 stay focused
    on the questions being asked. Returns (retriever, tokens) so the
    caller can hold onto the tokens if they want to inspect them.

    bm25_kwargs are passed to bm25s.BM25(). The two interesting ones
    are k1 (TF saturation, default 1.5) and b (length normalization,
    default 0.75). NB2 demonstrates both.
    """
    import bm25s

    tokens = bm25s.tokenize(docs, show_progress=False)
    retriever = bm25s.BM25(**bm25_kwargs)
    retriever.index(tokens, show_progress=False)
    return retriever, tokens


def retrieve(retriever, query: str, k: int = 5):
    """Tokenize + retrieve in one call.

    Returns (indices, scores) for the top-k matches. Both are plain
    Python lists, not nested arrays, so the caller can iterate them
    directly without unwrapping the bm25s batch dimension.
    """
    import bm25s

    q_tokens = bm25s.tokenize(query, show_progress=False)
    results, scores = retriever.retrieve(q_tokens, k=k, show_progress=False)
    return list(results[0]), [float(s) for s in scores[0]]


def rrf_combine(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked id lists.

    For each id, sum 1/(k + rank_in_each_list) across every list it
    appears in. Sort descending. The standard k is 60 (Cormack, Clarke,
    Buettcher 2009); larger k flattens the curve so lower-ranked items
    contribute more, smaller k makes the top of each list dominate.

    The win of RRF over weighted-score fusion is that no score
    normalization is needed: BM25 scores and cosine similarity scores
    are on different scales, but their ranks are comparable.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
