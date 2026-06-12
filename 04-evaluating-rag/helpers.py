"""Shared utilities for the four Evaluating RAG notebooks.

Same shape as 03-vectorless-rag/helpers.py: a typed Config, a loud env
loader, single-object client constructors. PINECONE keys are optional:
NB3's head-to-head runs if you have them, gracefully skips the vector
and hybrid arms if you don't.

NB1 doesn't call load_env() at all. It's entirely local: hand-rolled
retrieval metrics on a toy example plus BM25 over the seed corpus, no
LLM calls.

NB2, NB3, NB4 each call load_env() once at the top:

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

DEFAULT_KUZU_PATH = Path(__file__).resolve().parent.parent / "02-graph-rag" / "data" / "movies.kuzu"


def load_env(dotenv_path: str | Path | None = None) -> Config:
    """Load a .env file and return a typed Config.

    Fails loudly if required keys (just OPENAI_API_KEY) are missing.
    PINECONE_API_KEY is optional: if absent, Config.pinecone_api_key is
    None and NB3's vector arm prints a graceful skip note pointing at
    module 01's setup.
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
    values["pinecone_api_key"] = os.getenv("PINECONE_API_KEY") or None
    return Config(**values)


def get_openai_client(cfg: Config):
    from openai import OpenAI

    return OpenAI(api_key=cfg.openai_api_key)


def get_pinecone_client(cfg: Config):
    """Return a Pinecone client or None if the API key isn't set.

    Lets NB3's head-to-head gracefully degrade to BM25-only when the
    reader hasn't shipped module 01 yet.
    """
    if not cfg.pinecone_api_key:
        return None
    from pinecone import Pinecone

    return Pinecone(api_key=cfg.pinecone_api_key)


def get_kuzu_conn(path: str | Path | None = None):
    """Open module 02's Kuzu database read-only-ish and return a Connection.

    Default points at the module 02 build artifact. NB4 needs this. If
    the DB is missing, raises FileNotFoundError with a clear pointer at
    module 02's build script (same friction-floor pattern module 03 uses
    for the Pinecone arm).
    """
    import kuzu

    if path is None:
        path = DEFAULT_KUZU_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kuzu DB not found at {path}. NB4 reuses module 02's movie graph. "
            "Run `python scripts/build_corpus.py` inside 02-graph-rag/ first."
        )
    db = kuzu.Database(str(path))
    return kuzu.Connection(db)


# Hand-rolled retrieval metrics. Used in NB1 (where they're also derived from
# scratch on a toy example) and in NB3's matrix.

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """|retrieved[:k] ∩ relevant| / k.

    Edge case for adversarial questions (relevant_ids empty): returns
    1.0 if retrieved[:k] is also empty, else 0.0. Precision is formally
    undefined for empty ground truth; we treat the "should-have-refused"
    case as a generation-side concern instead. NB1 explains this.
    """
    if k <= 0:
        return 0.0
    if not relevant_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """|retrieved[:k] ∩ relevant| / |relevant|.

    Returns 1.0 when relevant_ids is empty (nothing to recall). The
    adversarial questions get a free pass here; NB1 explains why.
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_lists: list[list], relevant_lists: list[list]) -> float:
    """Average 1/rank_of_first_relevant across queries.

    Skips queries with empty relevant_lists (averaging over zero ground
    truth is meaningless). A query whose retrieved list contains no
    relevant doc contributes 0.0.
    """
    rrs = []
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        if not relevant:
            continue
        relevant_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    if not rrs:
        return 0.0
    return sum(rrs) / len(rrs)


# BM25 + RRF helpers copied verbatim from 03-vectorless-rag/helpers.py.
# Cross-module imports are a code smell and modules have already drifted
# on helpers.py shape; copying keeps this module self-contained.

def build_bm25_index(docs: list[str], **bm25_kwargs):
    """Tokenize + index a list of documents in one call.

    Returns (retriever, tokens). bm25_kwargs are passed to bm25s.BM25();
    the interesting ones are k1 (TF saturation, default 1.5) and b
    (length normalization, default 0.75).
    """
    import bm25s

    tokens = bm25s.tokenize(docs, show_progress=False)
    retriever = bm25s.BM25(**bm25_kwargs)
    retriever.index(tokens, show_progress=False)
    return retriever, tokens


def retrieve(retriever, query: str, k: int = 5):
    """Tokenize + retrieve in one call.

    Returns (indices, scores) for the top-k matches as plain Python
    lists.
    """
    import bm25s

    q_tokens = bm25s.tokenize(query, show_progress=False)
    results, scores = retriever.retrieve(q_tokens, k=k, show_progress=False)
    return list(results[0]), [float(s) for s in scores[0]]


def rrf_combine(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked id lists.

    For each id, sum 1/(k + rank_in_each_list) across every list it
    appears in. Sort descending. The standard k is 60 (Cormack, Clarke,
    Buettcher 2009). No score normalization needed: BM25 scores and
    cosine similarity scores are on different scales, but their ranks
    are comparable.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# RAGAS wrappers. NB2/NB3/NB4 import these to wire up the LLM judge and
# the embeddings (the latter is needed for ResponseRelevancy).

def get_evaluator_llm(cfg: Config):
    """Wrap ChatOpenAI in RAGAS's LangchainLLMWrapper.

    NB2/NB3/NB4 pass the result to ragas.evaluate(..., llm=...).
    Centralized so the judge model is one config value.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=cfg.openai_chat_model,
            temperature=0,
            api_key=cfg.openai_api_key,
        )
    )


def get_evaluator_embeddings(cfg: Config):
    """Wrap OpenAIEmbeddings in RAGAS's LangchainEmbeddingsWrapper.

    Required for ResponseRelevancy: it cosine-compares the question
    against LLM-generated hypothetical questions. Without an embedder,
    that metric raises a default-embedder error.
    """
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=cfg.openai_embed_model,
            api_key=cfg.openai_api_key,
        )
    )
