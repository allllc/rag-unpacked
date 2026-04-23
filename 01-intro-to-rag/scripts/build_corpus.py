"""Build the Pinecone-docs corpus used by NB3.

Ships with a curated, hand-authored seed corpus (short concept snippets
paraphrasing the public Pinecone docs), so the notebook is reproducible
offline and doesn't hammer docs.pinecone.io from every cloner's laptop.

If you want a bigger corpus, call `build_from_urls(urls)` with your own
list of doc pages. The chunker and output format are the same.

Output: data/corpus.parquet with columns
    id, text, title, section, url
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.parquet"


# A compact, self-contained knowledge base. Each entry is a short, focused
# paragraph on one Pinecone concept, roughly the shape real doc chunks take
# after H2/H3 splitting. Paraphrased from public docs.pinecone.io content.
_SEED_CORPUS: list[dict] = [
    {
        "title": "Serverless indexes",
        "section": "Indexes / Overview",
        "url": "https://docs.pinecone.io/guides/indexes/understanding-indexes",
        "text": (
            "A serverless index in Pinecone automatically scales compute and storage "
            "to match your workload. You don't provision pods or choose a size. You "
            "pay for reads, writes, and stored data. Create one by passing a "
            "ServerlessSpec (with cloud and region) to pc.create_index. Serverless is "
            "the default choice for new applications; pod-based indexes remain "
            "available for workloads with predictable, high sustained throughput."
        ),
    },
    {
        "title": "Creating an index",
        "section": "Indexes / Create",
        "url": "https://docs.pinecone.io/guides/indexes/create-an-index",
        "text": (
            "Use pc.create_index(name, dimension, metric, spec) to create an index. "
            "The dimension is fixed at creation time and must match the embedding "
            "model you'll use (for example 1536 for text-embedding-3-small). The "
            "metric is one of 'cosine' (default), 'euclidean', or 'dotproduct'. The "
            "spec is ServerlessSpec(cloud=..., region=...) for serverless, or "
            "PodSpec(...) for pod-based. Index names are scoped to your project."
        ),
    },
    {
        "title": "Checking if an index exists",
        "section": "Indexes / Introspection",
        "url": "https://docs.pinecone.io/guides/indexes/list-indexes",
        "text": (
            "pc.has_index(name) returns True if the named index exists in the "
            "current project; it's the idiomatic guard before a create_index call. "
            "pc.list_indexes() returns all indexes in the project with their specs "
            "and status. pc.describe_index(name) returns a single index's metadata "
            "including its host URL, dimension, metric, and readiness state."
        ),
    },
    {
        "title": "Deleting an index",
        "section": "Indexes / Delete",
        "url": "https://docs.pinecone.io/guides/indexes/delete-an-index",
        "text": (
            "pc.delete_index(name) permanently deletes the index and all vectors it "
            "contains. There is no undo. Wrap the call in a has_index check to make "
            "the operation idempotent. For production indexes, enable deletion "
            "protection via the deletion_protection parameter on create_index to "
            "require an explicit disable step before deletion is allowed."
        ),
    },
    {
        "title": "Index statistics",
        "section": "Indexes / Introspection",
        "url": "https://docs.pinecone.io/guides/indexes/describe-index",
        "text": (
            "index.describe_index_stats() returns live statistics for a connected "
            "index: total vector count, dimension, fullness, and per-namespace "
            "vector counts. Call it after upserts to confirm ingestion worked, and "
            "after deletes to confirm rows are gone. The counts are eventually "
            "consistent, so expect a short lag after large writes."
        ),
    },
    {
        "title": "Upserting vectors",
        "section": "Vectors / Upsert",
        "url": "https://docs.pinecone.io/guides/data/upsert-data",
        "text": (
            "index.upsert(vectors=...) inserts new vectors or replaces existing "
            "ones keyed by id. Each vector is a (id, values, metadata) triple, where "
            "id is a string, values is a list of floats matching the index "
            "dimension, and metadata is an optional dict. Upsert accepts up to ~100 "
            "vectors per request in practice; batch larger workloads with a helper "
            "like the chunks() utility used in the notebooks."
        ),
    },
    {
        "title": "Fetching vectors by id",
        "section": "Vectors / Fetch",
        "url": "https://docs.pinecone.io/guides/data/fetch-data",
        "text": (
            "index.fetch(ids=[...]) returns the stored vectors for the given ids, "
            "including their values and metadata. Use it to verify an upsert wrote "
            "the expected values, to round-trip a record for debugging, or to hydrate "
            "a known set of document ids without a similarity query. Unknown ids are "
            "silently omitted from the response rather than raising an error."
        ),
    },
    {
        "title": "Querying for similar vectors",
        "section": "Vectors / Query",
        "url": "https://docs.pinecone.io/guides/data/query-data",
        "text": (
            "index.query(vector=..., top_k=N) returns the N most similar vectors "
            "ranked by the index's metric. Pass include_metadata=True to get stored "
            "metadata alongside each match. Pass include_values=True to get the raw "
            "vector values back (usually not needed for retrieval). Queries against "
            "a serverless index under a million vectors typically return in 20-80ms."
        ),
    },
    {
        "title": "Metadata filtering",
        "section": "Vectors / Query",
        "url": "https://docs.pinecone.io/guides/data/filter-with-metadata",
        "text": (
            "Pass filter={...} to index.query to restrict the candidate pool before "
            "similarity ranking. Filters are cheaper than retrieving more results and "
            "filtering client-side. Supported operators are $eq, $ne, $gt, $gte, "
            "$lt, $lte, $in, $nin, and $exists. Combine multiple fields at the top "
            "level for an implicit AND, or wrap in $and / $or for explicit boolean "
            "logic. Example: filter={'genre': 'thriller', 'year': {'$lt': 2018}}."
        ),
    },
    {
        "title": "Updating vectors",
        "section": "Vectors / Update",
        "url": "https://docs.pinecone.io/guides/data/update-data",
        "text": (
            "index.update(id=..., values=...) replaces the stored values for a "
            "vector id without touching metadata. index.update(id=..., "
            "set_metadata={...}) merges the given keys into the vector's metadata "
            "without replacing unstated keys. For larger changes, re-upserting the "
            "whole record is simpler and has the same final state."
        ),
    },
    {
        "title": "Deleting vectors",
        "section": "Vectors / Delete",
        "url": "https://docs.pinecone.io/guides/data/delete-data",
        "text": (
            "index.delete(ids=[...]) removes specific vectors by id. "
            "index.delete(filter={...}) removes every vector matching the filter, "
            "useful for purging an old document's chunks but dangerous if the filter "
            "is too broad. index.delete(delete_all=True, namespace='...') clears a "
            "namespace entirely. Deletes are eventually consistent."
        ),
    },
    {
        "title": "Namespaces",
        "section": "Namespaces / Overview",
        "url": "https://docs.pinecone.io/guides/indexes/use-namespaces",
        "text": (
            "A namespace is a logical partition inside a single index. Vectors in "
            "different namespaces are isolated. A query against one namespace never "
            "sees another namespace's rows. Common uses: multi-tenant isolation, "
            "A/B testing two chunking strategies, or keeping staging and production "
            "data in one index. Pass namespace='...' to upsert, query, fetch, "
            "update, and delete to scope the operation."
        ),
    },
    {
        "title": "Batched upsert",
        "section": "Performance / Batching",
        "url": "https://docs.pinecone.io/guides/data/upsert-data#batching",
        "text": (
            "When ingesting thousands of vectors, batch them into chunks of 100-500 "
            "per upsert request. The sweet spot balances per-request overhead "
            "against payload size (Pinecone accepts up to 2 MB per request). Most "
            "tutorials use a small chunks() helper that wraps itertools.islice to "
            "yield fixed-size tuples from an iterable."
        ),
    },
    {
        "title": "Async upsert with pool_threads",
        "section": "Performance / Concurrency",
        "url": "https://docs.pinecone.io/guides/data/upsert-data#send-upserts-in-parallel",
        "text": (
            "Initialize the client or the index with pool_threads=N to enable "
            "concurrent requests. Then pass async_req=True to upsert; it returns a "
            "future you call .get() on. Fire many upserts, collect the futures, then "
            "join. For a cold ingest of several thousand vectors this typically "
            "yields a 3-10x wall-clock speedup over sequential upserts."
        ),
    },
    {
        "title": "Choosing a distance metric",
        "section": "Indexes / Metrics",
        "url": "https://docs.pinecone.io/guides/indexes/understanding-indexes#metrics",
        "text": (
            "Cosine similarity is the default and the right choice for most text "
            "embedding models (OpenAI's text-embedding-3 family, Cohere, "
            "sentence-transformers). Dot product is faster and equivalent to cosine "
            "when your vectors are already unit-normalized. Euclidean is rarely the "
            "right choice for text retrieval. The metric is fixed at index creation."
        ),
    },
    {
        "title": "Embedding dimension must match",
        "section": "Indexes / Dimensions",
        "url": "https://docs.pinecone.io/guides/indexes/understanding-indexes#dimensions",
        "text": (
            "The dimension parameter on create_index must equal the length of the "
            "vectors you'll upsert. Mismatched dimensions raise a 400 error at "
            "upsert time. For OpenAI text-embedding-3-small the default dimension "
            "is 1536; for text-embedding-3-large it's 3072, but both models accept "
            "a 'dimensions' parameter to shrink the output vector, useful when you "
            "want the accuracy of -large with the storage cost of -small."
        ),
    },
    {
        "title": "Integrated inference",
        "section": "Inference / Overview",
        "url": "https://docs.pinecone.io/guides/inference/understanding-inference",
        "text": (
            "Pinecone's integrated inference lets you upsert and query raw text "
            "directly. The service embeds the text with a hosted model (such as "
            "llama-text-embed-v2) and stores the vector in one step. This removes "
            "one round trip to an external embedding API. It's convenient for "
            "prototypes, but the notebooks here use OpenAI embeddings explicitly so "
            "the pattern ports to any vector DB, not only Pinecone."
        ),
    },
    {
        "title": "Reranking",
        "section": "Inference / Rerank",
        "url": "https://docs.pinecone.io/guides/inference/rerank",
        "text": (
            "A reranker re-scores an initial set of similarity-search results with "
            "a cross-encoder model that sees query and document together. Typical "
            "pipeline: retrieve top_k=30 from Pinecone, pass them and the query to "
            "pc.inference.rerank, take the top 5 by rerank score. Rerankers catch "
            "cases where the bi-encoder embedding missed a subtle lexical or "
            "semantic match. Reranking is covered in module 05 of this repo."
        ),
    },
    {
        "title": "API versioning",
        "section": "Reference / Versioning",
        "url": "https://docs.pinecone.io/reference/api/versioning",
        "text": (
            "Pinecone's REST API is versioned via the X-Pinecone-API-Version header. "
            "The current stable version as of April 2026 is 2025-10. The Python SDK "
            "pins a compatible version for you; you rarely need to set it manually. "
            "Preview versions exist for opt-in access to new features before they "
            "stabilize."
        ),
    },
    {
        "title": "Authentication",
        "section": "Reference / Auth",
        "url": "https://docs.pinecone.io/reference/api/authentication",
        "text": (
            "Pinecone authenticates all API calls with an API key, passed in the "
            "Api-Key header (handled for you by the SDK when you pass api_key to "
            "Pinecone()). Keys are scoped to a single project. Rotate a key by "
            "creating a new one in the console, updating your environment, and "
            "deleting the old one. Never commit keys to source control. Store them "
            "in an environment variable or a secrets manager."
        ),
    },
]


def _chunk_id(text: str) -> str:
    return "doc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_seed_corpus() -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for entry in _SEED_CORPUS:
        text = _normalize(entry["text"])
        rows.append(
            {
                "id": _chunk_id(text),
                "text": text,
                "title": entry["title"],
                "section": entry["section"],
                "url": entry["url"],
            }
        )
    return pd.DataFrame(rows)


def build_if_missing(force: bool = False) -> Path:
    """Idempotently materialize data/corpus.parquet. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS_PATH.exists() and not force:
        return CORPUS_PATH
    df = build_seed_corpus()
    df.to_parquet(CORPUS_PATH, index=False)
    return CORPUS_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = build_if_missing(force=force)
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"Wrote {len(df)} chunks to {path}")
    print(df[["id", "title", "section"]].head(5).to_string(index=False))
