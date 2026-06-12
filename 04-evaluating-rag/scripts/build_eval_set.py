"""Build the 20-question golden eval set used by NB1, NB2, NB3.

Twenty hand-authored question / ground-truth-answer / relevant-doc-ids
tuples on module 01's 20-doc Pinecone corpus. Engineered to expose
retriever differences across four shapes:

  - keyword (6): literal identifier or rare API token in the question.
    BM25 wins on context precision via IDF reward (pool_threads, $in,
    has_index, delete_all, fetch, Api-Key).
  - paraphrase (5): the question uses synonyms; the doc text doesn't.
    Vector retrieval wins on context recall (integrated inference,
    similarity score, rerank, staging vs production, push many at once).
  - synthesis (5): multiple relevant docs; both retrievers should pull
    more than one. Generation quality matters more than ranking here.
  - adversarial (4): not in the corpus. Faithfulness should be high
    (the LLM refuses) and answer relevance should be low.

The doc_ids below are verified to exist in
01-intro-to-rag/data/corpus.parquet at build time of this seed.

Output: data/golden.parquet (~8 KB).
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "golden.parquet"


_SEED_QUESTIONS: list[dict] = [
    # --- keyword (BM25-favored) ---
    {
        "question": "What does the pool_threads parameter do?",
        "ground_truth_answer": (
            "It opens that many parallel HTTP connections so upsert "
            "requests can run concurrently and overlap network latency."
        ),
        "relevant_doc_ids": ["doc_7eaf9d85e4b5"],
        "shape": "keyword",
    },
    {
        "question": "How do I use the $in operator in a metadata filter?",
        "ground_truth_answer": (
            "$in matches when the field equals any value in a list, "
            "for example {'tag': {'$in': ['a','b']}}."
        ),
        "relevant_doc_ids": ["doc_50ae76610fa0"],
        "shape": "keyword",
    },
    {
        "question": "What does pc.has_index(name) return?",
        "ground_truth_answer": (
            "True if an index with that name exists in the current "
            "project. It's the idiomatic guard before create_index."
        ),
        "relevant_doc_ids": ["doc_42de0361ce4a"],
        "shape": "keyword",
    },
    {
        "question": "What argument do I pass to delete() to wipe every vector in a namespace?",
        "ground_truth_answer": (
            "delete_all=True, with an optional namespace='...'. It "
            "removes every vector in scope and cannot be undone."
        ),
        "relevant_doc_ids": ["doc_c035ffcf5af2"],
        "shape": "keyword",
    },
    {
        "question": "What does fetch(ids=[...]) return?",
        "ground_truth_answer": (
            "The stored vectors for the requested ids. It silently "
            "omits any id not present, no error raised."
        ),
        "relevant_doc_ids": ["doc_dd8a34e59930"],
        "shape": "keyword",
    },
    {
        "question": "How do I authenticate to the Pinecone REST API directly?",
        "ground_truth_answer": (
            "Send the Api-Key header on each request. The Python SDK "
            "does this for you when you pass api_key=... to Pinecone()."
        ),
        "relevant_doc_ids": ["doc_325231fca76b"],
        "shape": "keyword",
    },
    # --- paraphrase (vector-favored) ---
    {
        "question": "How can I store text without computing embeddings on my own machine?",
        "ground_truth_answer": (
            "Use Pinecone's integrated inference: pass raw text in "
            "upsert and query, and Pinecone runs an embedding model "
            "server-side."
        ),
        "relevant_doc_ids": ["doc_f194968125f2"],
        "shape": "paraphrase",
    },
    {
        "question": "Which similarity score should I pick for OpenAI embeddings?",
        "ground_truth_answer": (
            "Cosine, because OpenAI embeddings are unit-normalized "
            "and cosine matches their training objective."
        ),
        "relevant_doc_ids": ["doc_318824736278"],
        "shape": "paraphrase",
    },
    {
        "question": "Can I make Pinecone reorder my search hits with a smarter model on top?",
        "ground_truth_answer": (
            "Yes. Pinecone's reranking inference takes the top-N from "
            "a query and re-scores them with a cross-encoder for a "
            "sharper top-K."
        ),
        "relevant_doc_ids": ["doc_de1fe4e4d84e"],
        "shape": "paraphrase",
    },
    {
        "question": "How do I keep my staging data separate from production inside one index?",
        "ground_truth_answer": (
            "Put each environment in its own namespace. Queries are "
            "scoped to one namespace and can't see another's vectors."
        ),
        "relevant_doc_ids": ["doc_1fa92232a250"],
        "shape": "paraphrase",
    },
    {
        "question": "What's the right way to push many vectors at once for speed?",
        "ground_truth_answer": (
            "Batched upsert: chunk records into groups of about 100, "
            "often combined with pool_threads parallelism to overlap "
            "network latency."
        ),
        "relevant_doc_ids": ["doc_5cdf73fb4299", "doc_7eaf9d85e4b5"],
        "shape": "paraphrase",
    },
    # --- synthesis (multi-doc) ---
    {
        "question": "Walk me through creating a serverless cosine index for text-embedding-3-small.",
        "ground_truth_answer": (
            "Call pc.create_index(name, dimension=1536, metric='cosine', "
            "spec=ServerlessSpec(cloud=..., region=...)). Check "
            "pc.has_index first to make it idempotent."
        ),
        "relevant_doc_ids": [
            "doc_cb65d310d336",
            "doc_4d45391193b6",
            "doc_318824736278",
            "doc_928bebc1453d",
            "doc_42de0361ce4a",
        ],
        "shape": "synthesis",
    },
    {
        "question": "How do I delete a single vector vs the whole index?",
        "ground_truth_answer": (
            "Vectors: index.delete(ids=[...]) or delete_all=True for "
            "a namespace. Whole index: pc.delete_index(name). Both are "
            "permanent."
        ),
        "relevant_doc_ids": ["doc_c035ffcf5af2", "doc_1729940ac580"],
        "shape": "synthesis",
    },
    {
        "question": "What's the difference between updating a vector and upserting it?",
        "ground_truth_answer": (
            "Upsert replaces the whole record by id. Update merges "
            "only the keys you pass (values, metadata fields) and "
            "leaves the rest untouched."
        ),
        "relevant_doc_ids": ["doc_1c5230d259c1", "doc_549cffb8e06c"],
        "shape": "synthesis",
    },
    {
        "question": "How do I check that my upserts actually landed?",
        "ground_truth_answer": (
            "Call index.describe_index_stats() and compare the "
            "namespace-level vector counts to what you sent. "
            "Ingestion is eventually consistent so allow a short lag."
        ),
        "relevant_doc_ids": ["doc_0a2ddfe050b1", "doc_549cffb8e06c"],
        "shape": "synthesis",
    },
    {
        "question": "I'm querying with cosine but my embedding model has dimension 3072. What goes wrong?",
        "ground_truth_answer": (
            "The upsert and query fail with a dimension-mismatch "
            "error. The index dimension must equal the embedding "
            "model's output size."
        ),
        "relevant_doc_ids": ["doc_928bebc1453d", "doc_318824736278"],
        "shape": "synthesis",
    },
    # --- adversarial (not in corpus) ---
    {
        "question": "What's the monthly price of a serverless index in dollars?",
        "ground_truth_answer": "Not stated in the provided context.",
        "relevant_doc_ids": [],
        "shape": "adversarial",
    },
    {
        "question": "Which Pinecone region is fastest for users in Singapore?",
        "ground_truth_answer": "Not stated in the provided context.",
        "relevant_doc_ids": [],
        "shape": "adversarial",
    },
    {
        "question": "How do I configure SSO with Okta on Pinecone?",
        "ground_truth_answer": "Not stated in the provided context.",
        "relevant_doc_ids": [],
        "shape": "adversarial",
    },
    {
        "question": "Compare Pinecone's pricing to Weaviate's.",
        "ground_truth_answer": "Not stated in the provided context.",
        "relevant_doc_ids": [],
        "shape": "adversarial",
    },
]


def build_seed_eval_set():
    import pandas as pd

    rows = []
    for i, entry in enumerate(_SEED_QUESTIONS, start=1):
        rows.append(
            {
                "question_id": f"q{i:02d}",
                "question": entry["question"],
                "ground_truth_answer": entry["ground_truth_answer"],
                "relevant_doc_ids": list(entry["relevant_doc_ids"]),
                "shape": entry["shape"],
            }
        )
    return pd.DataFrame(rows)


def build_if_missing(force: bool = False) -> Path:
    """Idempotently materialize data/golden.parquet. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EVAL_PATH.exists() and not force:
        return EVAL_PATH
    df = build_seed_eval_set()
    df.to_parquet(EVAL_PATH, index=False)
    return EVAL_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = build_if_missing(force=force)
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"Wrote {len(df)} questions to {path}")
    print(df[["question_id", "shape", "question"]].to_string(index=False))
    counts = df["shape"].value_counts().to_dict()
    print(f"\nShape breakdown: {counts}")
