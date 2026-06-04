# 03. Vectorless RAG with BM25

Three notebooks on the older, simpler, often-faster baseline that most people skip past on the way to vector retrieval. We build the BM25 scoring intuition from scratch, open up the knobs, then do a head-to-head against module 01's vector index on the same questions and resolve the leftover disagreements with hybrid retrieval via Reciprocal Rank Fusion.

**Crawl** (NB1) introduces the scoring intuition (IDF × TF saturation × length normalization) and runs the first queries on a 12-tip Python standard library corpus. **Walk** (NB2) opens up tokenization (stopwords, stemming), the `k1` and `b` parameters, where BM25 wins (rare identifiers) and where it loses (synonyms and paraphrases), plus persistence. **Run** (NB3) builds the full vectorless RAG loop with `gpt-4o-mini`, then switches to module 01's Pinecone corpus for a five-question head-to-head against vector search, then closes with hybrid retrieval via RRF.

Each notebook picks up where the last one left off.

![BM25 + Hybrid Vectorless RAG cheat sheet](./assets/cheatsheet.png)

## The three notebooks

| Notebook | Focus | Uses OpenAI? | Uses Pinecone? |
|---|---|---|---|
| [01. Introduction](./01_introduction.ipynb) | What BM25 is. Scoring intuition, the `bm25s` API in 5 lines, first queries on a 12-tip corpus. | No | No |
| [02. BM25 Mechanics](./02_bm25_mechanics.ipynb) | Tokenization, `k1`/`b` parameters, where BM25 wins and loses, persistence. | No | No |
| [03. Vectorless RAG, head-to-head, and hybrid](./03_vectorless_rag.ipynb) | The full vectorless RAG loop. Five-question head-to-head against module 01's Pinecone index. Hybrid via Reciprocal Rank Fusion. | Yes | Optional |

## Why `bm25s` and not `rank-bm25`

`bm25s` is a modern Python BM25 library that's roughly 10x faster than `rank-bm25` on real corpora. It has a simple API, pulls only `numpy` as a hard dependency, and the documentation is clean. `rank-bm25` is more pedagogically transparent (a few hundred lines of pure Python you can read), but for a portfolio piece that should also work on production-sized data, `bm25s` is the better baseline. If you want to read the math, the [`bm25s` paper](https://arxiv.org/abs/2407.03618) has it.

## Setup

```bash
cd 03-vectorless-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your OpenAI key. (Only NB3 actually uses it.)
# Optional: also paste a Pinecone key if you want NB3's head-to-head
# to run against module 01's existing vector index.
```

Get the OpenAI key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys). The Pinecone key is the same one module 01 uses; if you've already run module 01, you already have it.

### Build the corpus

```bash
python scripts/build_corpus.py
```

Writes the 12-tip Python standard library corpus to `data/corpus.parquet` (about 6 KB). Idempotent.

### Run

```bash
jupyter lab
```

## What it costs

NB1 and NB2 don't call OpenAI at all, only local BM25 retrieval over a 12-tip corpus.

NB3 makes a handful of `gpt-4o-mini` calls in Act I (three demo questions, each one chat completion) and Act II (five head-to-head questions, each one embedding call). Pinecone reads come from module 01's existing free-tier index, so no extra index storage cost. A full NB3 run is well under one cent.

If `PINECONE_API_KEY` isn't set, NB3 gracefully skips the vector arm and the hybrid demo (which needs both arms to fuse). Act I still runs in full.

## Layout

```
03-vectorless-rag/
├── 01_introduction.ipynb
├── 02_bm25_mechanics.ipynb
├── 03_vectorless_rag.ipynb
├── helpers.py                       # Config + load_env + 4 BM25/RRF helpers
├── .env.example
├── requirements.txt
├── assets/
│   ├── cheatsheet.png               # rendered above
│   └── cheatsheet.svg               # crisp, searchable source
├── scripts/
│   ├── build_corpus.py              # writes data/corpus.parquet (idempotent)
│   └── render_cheatsheet.py         # regenerates the cheat sheet
└── data/                            # gitignored, rebuilt by the scripts
    ├── corpus.parquet               # 12-tip Python standard library corpus
    └── bm25_index/                  # NB2 persistence demo output
```

`helpers.py` mirrors module 02's helper: it loads `.env`, builds the OpenAI client, optionally builds a Pinecone client (or returns `None` if the key is absent), and exposes `build_bm25_index`, `retrieve`, and `rrf_combine`. The shared opening of NB2 and NB3:

```python
from helpers import load_env, get_openai_client, build_bm25_index, retrieve, rrf_combine
cfg = load_env()
client = get_openai_client(cfg)  # NB3 only
retriever, _ = build_bm25_index(df["text"].tolist())
```

## Regenerating the cheat sheet

```bash
python scripts/render_cheatsheet.py
```

Writes both `assets/cheatsheet.svg` and `assets/cheatsheet.png`.

## Cleanup

There's no remote state. Delete `data/corpus.parquet` and `data/bm25_index/` for a fresh start. The build script will recreate them on the next run.

## A note on the head-to-head

NB3 Act II loads `01-intro-to-rag/data/corpus.parquet` directly via a relative path and queries module 01's existing `rag-unpacked-intro` Pinecone index. If you skipped module 01, NB3 will tell you exactly what to do: run module 01's `scripts/build_corpus.py` to populate the corpus file, and run module 01's NB3 end-to-end to populate the Pinecone index. Or skip the vector arm entirely: NB3's BM25-only path runs end-to-end without Pinecone.

## What this module doesn't cover

This module is the canonical BM25 + hybrid retrieval pattern. Real production retrieval layers in more:

- **Cross-encoder reranking**, the second-stage model that re-scores the top-N candidates after fusion. The fused list provides recall; the reranker provides precision. Module 05.
- **Learned sparse retrieval** (SPLADE), BM25's spirit with learned term weights. Module 05.
- **Query rewriting**, where the LLM rewrites the user's natural-language question into a keyword-rich search query before any retrieval happens. Module 05.
- **Evaluation**: faithfulness, context precision, answer relevance. How do you measure which arm actually wins on your traffic? Module 04.
