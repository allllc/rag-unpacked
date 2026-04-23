# 01. Intro to RAG with Pinecone

Three notebooks that go from "what even is a vector database" to a working Q&A bot.

**Crawl** (NB1) gets five vectors into Pinecone and one query back out. No model, no corpus, no magic. **Walk** (NB2) takes the same shape of work and pushes on every side: filters, updates, deletes, namespaces. **Run** (NB3) drops the toys, loads a real corpus, embeds it with OpenAI, times batched vs async ingest, and wires the whole thing into a small Q&A bot over the Pinecone documentation.

Each notebook picks up where the last one left off.

![cheat sheet](./assets/cheatsheet.png)

## The three notebooks

| Notebook | Focus | Uses OpenAI? |
|---|---|---|
| [01. Introduction](./01_introduction.ipynb) | What a vector DB is. Five hand-written 4-d vectors, one query, one moment of "oh, that's all it is." | No |
| [02. Vector Manipulation](./02_vector_manipulation.ipynb) | CRUD, metadata filters, namespaces. Seeded 20-movie toy catalog. | No |
| [03. Performance and AI Applications](./03_performance_and_rag.ipynb) | Real embeddings, batched vs async ingest, and the full retrieve-augment-generate loop. | Yes |

## Setup

```bash
cd 01-intro-to-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your Pinecone + OpenAI keys.
```

Where to get the keys:

- Pinecone: [app.pinecone.io](https://app.pinecone.io/), Settings, API Keys. The free tier is plenty for this module.
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

Then:

```bash
jupyter lab
```

and open the notebooks in order.

## What it costs

NB1 and NB2 don't call OpenAI at all, only Pinecone's free-tier reads and writes.

NB3 embeds roughly twenty short docs with `text-embedding-3-small` once, then runs three chat completions with `gpt-4o-mini`. A full run costs well under one cent. Embeddings are cached to `data/embeddings.parquet` so re-runs don't re-spend credits.

## Layout

```
01-intro-to-rag/
├── 01_introduction.ipynb
├── 02_vector_manipulation.ipynb
├── 03_performance_and_rag.ipynb
├── helpers.py                      # shared config and utilities
├── .env.example
├── requirements.txt
├── assets/
│   ├── cheatsheet.png              # rendered above
│   └── cheatsheet.svg              # crisp, searchable source
├── scripts/
│   ├── build_corpus.py             # writes data/corpus.parquet (idempotent)
│   └── render_cheatsheet.py        # regenerates the cheat sheet
└── data/                           # gitignored, rebuilt by the scripts
    ├── corpus.parquet
    └── embeddings.parquet
```

`helpers.py` is the one shared module every notebook imports. It loads `.env`, builds both clients, and exposes `ensure_index`, `chunks`, `make_toy_movies`, and a couple of small display helpers. Every notebook starts the same way on purpose:

```python
from helpers import load_env, get_pinecone_client, get_openai_client
cfg = load_env()
pc = get_pinecone_client(cfg)
```

## Regenerating the cheat sheet

```bash
python scripts/render_cheatsheet.py
```

Writes both `assets/cheatsheet.svg` and `assets/cheatsheet.png`.

## Cleanup

Each notebook's final cell is an idempotent `delete_index` to free up your free-tier quota. NB1's is active. NB2 and NB3 leave it commented so you can keep poking around after the notebook runs.

## What this module doesn't cover

This is the baseline RAG pattern: straight similarity search followed by a chat completion. The things that move the needle in production live in later modules of this repo:

- **Re-ranking** (module 05)
- **Query rewriting** (module 05)
- **Hybrid and keyword search** (module 03)
- **Evaluation:** faithfulness, context precision, answer relevance (module 04)
- **Graph-structured retrieval** (module 02)
