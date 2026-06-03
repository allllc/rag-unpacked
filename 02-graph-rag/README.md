# 02. Graph RAG with Kuzu

Three notebooks that go from "what's a node" to a small Q&A bot that answers natural-language questions about a movie graph by translating them into Cypher.

**Crawl** (NB1) sets up a tiny three-node graph in Kuzu, declares a schema, runs one `MATCH` query. No LLM, no LangChain. **Walk** (NB2) loads the full movie graph (14 people, 11 movies, 5 studios) and walks through the Cypher you actually write for graph RAG: filters, multi-hop traversals, aggregations, `OPTIONAL MATCH`, parameterized queries. **Run** (NB3) wires an LLM in. We build the natural-language-to-Cypher loop two ways: by hand first, then with LangChain's `KuzuQAChain`, so you can see exactly what the framework wraps.

Each notebook picks up where the last one left off.

![Kuzu + LangChain Graph RAG cheat sheet](./assets/cheatsheet.png)

## The three notebooks

| Notebook | Focus | Uses OpenAI? |
|---|---|---|
| [01. Introduction](./01_introduction.ipynb) | What a graph is. Three people, three movies, one MATCH query, two-hop co-actor traversal. | No |
| [02. Querying the Graph](./02_querying_the_graph.ipynb) | The full movie graph. WHERE filters, two and three-hop traversals, count and collect, OPTIONAL MATCH, parameterized queries. | No |
| [03. Graph RAG with LangChain](./03_graph_rag_with_langchain.ipynb) | The full RAG loop, twice: by hand (four explicit steps) and with `KuzuQAChain`. Plus the multi-hop demo question vector search can't answer. | Yes |

## Why Kuzu and not Neo4j

Kuzu is an embedded graph database. `pip install kuzu` and you're done. No Docker, no cloud signup, no separate server process, the database is a single file on disk. It uses the same openCypher query language as Neo4j, so every query you learn here ports unchanged.

For production with concurrent writers or graphs bigger than a single box, Neo4j is the more battle-tested choice. NB3 ends with a short appendix showing the three-line swap. The notebooks default to Kuzu so anyone cloning the repo can run them in under a minute.

## Setup

```bash
cd 02-graph-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your OpenAI key. (Only NB3 actually uses it.)
```

Get the OpenAI key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Kuzu doesn't need any keys.

Then:

```bash
python scripts/build_corpus.py
jupyter lab
```

`build_corpus.py` writes the movie graph to `data/movies.kuzu`. It's idempotent. NB1 makes its own throwaway database at `data/intro_demo.kuzu` so it doesn't conflict.

## What it costs

NB1 and NB2 don't call OpenAI at all, only Kuzu, which is local and free.

NB3 runs the four-step RAG loop a few times (two demo questions in Act I, one in Act II, one multi-hop question in Act III) with `gpt-4o-mini`. Each question is two chat completions: one to write the Cypher, one to write the answer. End-to-end a full run is well under one cent.

## Layout

```
02-graph-rag/
├── 01_introduction.ipynb
├── 02_querying_the_graph.ipynb
├── 03_graph_rag_with_langchain.ipynb
├── helpers.py                      # shared config and utilities
├── .env.example
├── requirements.txt
├── assets/
│   ├── cheatsheet.png              # rendered above
│   └── cheatsheet.svg              # crisp, searchable source
├── scripts/
│   ├── build_corpus.py             # writes data/movies.kuzu (idempotent)
│   └── render_cheatsheet.py        # regenerates the cheat sheet
└── data/                           # gitignored, rebuilt by the scripts
    ├── movies.kuzu                 # full movie graph (used by NB2 and NB3)
    └── intro_demo.kuzu             # NB1's throwaway demo graph
```

`helpers.py` mirrors module 01's helper: it loads `.env`, builds both clients, and exposes `get_kuzu_conn`. Every notebook starts the same way on purpose:

```python
from helpers import load_env, get_openai_client, get_kuzu_conn
cfg = load_env()
conn = get_kuzu_conn()
client = get_openai_client(cfg)  # NB3 only
```

## Regenerating the cheat sheet

```bash
python scripts/render_cheatsheet.py
```

Writes both `assets/cheatsheet.svg` and `assets/cheatsheet.png`.

## Cleanup

Kuzu's database is a single file on disk. Delete `data/movies.kuzu` and `data/intro_demo.kuzu` whenever you want a fresh start. The build script will recreate them on the next run.

## What this module doesn't cover

This module is the canonical graph RAG pattern: schema, Cypher, LLM-as-translator, chain-the-result. Real production graph RAG layers in more:

- **Entity linking** (fuzzy match user input to graph nodes before generating Cypher)
- **Reranking** of graph results (module 05)
- **Hybrid retrieval** (combining vector and graph) (module 05)
- **Evaluation** (faithfulness, schema coverage, Cypher correctness) (module 04)
- **Production deployment** (concurrent writes, replication, monitoring), for which you'd graduate to Neo4j or another server-backed store
