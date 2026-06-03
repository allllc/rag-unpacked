# 02. Graph RAG with Kuzu

Four notebooks that go from "what's a node" to a small Q&A bot that answers natural-language questions by retrieving from a graph that mixes structured facts (people, movies, studios) with the actual text of eleven movie scripts.

**Crawl** (NB1) sets up a tiny three-node graph in Kuzu, declares a schema, runs one `MATCH` query. No LLM, no LangChain. **Walk** (NB2) loads the full movie graph (30 people, 11 movies, 8 studios, 54 edges) and walks through the Cypher you actually write for graph RAG: filters, multi-hop traversals, aggregations, `OPTIONAL MATCH`, parameterized queries. **Run** (NB3) wires an LLM in. We build the natural-language-to-Cypher loop two ways: by hand first, then with LangChain's `KuzuQAChain`, so you can see exactly what the framework wraps. **Real text** (NB4) attaches the full text of eleven movie scripts to the graph as `Script` nodes, splits each script on `INT./EXT.` slug lines into 1,800 `Scene` nodes, and shows the production pattern: graph picks the document *and* the chunk, the LLM reads only the survivors.

Each notebook picks up where the last one left off.

![Kuzu + LangChain Graph RAG cheat sheet](./assets/cheatsheet.png)

## The four notebooks

| Notebook | Focus | Uses OpenAI? |
|---|---|---|
| [01. Introduction](./01_introduction.ipynb) | What a graph is. Three people, three movies, one MATCH query, two-hop co-actor traversal. | No |
| [02. Querying the Graph](./02_querying_the_graph.ipynb) | The full movie graph. WHERE filters, two and three-hop traversals, count and collect, OPTIONAL MATCH, parameterized queries. | No |
| [03. Graph RAG with LangChain](./03_graph_rag_with_langchain.ipynb) | The full RAG loop, twice: by hand (four explicit steps) and with `KuzuQAChain`. Plus the multi-hop demo question vector search can't answer. | Yes |
| [04. Graph RAG with Real Text](./04_graph_rag_with_text.ipynb) | Attach real movie scripts to graph nodes two ways: as whole `Script` documents and as chunked `Scene` nodes split on screenplay slug lines. Watch whole-document retrieval fail on mid-script questions, then fix it with scene-level retrieval. | Yes |

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
# Open .env and paste your OpenAI key. (Only NB3 and NB4 use it.)
```

Get the OpenAI key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Kuzu doesn't need any keys.

### Build the graph

```bash
python scripts/build_corpus.py
```

This writes the movie graph to `data/movies.kuzu`. Idempotent.

### Fetch movie scripts (one time, for NB4 only)

```bash
python scripts/download_scripts.py
python scripts/build_corpus.py --force
```

The downloader pulls eleven movie scripts (~2 MB total) from [IMSDb](https://imsdb.com) and caches them locally to `data/scripts/*.txt`. Then `--force` rebuilds the Kuzu graph so the `Script` nodes get populated. Scripts are gitignored: each cloner fetches them themselves.

If you only want to do NB1–NB3, you can skip both download steps. NB4 will tell you what's missing.

### Run

```bash
jupyter lab
```

## What it costs

NB1 and NB2 don't call OpenAI at all, only Kuzu, which is local and free.

NB3 runs the four-step RAG loop a few times with `gpt-4o-mini`. Each question is two chat completions: one to write Cypher, one to write the answer. A full run is well under one cent.

NB4 has two cost profiles. Act II sends ~40K characters (~10K tokens) of whole-script context per question. Act III sends ~5K-20K characters of *just the matching scenes*, roughly 5x cheaper per question and a much higher signal-to-noise ratio for the LLM. A full NB4 run is well under one cent.

## Layout

```
02-graph-rag/
├── 01_introduction.ipynb
├── 02_querying_the_graph.ipynb
├── 03_graph_rag_with_langchain.ipynb
├── 04_graph_rag_with_text.ipynb     # new in this module's NB4
├── helpers.py                       # shared config and utilities
├── .env.example
├── requirements.txt
├── assets/
│   ├── cheatsheet.png               # rendered above
│   └── cheatsheet.svg               # crisp, searchable source
├── scripts/
│   ├── build_corpus.py              # writes data/movies.kuzu (idempotent)
│   ├── download_scripts.py          # fetches IMSDb scripts (one-time, NB4)
│   └── render_cheatsheet.py         # regenerates the cheat sheet
└── data/                            # gitignored, rebuilt by the scripts
    ├── movies.kuzu                  # full movie graph (NB2-NB4)
    │                                # incl. 11 Script + 1809 Scene nodes
    ├── intro_demo.kuzu              # NB1's throwaway demo graph
    └── scripts/                     # cached IMSDb scripts (NB4)
        └── *.txt
```

`helpers.py` mirrors module 01's helper: it loads `.env`, builds both clients, and exposes `get_kuzu_conn`. Every notebook starts the same way on purpose:

```python
from helpers import load_env, get_openai_client, get_kuzu_conn
cfg = load_env()
conn = get_kuzu_conn()
client = get_openai_client(cfg)  # NB3 and NB4 only
```

## Regenerating the cheat sheet

```bash
python scripts/render_cheatsheet.py
```

Writes both `assets/cheatsheet.svg` and `assets/cheatsheet.png`.

## Cleanup

Kuzu's database is a single file on disk (plus a sibling `.wal` write-ahead log). Delete both whenever you want a fresh start:

```bash
rm data/movies.kuzu data/movies.kuzu.wal
rm data/intro_demo.kuzu data/intro_demo.kuzu.wal
```

The build script will recreate them on the next run. The script downloads stay cached in `data/scripts/` so you don't re-fetch them from IMSDb.

## A note on the scripts

The eleven movie scripts come from [IMSDb](https://imsdb.com), which has hosted public-domain or fan-archived shooting drafts for two decades. We're using them as a teaching corpus and the local cache is gitignored: the repo never redistributes them. If you build something on top of this for anything beyond learning, do your own clearance.

## What this module doesn't cover

This module is the canonical graph RAG pattern: schema, Cypher, LLM-as-translator, chain-the-result, real text on nodes, and structural chunking of that text into scenes. The remaining production layers live in later modules:

- **Entity linking** (fuzzy match user input to graph nodes before generating Cypher)
- **Vector search inside chunked text** (NB4 Act III filters scenes with `CONTAINS`; module 05 swaps that for cosine similarity over scene embeddings)
- **Reranking** of graph results (module 05)
- **Hybrid retrieval** (graph + vector together, the full production answer) (module 05)
- **Evaluation** (faithfulness, schema coverage, Cypher correctness) (module 04)
- **Production deployment** (concurrent writes, replication, monitoring), for which you'd graduate to Neo4j or another server-backed store
