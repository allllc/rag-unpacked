# 04. Evaluating RAG

Four notebooks on closing the loop. NB1 hand-rolls the cheap retrieval metrics. NB2 introduces RAGAS for the expensive LLM-judge metrics. NB3 runs both across module 03's three retrievers on module 01's 20-doc Pinecone corpus and produces a six-by-three matrix. NB4 brings the framework to module 02's movie graph with two extra metrics specific to graph RAG.

The portfolio thesis: evaluation is the framework you wrap around modules 01 to 03, not a fourth retriever. The same six metrics that produce the matrix here work on graph RAG (NB4 demonstrates that), and they'll work on whatever module 05 turns out to be.

![RAGAS + hand-rolled Evaluating RAG cheat sheet](./assets/cheatsheet.png)

## The four notebooks

| Notebook | Focus | Uses OpenAI? | Uses Pinecone? | Uses Kuzu? |
|---|---|---|---|---|
| [01. Introduction](./01_introduction.ipynb) | Hand-roll precision@k, recall@k, MRR on a toy example and on the real corpus. Shape-tag breakdown reveals what a single number hides. | No | No | No |
| [02. RAGAS basics](./02_ragas_basics.ipynb) | The four columns of an EvaluationDataset, four metrics, one `evaluate()` call. Stochasticity demo. The cost-vs-precision trade. | Yes | No | No |
| [03. Head-to-head](./03_head_to_head.ipynb) | Vector vs BM25 vs hybrid (RRF) on the 20-doc corpus. Six metrics each. One matrix. Caches generations to disk so reruns are free. | Yes | Optional | No |
| [04. Evaluating graph RAG](./04_evaluating_graph_rag.ipynb) | Cypher-RAG vs Scene-RAG on the movie graph with two hand-rolled metrics: `cypher_hit` and `schema_coverage`. Three failure modes graph RAG has that text RAG doesn't. | Yes | No | Yes (module 02's DB) |

## Why RAGAS

It's the de facto standard for RAG eval. The API is small enough to print on a cheat sheet (four metric classes, one `evaluate()` call). The metrics map cleanly to what the rest of the repo promised (faithfulness, answer relevance, context precision, context recall). And the LLM-as-judge wiring works through a single LangChain wrapper so we can swap judge models without rewriting metric code.

Other options I considered: DeepEval (more pytest-style, larger API), TruLens (provider-agnostic, more production-shaped), LangChain's built-in evaluation module (less mature). RAGAS won on API minimalism. If you build a production eval harness, any of these is fine; the framework in this module is library-agnostic.

## Setup

```bash
cd 04-evaluating-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your OpenAI key. (NB1 doesn't need it; NB2/3/4 do.)
# Optional: also paste a Pinecone key if you want NB3's head-to-head
# to run all three arms (vector + BM25 + hybrid).
```

Get the OpenAI key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys). The Pinecone key is the same one modules 01 and 03 use; if you've already run module 01, you already have it.

### Build the eval sets

```bash
python scripts/build_eval_set.py            # 20 questions on Pinecone docs (~8 KB)
python scripts/build_eval_set_movies.py     # 12 questions on the movie graph (~6 KB)
```

The movies builder verifies ground-truth scene numbers and movie titles against module 02's Kuzu DB at build time. If the DB isn't there, it prints a clear pointer at module 02's build script and exits.

### Run

```bash
jupyter lab
```

## What it costs

| Notebook | Approx cost | Why |
|---|---|---|
| NB1 | $0.00 | No OpenAI calls. |
| NB2 | ~$0.03 | 20 RAG generations plus 4 metrics x 20 judge calls. |
| NB3 | ~$0.07 first run, ~$0.05 cached | 60 RAG generations plus 3 retrievers x 60 judge calls. Generations cached to `data/generation_cache.pkl`. |
| NB4 | ~$0.03 | 24 RAG generations plus 2 retrievers x ~36 judge calls. |

Whole module: under 20 cents worst case, under 10 cents typical, under 5 cents if NB3 runs from cache.

## Layout

```
04-evaluating-rag/
├── 01_introduction.ipynb
├── 02_ragas_basics.ipynb
├── 03_head_to_head.ipynb
├── 04_evaluating_graph_rag.ipynb
├── helpers.py                       # Config + load_env + 6 helper functions
├── .env.example
├── requirements.txt
├── assets/
│   ├── cheatsheet.png               # rendered above
│   └── cheatsheet.svg               # crisp, searchable source
├── scripts/
│   ├── build_eval_set.py            # 20-question Pinecone-docs eval set
│   ├── build_eval_set_movies.py     # 12-question movie-graph eval set
│   └── render_cheatsheet.py         # regenerates the cheat sheet
└── data/                            # gitignored, rebuilt by the scripts
    ├── golden.parquet               # 20 (question, answer, relevant_doc_ids)
    ├── golden_movies.parquet        # 12 (question, answer, cypher_targets, scene_keys)
    ├── retrieval_cache.pkl          # NB3 retrieval cache
    ├── generation_cache.pkl         # NB3 generation cache
    ├── generation_scores.pkl        # NB3 RAGAS scores cache
    ├── graph_rag_runs.pkl           # NB4 cypher + scene RAG runs
    └── graph_rag_scores.pkl         # NB4 RAGAS scores cache
```

`helpers.py` mirrors module 03's shape: it loads `.env`, builds the OpenAI client, optionally builds a Pinecone client (or returns `None` if the key is absent), and opens a connection to module 02's Kuzu DB. It also exports the three hand-rolled retrieval metrics, the BM25 and RRF helpers (copied verbatim from module 03 so this module stays self-contained), and the two RAGAS wrappers (`get_evaluator_llm`, `get_evaluator_embeddings`). The shared opening for NB2 and NB3:

```python
from helpers import (
    load_env, get_openai_client, get_pinecone_client,
    get_evaluator_llm, get_evaluator_embeddings,
    build_bm25_index, retrieve, rrf_combine,
    precision_at_k, recall_at_k, mean_reciprocal_rank,
)
cfg = load_env()
```

## Regenerating the cheat sheet

```bash
python scripts/render_cheatsheet.py
```

Writes both `assets/cheatsheet.svg` and `assets/cheatsheet.png`.

## Cleanup

No remote state. Delete `data/golden.parquet`, `data/golden_movies.parquet`, and the four `.pkl` caches for a fresh start. The build scripts will recreate the parquet files; the caches will repopulate on the next NB3 or NB4 run.

## A note on cross-module data

NB3 loads `01-intro-to-rag/data/corpus.parquet` directly via a relative path and queries module 01's existing `rag-unpacked-intro` Pinecone index. NB4 opens module 02's `02-graph-rag/data/movies.kuzu` directly. Both have graceful friction-floor messages: if the data is missing, the notebook tells you exactly which prior-module build script to run.

If `PINECONE_API_KEY` isn't set, NB3 gracefully skips the vector and hybrid arms and prints a clear note. The BM25 arm still runs.

## What this module doesn't cover

This is the canonical RAG evaluation pattern at portfolio scale (20 to 30 hand-authored questions, two LLM-judge metrics, one matrix). Real production eval layers in more:

- **Bigger eval sets** (200 to 2000 questions), often LLM-bootstrapped from real production traffic and then human-reviewed. Module 05.
- **Human calibration of the judge model.** If your judge says "faithfulness = 0.7," what's that mean in absolute terms? You calibrate by having humans score 50 to 100 examples and comparing.
- **Drift monitoring on live traffic.** The eval set doesn't drift; production does. Run RAGAS continuously on a sample of real questions.
- **Online A/B and bandit testing.** The matrix says vector wins on this corpus; production tells you whether users prefer the vector answer to the BM25 one.
- **Golden-set version control.** Tag eval set releases, diff scores across versions, gate PRs on score regressions.
- **Better graph-RAG metrics.** `cypher_hit` and `schema_coverage` are starting points. Production graph RAG eval wants Cypher syntax checks, semantic query-equivalence judging, recall over relationships (not just nodes), and entity-linking precision.
