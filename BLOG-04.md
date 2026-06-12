# The matrix that closes the loop: a 20-question RAG eval set, four notebooks, two surprises

The fourth module of [rag-unpacked](https://github.com/allllc/rag-unpacked) is the one each of the first three pointed at and couldn't deliver. Module 01 shipped vector RAG and the only quality signal was "the answer looks right." Module 02 shipped graph RAG and got the same signal. Module 03 declared the BM25-vs-vector winner on five hand-picked questions and was honest about the spread being tighter than the plan predicted. Each module saw one slice; none could measure the others.

Module 04 is the loop that closes those open questions. The portfolio thesis is single-sentence: **evaluation is the framework you wrap around modules 01 to 03, not a fourth retriever.** The same six metrics that produce the head-to-head matrix in NB3 work on graph RAG too (NB4 demonstrates that, with two extra hand-rolled metrics specific to graphs), and they'll work on whatever module 05 turns out to be.

If you want to skip straight to the code it lives at [04-evaluating-rag](https://github.com/allllc/rag-unpacked/tree/main/04-evaluating-rag). The rest of this writeup is why the matrix matters more than any single number in it.

## What evaluation actually is

The frame that matters most: **retrieval quality and generation quality are separable.**

Retrieval quality asks "did the right doc come back?" You can answer that with set arithmetic and ranks. No LLM, no embeddings, no money. Precision@k, recall@k, MRR. Cheap enough to run on every PR.

Generation quality asks "did the LLM use the doc faithfully and answer the question?" You can't answer that without another LLM as a judge. Faithfulness, answer relevance, context precision, context recall. Slow, stochastic, costs a couple cents per pass. Right cadence: nightly, or per release.

You want both. The point isn't that one is better, it's that they measure different things. Retrieval metrics catch ranking regressions. Generation metrics catch prompt regressions, hallucinations, and refusal failures.

## Why RAGAS and not DeepEval, TruLens, or LangChain's eval module

Brief library comparison, same energy as BLOG-03's bm25s rundown:

| Library | Pros | Cons |
|---|---|---|
| **RAGAS** | Small API. Four metric classes. One `evaluate()` call. Metrics map cleanly to what the rest of the repo promised (faithfulness, answer relevance, context precision, context recall). Library-agnostic at the framework layer. | Stochastic scores. Embedder must be wired up explicitly for `ResponseRelevancy`. |
| **DeepEval** | Pytest-style integration. Production-shaped. Larger ecosystem. | Bigger API surface. More opinionated. Less notebook-friendly. |
| **TruLens** | Provider-agnostic. Feedback functions are nice for production. | More production-shaped than portfolio-shaped. |
| **LangChain's `evaluation` module** | Bundled with LangChain. | Less mature than the dedicated libraries. |

RAGAS won on minimalism. If you build a production eval harness, any of these is fine; the framework in module 04 is library-agnostic.

## The four notebooks

### Crawl: [01_introduction.ipynb](https://github.com/allllc/rag-unpacked/blob/main/04-evaluating-rag/01_introduction.ipynb)

Hand-roll precision@k, recall@k, and MRR on a five-question, five-doc toy example so the reader can verify by hand, then run them on the real 20-doc Pinecone corpus from module 01 with the 20-question golden eval set this module ships.

The teaching beat I'm proudest of: the eval set is tagged by **question shape** (`keyword`, `paraphrase`, `synthesis`, `adversarial`), and the metrics are broken down by shape. The shape breakdown is the cheapest analytic win available. A single overall p@1 of 0.45 hides four very different stories. BM25's keyword-shape p@1 is 0.5, not 1.0 like I'd assumed. BM25's recall@5 on adversarial questions reads as 1.0 because the metric returns 1.0 when the relevant set is empty (nothing to recall). Each of these is a footnote that becomes interesting once you tag the eval set.

### Walk: [02_ragas_basics.ipynb](https://github.com/allllc/rag-unpacked/blob/main/04-evaluating-rag/02_ragas_basics.ipynb)

The four RAGAS columns (`user_input`, `retrieved_contexts`, `response`, `reference`), the four metrics, one `evaluate()` call. Stochasticity demo: re-run faithfulness on the same 20 rows and diff the scores. Mean absolute delta in the run that's checked in: 0.06. Max delta on a single row: 1.0 (one question flipped from one extreme to the other).

That "0.06 mean, 1.0 max" pair is the lesson. Most rows agree across runs. A few wildly disagree. The production fix is to average three runs (cuts variance, triples cost) or use a stronger judge model. Neither is free, both work.

### Run: [03_head_to_head.ipynb](https://github.com/allllc/rag-unpacked/blob/main/04-evaluating-rag/03_head_to_head.ipynb)

The portfolio centerpiece. Three retrievers (BM25, Pinecone vector, hybrid via RRF), same 20 questions, all six metrics. One six-by-three matrix at the end.

What actually happened, when I ran it against the live index:

|                       | BM25 | vector | hybrid |
|---|---|---|---|
| **p@1** | 0.45 | **0.75** | 0.65 |
| **r@5** | 0.89 | **0.99** | 0.94 |
| **MRR** | 0.74 | **0.96** | 0.91 |
| **faithfulness** | 0.71 | 0.69 | **0.78** |
| **answer_relevancy** | 0.54 | 0.53 | **0.58** |
| **context_precision** | 0.63 | **0.84** | 0.83 |

Bold marks the per-row winner. Two surprises:

**Vector sweeps the cheap retrieval row.** That includes the keyword shape, where I'd expected BM25 to dominate. On this 20-doc corpus, `text-embedding-3-small` is good enough that it preserves distinctive tokens (so it wins on keywords too) *and* catches the paraphrases BM25 misses. BLOG-03's "BM25 is the baseline you should beat" framing is correct in spirit; module 04 quantifies by how much. A lot.

**Hybrid wins the end-to-end generation metrics.** That's the second surprise. Vector retrieved better but the LLM-judged generation quality is highest when you fuse BM25 and vector with RRF. The mechanism worth thinking about: hybrid surfaces a slightly different top-3 (BM25's diversity plus vector's accuracy), and that diversity gives the LLM a richer context to ground the answer in. Vector's top-3 are slightly more clustered (they're cosine-close to the query *and* to each other), which gives the LLM less to work with.

The honest takeaway from those two surprises together: **if you're choosing what to ship, you can't decide from retrieval metrics alone.** Vector might have the best p@1, but hybrid produces more faithful, more relevant answers. The matrix exists to surface that trade.

### Bonus: [04_evaluating_graph_rag.ipynb](https://github.com/allllc/rag-unpacked/blob/main/04-evaluating-rag/04_evaluating_graph_rag.ipynb)

Same framework, different corpus, sharper failure modes. Two graph-RAG variants from module 02 (Cypher-RAG and Scene-RAG) run against a 12-question movie eval set, plus two hand-rolled metrics RAGAS doesn't ship: `cypher_hit_at_k` (did the generated Cypher actually return any row containing the ground-truth target?) and `schema_coverage` (did the Cypher use the schema tokens it should have?).

Three failure modes graph RAG has that text RAG doesn't:

1. **Hallucinated Cypher that runs but returns the wrong row.** mq12 in the live run: Scene-RAG's query for "Wachowski film mentions Zion" returned 12 rows across multiple movies, and the LLM picked "The Matrix" instead of the ground-truth "The Matrix Reloaded." Faithfulness scored it OK because the claim *is* supported by the returned rows. Only a reference-comparing metric catches this.
2. **Hallucinated Cypher that doesn't compile.** mq08 (Inception totem) via Cypher-RAG: the LLM generated invalid Cypher, the parser raised, the wrapper caught it, the answer became "Cypher failed: ..." and faithfulness scored it accordingly. Easier to debug than failure mode 1 because it's loud.
3. **Schema-correct Cypher that misses the right relationship.** mq11 (Nolan film with Polaroid) via Cypher-RAG queried for a Nolan film with "Polaroid" in the title instead of going through `Movie -> Scene -> body`. `schema_coverage` catches this because the question's expected schema tokens include `HAS_SCENE` and `body`, which the Cypher didn't use.

The honest closing: **graph RAG eval needs more than RAGAS, and it needs more than `cypher_hit` and `schema_coverage` too.** You want answer-equivalence judging, semantic Cypher comparison, and a metric for "the query reflected the right traversal." This module shows the framework; production graph RAG eval extends it.

## What surprised me

Three things, two of them already mentioned in the matrix section but worth restating because they're the strongest takeaways from running the whole thing:

**Vector dominates on a 20-doc corpus more than my plan predicted.** I'd expected BM25 to win the keyword shape clean (`pool_threads`, `$in`, the rare-identifier questions). It didn't. Vector won. On a 20-doc corpus, `text-embedding-3-small` is just good enough that it preserves distinctive tokens, and BM25 doesn't have any corpus-wide token statistics to lean on for the IDF reward (20 docs is too small to make the IDF reliable). The interesting question this opens: at what corpus size does BM25 start winning the keyword shape? My guess: 1000s to 10000s. The interesting follow-up: at what corpus size does *hybrid* start winning every row? Same range, I'd bet. That's a module 05 question.

**Hybrid wins the generation metrics by surfacing more diverse context.** I'd carried in the BLOG-03 framing that hybrid is "a robustness move, not a quality move." That's not what this matrix says. Hybrid produced more faithful and more relevant answers than either single arm. The mechanism I half-suspect: vector's top-3 are clustered (all close to the query embedding *and* to each other), so the LLM sees three views of the same thing. Hybrid's top-3 are more diverse (BM25's keyword match plus vector's semantic match), so the LLM sees three angles on the answer. Whether that holds on a 200-doc corpus is the kind of question this module's framework lets you answer instead of guess.

**The stochasticity tax is real but small.** Mean absolute delta on a faithfulness rerun: 0.06. Max delta: 1.0. The mean is reassuring (the metric is mostly stable). The max is sobering (one question can flip completely). The production fix is to average three runs, but that triples the cost, and for a 20-question portfolio eval set the right call is to know about it and ship anyway. For a 200-question eval set you'd want the averaging.

## Five takeaways

1. **Cheap and expensive metrics are different jobs, not redundant.** NB1's set arithmetic runs on every PR. NB2's LLM judge runs nightly. The split itself is the lesson.
2. **Shape-tagging your eval set is the cheapest win available.** A single mean precision number hides four very different stories. Tag your questions by shape, break the metrics down by shape, every time.
3. **You can't decide which retriever to ship from retrieval metrics alone.** The vector arm had the best retrieval, the hybrid arm had the best generation. Different metrics, different winners. The matrix surfaces the trade.
4. **Graph RAG eval needs metrics RAGAS doesn't ship.** `cypher_hit` and `schema_coverage` are 15 lines each. They catch failure modes RAGAS can't see (schema-incorrect Cypher, queries that run but return the wrong row). The framework still applies; only the metric set grows.
5. **A 20-question eval set is enough to ship behind, not enough to publish about.** The production version looks the same with 200 to 2000 questions; the shape is identical. What changes at scale is how you build the eval set (LLM-bootstrap plus human review, drift monitoring on production traffic, A/B testing) not what you measure.

## Where this doesn't go

Module 05 picks up reranking and query rewriting; eval is the meta-layer that lives under all of it. The gaps this module doesn't address:

- **Production-scale eval sets** (200 to 2000 questions), often LLM-bootstrapped from real production traffic and then human-reviewed.
- **Human calibration of the judge model.** A faithfulness score of 0.7 from gpt-4o-mini doesn't mean the same thing as 0.7 from gpt-4o. You calibrate by having humans score 50 to 100 examples and comparing.
- **Drift monitoring on live traffic.** The eval set doesn't drift; production does. Run RAGAS continuously on a sample of real questions.
- **Online A/B and bandit testing.** The matrix tells you vector won on the eval set; production tells you whether users prefer the vector answer to the BM25 one.
- **Golden-set version control.** Tag eval set releases, diff scores across versions, gate PRs on score regressions.

## Dig in

Code: [github.com/allllc/rag-unpacked/tree/main/04-evaluating-rag](https://github.com/allllc/rag-unpacked/tree/main/04-evaluating-rag)

Notebooks, helpers, eval-set builders, and the cheat sheet renderer are all there. The repo's top-level [README](https://github.com/allllc/rag-unpacked) embeds the cheat sheet and links each module's standalone README. Modules 01 ([Pinecone vector RAG](https://github.com/allllc/rag-unpacked/tree/main/01-intro-to-rag)), 02 ([Kuzu graph RAG](https://github.com/allllc/rag-unpacked/tree/main/02-graph-rag)), 03 ([BM25 + hybrid](https://github.com/allllc/rag-unpacked/tree/main/03-vectorless-rag)), and now 04 are all shipped. Module 05 (advanced patterns: reranking, query rewriting, learned sparse retrieval) is next.
