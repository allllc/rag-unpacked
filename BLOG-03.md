# The baseline you should beat: BM25, head-to-head with vectors

The third module of [rag-unpacked](https://github.com/allllc/rag-unpacked) is the one I almost talked myself out of writing. Vector RAG is the famous version. Graph RAG is the structural alternative. What's left for a third module? The answer turns out to be the thing everybody skips past on the way to embeddings: BM25, the older and simpler keyword retriever that's been the production baseline for two decades.

The honest version of what I built is sharper than "and now keyword search." Vector retrieval is famous; BM25 is older, simpler, faster, and on a surprising number of real workloads strictly *better* than the embedding-based version of the same query. The portfolio value of module 03 turned out to be teaching readers **when not to reach for vectors**, then doing the head-to-head against module 01's existing Pinecone corpus so the comparison was concrete, not abstract.

If you want to skip straight to the code it lives at [03-vectorless-rag](https://github.com/allllc/rag-unpacked/tree/main/03-vectorless-rag). The rest of this writeup is why the code looks the way it does.

## What BM25 actually is

In one diagram:

```
              ┌────────────┐   ┌──────────────────┐   ┌─────────────────────┐
score(doc) =  │    IDF     │ × │ TF saturation    │ × │ length normalization │
              │ rare words │   │ more matches     │   │ shorter docs win    │
              │ count more │   │ count more, but  │   │ ties (length-       │
              │            │   │ with diminishing │   │ adjusted)           │
              │            │   │ returns          │   │                     │
              └────────────┘   └──────────────────┘   └─────────────────────┘
```

That's it. Three statistics about word counts, multiplied together. The implementation in Python is roughly twenty lines of code. The library I used (`bm25s`) is a faster vectorized version of those twenty lines.

What there isn't: a neural network, a learned embedding, a transformer, an attention mechanism, or anything else that's been hot in machine learning in the past decade. BM25 is from 1994 and it still works. The reason it works is the reason most retrieval problems aren't that hard: when the user's query and the relevant document share distinctive vocabulary, finding the document is mostly a counting problem.

## Why `bm25s` and not `rank-bm25`

I considered three libraries:

| Library | Pros | Cons |
|---|---|---|
| `rank-bm25` | Pure Python, ~200 LOC you can read end to end | Slow on real corpora; harder to ship in production |
| `tantivy-py` | Rust-backed, Lucene-style, production-grade | Larger API surface, more to learn before first query |
| `bm25s` | ~10x faster than `rank-bm25`, single light dependency, simple API | Newer, less name recognition |

`rank-bm25` would have been the pure pedagogical choice. `tantivy-py` would have been the pure pragmatic choice. `bm25s` is the middle path that costs almost nothing on either axis. The API is five calls (`tokenize`, `BM25()`, `index`, `tokenize` again for the query, `retrieve`), which is short enough to print on a cheat sheet and small enough to remember after one notebook.

Same kind of decision module 02 made when it picked Kuzu over Neo4j: keep the friction floor as low as possible without sacrificing portability of what the reader learns.

## The three notebooks

### Crawl: [01_introduction.ipynb](https://github.com/allllc/rag-unpacked/blob/main/03-vectorless-rag/01_introduction.ipynb)

A three-document toy corpus inline (cats, dogs, pythons), then a 12-tip Python standard library corpus loaded from a parquet file. The scoring intuition presented as the three-box diagram above, the `bm25s` API in five lines, and two real queries we can hand-verify the ranking on.

The teaching beat I'm proudest of: the second query is `"binary search insert"`, which surfaces tip_03 (`bisect`) with a clean score gap behind it. The second and third results both score zero, meaning they share no non-stopword tokens with the query at all. That's the **IDF reward in action**, made visible. The reader gets to see BM25 doing what it's good at on its second-ever query.

The teaser at the end: same kind of question, swap one word. `"how do I sort a list of dictionaries"` *fails*, because the relevant document says `dicts` not `dictionaries` and BM25 doesn't know they mean the same thing. That's the vocabulary-mismatch problem, and it's the reason module 03 has a NB2 and NB3.

### Walk: [02_bm25_mechanics.ipynb](https://github.com/allllc/rag-unpacked/blob/main/03-vectorless-rag/02_bm25_mechanics.ipynb)

The knobs. Tokenization with stopwords on and off, stemming via PyStemmer (which collapses "sort", "sorted", "sorting", and "sorts" all to one token), the `k1` and `b` parameters with side-by-side rankings.

The honest-defaults-take I landed on: on a corpus this small, `k1` shifts the scores but rarely flips the rankings. `b` barely moves anything because all twelve docs are about the same length (46-67 words). The portfolio voice doesn't oversell tuning. **Defaults are good. When BM25 isn't producing the rankings you want, the problem is almost always upstream: your tokenization, your stopword list, the vocabulary of your corpus vs your queries. Not the hyperparameters.**

The thesis section: `"__slots__"` wins clean (tip_06 with a big score gap). `"make my Python code faster"` *doesn't* find tip_12 ("Speed up loops by avoiding global name lookups"), because the query says "make faster" and the doc says "speed up". Same intent, no shared vocabulary, BM25 misses entirely. That's the synonym trap, and NB3 spends Act III resolving it.

### Run: [03_vectorless_rag.ipynb](https://github.com/allllc/rag-unpacked/blob/main/03-vectorless-rag/03_vectorless_rag.ipynb)

Three acts.

**Act I, the full vectorless RAG loop with `gpt-4o-mini`.** Tokenize, Retrieve, Augment, Generate. Same four-step shape as module 01's vector RAG, with one step swapped. The first demo question (`"how do I cache function results in Python?"`) retrieves tip_09 (`lru_cache`) and the LLM writes a real grounded answer with code. The second question retrieves two documents and the LLM synthesizes across them. The third question is the synonym trap from NB2, retrieved by BM25, then handed to the LLM. The LLM honestly reports it can't answer from the context. The retrieval is the bottleneck, not the LLM.

**Act II, head-to-head against module 01's vector index.** This is the centerpiece of the module. We load module 01's `data/corpus.parquet` (20 paraphrased Pinecone-docs chunks) and run five engineered demo questions through both BM25 and Pinecone vector retrieval. The questions were engineered to make the spread meaningful: three rare-identifier queries where BM25 should win, one paraphrase query where vector should win, one well-worded English query that should be a tie.

What actually happened, when I ran it against the live index:

| Question | BM25 top-1 | Vector top-1 | Agree? |
|---|---|---|---|
| `pool_threads` | Async upsert | Async upsert | ✅ |
| Serverless index | Serverless indexes | Serverless indexes | ✅ |
| `$in` operator | Metadata filtering | Metadata filtering | ✅ |
| "Store text without embedding it myself" | Choosing a distance metric | Integrated inference | ❌ vector wins |
| "What does the metric parameter control" | Embedding dimension | Creating an index | ❌ vector wins |

Three agreements, two vector wins, zero BM25-only wins. **Not the spread I'd planned.** On my hand-engineered "BM25 should win" questions, the embedding model produced the same top-1 BM25 did. The structural BM25 advantage on rare identifiers was real, but it didn't surface as a *disagreement* because there were only 20 candidate docs and the embedding had room to converge.

That's a more honest result than what I'd planned. I rewrote the prose to say exactly that: "on a small corpus with distinctive identifiers, both methods often agree. The point of the head-to-head isn't to count wins; it's to see each method's structural strength and weakness side by side."

**Act III, hybrid via Reciprocal Rank Fusion.** Combine BM25's ranking and vector's ranking with `sum(1/(k + rank))` across both lists, sort, take the top results. `k=60` is the standard pick. Ten lines of code, no score normalization needed (RRF combines ranks, which are comparable across methods, not raw scores).

What I expected: hybrid would pick the right answer on every disagreement. What actually happened: hybrid picked the BM25 answer in both disagreement cases, including the ones where vector was structurally better. Why? Because in both cases, the right doc was vector's #1 but BM25's #5, and the wrong-but-defensible BM25 #1 was vector's #3. RRF doesn't care about which one is *correct*; it sums ranks. The doc that ranks high in *both* lists wins, even when the doc that ranks #1 in just one list is the one the user wanted.

That's the lesson I didn't expect to land. **Hybrid is a robustness move, not a quality move.** It doesn't reliably pick the better single-method answer. It reliably picks an answer that's at least defensible according to both methods. The win is consistency across query shapes (*not falling off a cliff* when one retriever fails), not picking the optimum on every query. To get back to optimum-per-query you layer a cross-encoder reranker on top of the fused list, and that's module 05.

I rewrote the closing thesis to match. The first version said "hybrid usually beats either alone." The honest version says "hybrid is a robustness move, not a quality move." Same claim in the literature, but my five-question demo didn't show the clean win. Saying so makes the module stronger.

## What surprised me

A few specific things from actually building this.

**The hand-engineered head-to-head was almost a tie.** I went in expecting BM25 to crush the rare-identifier queries (`pool_threads`, `$in`) while vector handled the paraphrases. What I got was three agreements on the supposedly BM25-favored questions. The lesson: at small corpus sizes, embedding models often converge on the same answers BM25 produces for keyword-shaped queries. The BM25 advantage shows up more clearly at scale, where vector has more semantically-adjacent candidates to confuse itself with. On 20 docs, the structural difference doesn't bite hard.

**RRF doesn't do what I assumed it does.** I'd internalized "RRF combines rankings to pick the best of both worlds." That's marketing. What RRF actually does is sum reciprocal ranks. It privileges *consistency* over *correctness*. A doc that's at rank 3 in both lists outranks a doc that's at rank 1 in just one. Sometimes that's exactly what you want (robustness). Sometimes it's not (when one retriever is structurally right for the query). The honest take is in the notebook.

**`bm25s` is fast enough to be invisible.** Indexing the 12-tip corpus, running a query, getting the results, all of it is under 10ms on a laptop. I built more elaborate timing code planning to demonstrate "BM25 is fast" and then deleted it because there was nothing to demonstrate. The latency is in the LLM call, not the retrieval. Vector retrieval has the same property over a Pinecone serverless free tier (~50ms per query), so the latency comparison would be a wash on this scale. At a million docs the comparison gets more interesting.

**Defaults really are good.** I spent a couple of hours sweeping `k1` and `b` values trying to find a query where the rankings flipped meaningfully. Couldn't, because at 12 docs with similar lengths the parameters don't have enough corpus variance to shift things. So I wrote it that way: "When to tune k1 or b: rarely." That's a real production lesson, not a cop-out.

## Key takeaways

Five things from actually building this.

**BM25 is free.** No embedding API calls, no GPU, no index storage cost beyond a few KB of token statistics, no embedding model to depreciate when OpenAI deprecates `text-embedding-3-small`. The free-ness is structural, not "free at small scale." If your retrieval problem can be solved by BM25, BM25 will solve it at zero marginal cost.

**The first thing to know before reaching for an embedding model is what BM25 produces on your corpus.** If BM25 is already getting the right answer on 80% of your queries, the value of adding vector retrieval is only the 20% it misses, minus the embedding API costs, minus the operational complexity. That ratio is your business case for vectors, and you can only compute it after you've measured the BM25 baseline. Module 04 (evaluation) is where you actually do that measurement.

**The synonym trap is the central failure mode of BM25 on natural language.** Stemming helps a little, query expansion helps a little more, vector retrieval solves it natively. If your users phrase questions in their own words rather than copy-pasting your documentation's vocabulary, you'll feel this. If they're searching for an error code or an API name, you won't.

**Hybrid is a robustness move, not a quality move.** This was the surprise. RRF gives you a system that doesn't catastrophically fail when one retriever is wrong, in exchange for not always picking what the better retriever would have picked. For most production systems that's the right trade because you can't predict which retriever will win on each query. For high-stakes retrieval (legal, medical, code completion) you want a reranker on top of the fused list, and that's module 05's territory.

**Most retrieval problems aren't ML problems.** They're keyword problems where BM25 wins, vocabulary-mismatch problems where vectors win, and structural problems where graphs win (module 02). The interesting work is figuring out which problem you have. The boring work is building the right baseline. This module is the boring work.

## Where this module doesn't go

Three gaps, all pointer-forward.

**Cross-encoder reranking.** A second-stage model that re-scores the top-N candidates from the fused list using a model that sees query and document together. This is the production fix for "hybrid picks the wrong best-of-each-method doc." Module 05.

**Learned sparse retrieval** (SPLADE and friends). BM25's spirit (sparse term-based scoring) with learned term weights rather than statistical ones. Often beats BM25 on benchmarks while keeping the latency profile. Module 05.

**Evaluation.** How do you actually measure which arm wins on your traffic? Faithfulness, context precision, answer relevance, mean reciprocal rank. Both modules 01 and 02 punted on this. Module 03 makes the question concrete (which BM25-vs-vector wins matter, and how do you tell), but resolves it only by pointing at module 04.

## Dig in

- Repo: [github.com/allllc/rag-unpacked](https://github.com/allllc/rag-unpacked)
- Module 03: [03-vectorless-rag](https://github.com/allllc/rag-unpacked/tree/main/03-vectorless-rag)
- Cheat sheet: [03-vectorless-rag/assets/cheatsheet.png](https://github.com/allllc/rag-unpacked/blob/main/03-vectorless-rag/assets/cheatsheet.png)
- The 12-tip corpus builder: [scripts/build_corpus.py](https://github.com/allllc/rag-unpacked/blob/main/03-vectorless-rag/scripts/build_corpus.py)

If you find a query where my head-to-head story comes out differently, open an issue with the question and your top-1 from each arm. The honest version of this module gets stronger the more independent data points it has. Module 04 (evaluating RAG) is up next.
