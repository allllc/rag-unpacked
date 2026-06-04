# When the graph is the index, not the answer

Module 02 of [rag-unpacked](https://github.com/allllc/rag-unpacked) is the graph RAG module. It started as a question: if vector RAG is "find documents that look like your question," what does it look like when the answer depends on *connections* instead of similarity? You can't embed your way out of "which directors have worked with actors who appeared in Warner Bros films." There's no single document that says that. The fact is a path through three different ones.

The honest version of what I built across four notebooks is more nuanced than that, though. The graph isn't actually the answer in production graph RAG. It's the *index* over the answer. The most important arc in this module is the shift from NB3 ("Cypher returns rows of structured data") to NB4 ("Cypher returns the *text chunks* the LLM needs to read"). The graph picks the document, then picks the passage. The LLM only ever sees what it actually needs.

If you want to skip straight to the code it lives at [02-graph-rag](https://github.com/allllc/rag-unpacked/tree/main/02-graph-rag). The rest of this writeup is why the code looks the way it does.

## Why Kuzu instead of Neo4j

The module 01 blog had a section on how secret-hygiene is the real day-one friction in vector RAG. Module 02's equivalent friction is the *graph database itself*. Neo4j is the industry standard. It also wants Docker, a JVM, a server process, a username and password, and (if you're using AuraDB Free) a cloud sign-up before you can run your first query. For a portfolio piece that someone might clone on a Tuesday night, that's a steep wall.

[Kuzu](https://kuzudb.com) is `pip install kuzu` and you're done. Embedded, in-process, the database is one file on disk. No server. It speaks openCypher, so every query I wrote against Kuzu ports to Neo4j unchanged. For production with concurrent writers or graphs that don't fit on a single box, Neo4j is still the more battle-tested choice. NB3 ends with a four-line appendix showing the import swap. But for *learning* a graph database, "I have a file" is a much better starting point than "I have a cloud instance."

That choice rippled. Because Kuzu doesn't need a server, the entire module runs offline against a single 25 MB file. Because that file is gitignored, the seed corpus is rebuilt by a script on every clone, which forced me to write a real corpus builder instead of checking in a binary. Because the build is fast, I could iterate on the schema multiple times without operational overhead.

## The graph itself

The seed is eleven movies, thirty people, eight studios, fifty-four relationships. I picked the films so the multi-hop queries had real answers worth finding (Carrie-Anne Moss bridges The Wachowskis and Christopher Nolan via Memento; Leonardo DiCaprio bridges Tarantino and Nolan via Django Unchained and Inception). It's small enough to print, big enough that two and three-hop traversals are non-trivial.

The schema deliberately gets richer over the course of the module. NB1 introduces `Person`, `Movie`, and `ACTED_IN`. NB2 adds `Studio`, `DIRECTED`, and `PRODUCED_BY`. NB4 introduces `Script` (the whole text of each movie's screenplay, attached via `HAS_SCRIPT`) and `Scene` (the same screenplay chunked on `INT./EXT.` slug lines, attached via `HAS_SCENE`). 1,809 Scene nodes total, ranging from 45 in Reservoir Dogs to 285 in Inception.

The scripts themselves come from [IMSDb](https://imsdb.com), the public archive that has hosted shooting drafts for two decades. A [download script](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/scripts/download_scripts.py) fetches them on first run (~2 MB total) and caches them to a gitignored folder. The corpus builder chunks the cached files on screenplay slug lines and loads both Script and Scene nodes into the graph. None of the script text is committed to the repo.

## The four notebooks

### Crawl: [01_introduction.ipynb](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/01_introduction.ipynb)

Three hand-written nodes (Keanu Reeves, Carrie-Anne Moss, Laurence Fishburne), three movies (Matrix, Matrix Reloaded, John Wick), six `ACTED_IN` edges, one schema declaration, one `MATCH` query. The whole notebook is "what is a graph, exactly," using the smallest possible Kuzu surface. No LLM. No LangChain. Just openCypher.

There's one teaching moment at the end I'd recommend not glossing over: the two-hop co-actor query without a `WHERE` clause returns Keanu Reeves as his own co-actor (three times, once for each film he's in). Same shape of gotcha as a self-join in SQL. NB2 fixes it with `WHERE other.name <> 'Keanu Reeves'`. NB1's job is just to make you see the gotcha.

### Walk: [02_querying_the_graph.ipynb](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/02_querying_the_graph.ipynb)

The toys are gone. The full movie graph loads from disk and the notebook walks through everything you write a lot of in graph RAG: `WHERE` filters (numeric ranges, string CONTAINS), multi-hop traversals (two-hop co-actors, three-hop director-via-films), aggregations (`count`, `collect`), `OPTIONAL MATCH` (graph's `LEFT JOIN`), parameterized queries.

Still no LLM. The point of NB2 is to get fluent in Cypher *before* an LLM has to write it for you. If you can't read the Cypher, you can't debug what the LLM is generating in NB3 and NB4.

The "two Kuzu gotchas" cell at the end is the kind of touch I tried to make more honest in this module than in module 01: `ORDER BY` references the projection alias, not the pattern variable (the error message is misleadingly "Variable not in scope"). `cast` is a reserved word in Kuzu's parser (it parses `CAST(x AS TYPE)` for type coercion). Both are small papercuts that look like much bigger bugs the first time you hit them.

### Run: [03_graph_rag_with_langchain.ipynb](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/03_graph_rag_with_langchain.ipynb)

The portfolio centerpiece, structurally speaking. The full RAG loop, twice.

First by hand: four explicit steps. Take a natural-language question. Prompt the LLM with the schema and ask for Cypher. Run the Cypher. Send the question and the result rows back to the LLM, ask for a sentence. Every prompt is visible in the cell. Every intermediate output is printed. Forty lines of code, two LLM calls per question.

Then the same thing through LangChain's `KuzuQAChain.from_llm(...)`. Five lines. The result is identical. The chain wraps the four-step loop and saves you the orchestration. What it hides is the prompts: the [default templates](https://github.com/kuzudb/langchain-kuzu/blob/main/libs/kuzu/langchain_kuzu/chains/graph_qa/prompts.py) live in `langchain_kuzu.chains.graph_qa.prompts` and you'd have to dig into the source the first time the LLM writes wrong Cypher.

This is module 02's opinionated take, and it's not "don't use LangChain." It's "build the manual loop first so you can see the seams, then use a chain if you want to." The order matters because the chain is an abstraction over something that's simple enough to write yourself in forty lines. If you've written those forty lines, you understand what the chain costs you (visibility into prompts, two LLM calls per question, customization friction) and what it buys you (no orchestration code to maintain).

Act III is the genuinely interesting question. "Which directors have worked with actors who appeared in films produced by Warner Bros?" Vector RAG can't answer that. There's no single document that says it. The graph RAG version is one Cypher query that returns twelve rows with non-obvious answers: Bong Joon-ho through Robert Pattinson (Tenet → Mickey 17), Tarantino through Leonardo DiCaprio (Inception → Django Unchained), Stahelski through Keanu Reeves (Matrix → John Wick). The path through the graph is the answer.

### Real text: [04_graph_rag_with_text.ipynb](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/04_graph_rag_with_text.ipynb)

This is where I'd start if I read the module front-to-back today. NB3 is a fine demonstration of structured graph RAG, but production RAG is mostly about text. The answer the user wants is rarely "Christopher Nolan." It's "*what did Jules say about the path of the righteous man.*"

NB4 makes the shift in three acts.

**Act I** attaches the whole text of each movie's screenplay to its Movie node as a `Script` node with a `HAS_SCRIPT` edge. Pulp Fiction is 296,961 characters of screenplay in one database column. We retrieve it with a one-line Cypher traversal. Sit with that for a second: a graph database is now storing 300 KB of plain text on a single node and the query language pulls it out as easily as a name or a year.

**Act II** asks a question whose answer is in the opening of the script (Pumpkin and Honey Bunny's robbery personas). The LLM truncates the script to 40K characters, reads the opening, answers correctly. So far so good. Then I ask the same shape of question about something in the middle of the script: the Sunken Place sequence in Get Out, which lives halfway through a 232 KB screenplay. The truncation cuts it off. The LLM honestly reports that the text doesn't say. The notebook tells you this is going to happen so you can see it.

**Act III** introduces the `Scene` node. Same screenplay, chunked on `INT./EXT.` slug lines. The Get Out script is one Script node containing 232 KB of text, but it's also 118 Scene nodes averaging two thousand characters each. The Sunken Place reveals live in three specific scenes. The Cypher changes shape: instead of returning the whole script, it filters Scene bodies with `CONTAINS` and returns only the matching chunks. Context shrinks ten to a hundred times. The LLM gets clean, focused passages instead of a 50,000-token wall of text. The questions that failed in Act II all work.

This is the production pattern for graph + text RAG, full stop. The graph picks the document *and* the right chunks inside it. The LLM only ever reads what it actually needs.

## What surprised me

A few specific things I want to flag because they'd cost a junior engineer a couple of hours each.

**Kuzu's `CONTAINS` is case-sensitive.** The first time I shipped NB4, the LLM wrote `WHERE sc.body CONTAINS 'sunken place'` and got zero rows. The script capitalizes "Sunken Place." The fix is to wrap both sides with `lower()`: `WHERE lower(sc.body) CONTAINS lower('sunken place')`. The same idiom exists in SQL and most people learn it once and never get bit again. But the failure mode is *silent zero rows*, not an error, and the LLM correctly reported "the text doesn't say." Which makes the model look broken when it isn't.

**Movie scripts have unpredictable line breaks.** The Ezekiel speech in Pulp Fiction is in the script, but the literal phrase "path of the righteous man" isn't, because the screenplay wraps it across lines with deep indentation. `CONTAINS 'path of the righteous man'` returns zero rows. `CONTAINS 'Ezekiel'` finds it instantly. The lesson generalizes: when you're filtering chunked text by substring, *prefer one short distinctive keyword over a long phrase*. Or accept the brittleness and reach for vector search instead, which is the next module of this repo.

I updated NB4's Cypher-generation prompt to teach `gpt-4o-mini` both of these. `temperature=0` makes the model follow the instructions reliably. The prompt itself is documented in the notebook so you can see exactly what changed.

**Kuzu also changed its on-disk format mid-development.** Older Kuzu versions stored the database as a directory; Kuzu 0.11+ stores it as a single file plus a sibling `.wal` write-ahead log. My corpus builder originally only deleted the main file. The orphaned WAL replayed the old schema on the next open, and `CREATE TABLE` failed with "already exists." Three lines of fix in the cleanup logic, but for a minute I thought I'd broken Cypher itself.

These three things are why I think the notebooks earn their length. The path to "it works" in graph RAG has more friction than vector RAG, and most of it isn't documented anywhere except in the moment you hit it.

## Key takeaways

The same shape of section I had in the module 01 blog. Five things from the experience of actually building this.

**The first move in production graph RAG is choosing your chunk.** A vector RAG system has a chunk size and an embedding model. A graph RAG system has a chunk *and* a graph schema. NB4's `Scene` chunking on `INT./EXT.` slug lines worked because screenplays have natural section breaks. For your domain it'll be something else (support tickets per ticket, chapters per chapter, function bodies per function, paragraphs per paragraph). The wrong chunk is the source of most of the pain in this kind of system.

**The graph is the index, not the answer.** The thing I kept catching myself doing in NB3 was using the graph as the answer source. The LLM rephrased structured rows into a sentence. That's a perfectly valid kind of retrieval ("give me a tabular fact synthesized as prose" is a real use case). But the meatier version, the one that does what people actually mean by "RAG," is NB4's pattern. The graph filters the corpus down to relevant chunks. The LLM reads the chunks. The graph *informs* the answer; the text *is* the answer.

**LangChain is a fine abstraction over something simple.** Module 02 used LangChain because it's the de facto orchestration layer for graph RAG, and I wanted to be honest about what it buys you. The blog version: it saves forty lines of orchestration code and hides two LLM calls per question behind a single `chain.invoke`. The chain is fine. The reason NB3 builds the manual loop first is so that when (not if) the chain misbehaves, you know what to look for. Building from primitives is a habit. The habit pays off the first time something doesn't work.

**Two LLM calls per question adds up faster than you'd think.** Each graph RAG question is one LLM call to write Cypher, plus one to write the answer. For my eleven-movie corpus with `gpt-4o-mini`, a full NB3 run is under one cent and a full NB4 run is under two cents. Scale that to a thousand users a day and you're talking about real money. The first cost-saving move in production is dropping the answer-synthesis step when you can return raw rows, which is doable on certain query shapes but not on text-RAG ones. The second is moving to vector search where the LLM only writes Cypher once during setup. Both are module 05 territory.

**Maintenance is a different conversation for graph RAG than vector RAG.** Module 01's blog had a section on maintenance for vector RAG: index freshness, embedding model drift, evaluation, cost monitoring. Graph RAG inherits all of those and adds three more. The schema is a contract. Rename a relationship and every Cypher prompt is stale. The chunking strategy is a contract too. Change `INT./EXT.` to something else and 1,809 nodes need rebuilding. And entity linking matters more: "Keanu Reeves" matches the graph; "keanu reaves" doesn't. The fix is usually a vector lookup over node names before generating Cypher, but it's a layer you have to build.

## Where this module doesn't go

Two big gaps, both deliberate.

**Vector search inside graph-located documents.** NB4's Act III is keyword-based: filter Scene bodies with `CONTAINS`. That works when the question contains the answer's keywords. It breaks when the question is conceptually related to the answer but uses different words. The fix is embedding each Scene's text and filtering by cosine similarity instead of `CONTAINS`. That's the hybrid pattern, and it's the main subject of [module 05](https://github.com/allllc/rag-unpacked/tree/main/05-advanced-rag).

**Evaluation.** Both modules 01 and 02 punted on this. How do you know if your graph RAG system actually got better when you changed the schema, or the prompt, or the chunking? Eyeballing the answers is fine for a portfolio piece, not for production. [Module 04](https://github.com/allllc/rag-unpacked/tree/main/04-evaluating-rag) is the answer.

## Dig in

- Repo: [github.com/allllc/rag-unpacked](https://github.com/allllc/rag-unpacked)
- Module 02: [02-graph-rag](https://github.com/allllc/rag-unpacked/tree/main/02-graph-rag)
- Cheat sheet: [02-graph-rag/assets/cheatsheet.png](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/assets/cheatsheet.png)
- The corpus builder, including the scene-chunking regex: [scripts/build_corpus.py](https://github.com/allllc/rag-unpacked/blob/main/02-graph-rag/scripts/build_corpus.py)

If you find something I got wrong or want to argue about the chunk strategy, open an issue. Module 03 (vectorless RAG) is up next.
