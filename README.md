# RAG Unpacked

A hands-on exploration of Retrieval-Augmented Generation — from the basics to advanced patterns.

RAG is the technique of grounding a language model's responses in external knowledge: retrieve relevant context from a data store, then generate an answer that cites it. This repo unpacks that idea across five modules, each a self-contained notebook or project exploring a different facet of the problem.

## Modules

| # | Module | Description | Status |
|---|--------|-------------|--------|
| 01 | [Intro to RAG](./01-intro-to-rag/) | The canonical pattern: embeddings, a vector database (Pinecone), and a retrieval-augmented prompt. | 🚧 In progress |
| 02 | [Graph RAG](./02-graph-rag/) | Retrieval over a knowledge graph using Neo4j and LangChain — when structure beats similarity. | 📋 Planned |
| 03 | [Vectorless RAG](./03-vectorless-rag/) | Retrieval without embeddings. BM25, keyword search, and why you don't always need a vector store. | 📋 Planned |
| 04 | [Evaluating RAG](./04-evaluating-rag/) | Measuring what matters: faithfulness, answer relevance, context precision. | 📋 Planned |
| 05 | [Advanced RAG](./05-advanced-rag/) | Re-ranking, query rewriting, hybrid search, and other patterns that move the needle in production. | 📋 Planned |

## Getting started

Each module has its own README with setup instructions and dependencies. Clone the repo and jump into whichever module interests you:

```bash
git clone https://github.com/allllc/rag-unpacked.git
cd rag-unpacked/01-intro-to-rag
```

## License

MIT — see [LICENSE](./LICENSE).
