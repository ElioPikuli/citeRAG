# CiteRAG: design

## 1. Goal

A retrieval-augmented question-answering service over a folder of documents and code that:

1. answers only from the indexed sources, and says so when it can't;
2. shows **where every claim came from** (file, section, line range);
3. can be **measured**: retrieval quality is a number in CI, not a feeling;
4. runs **fully offline** for development and tests, and uses Claude for answers when an API key is present.

The demo corpus is the author's own portfolio projects ("Ask my projects"), so a recruiter can ask
*"How does ConcreteCalc compute labor days?"* and get a cited answer.

## 2. Non-goals (v1)

- Multi-tenant auth, user accounts, rate limiting.
- A hosted vector database. The corpus is thousands of chunks, not millions; an in-memory NumPy matrix is
  faster and simpler than a network hop. The `VectorStore` interface leaves room to swap one in.
- Agentic multi-hop retrieval. One retrieve → generate pass is enough for this corpus and much cheaper.

## 3. Architecture

```
            ┌───────────── ingest (offline, incremental) ─────────────┐
 files ──▶ loaders ──▶ structure-aware chunker ──▶ embedder ──▶ index/ (chunks.jsonl, vectors.npy, manifest.json)
 (.md .html .py .js …)   (headings, code blocks,      (fastembed bge-small, local ONNX;
                          line ranges kept)            hashing embedder for tests)

            ┌──────────────────── query (online) ─────────────────────┐
 question ─▶ BM25 (lexical) ─┐
            └▶ dense cosine ──┴▶ Reciprocal Rank Fusion ─▶ MMR dedupe ─▶ abstain gate ─▶ generator ─▶ SSE stream
                                                                                         │
                                                         Claude (search_result blocks + native citations)
                                                         or extractive fallback (no API key)
```

| Module | Responsibility |
|---|---|
| `citerag/loaders.py` | Read files, strip HTML to text, skip binaries / `node_modules` / lockfiles. |
| `citerag/chunking.py` | Split by Markdown headings and code structure first, then by size with overlap. Every chunk keeps `source`, `heading path`, `start_line`, `end_line`. |
| `citerag/text.py` | Tokenizer shared by BM25 and the extractive generator: Unicode words, splits `camelCase` / `snake_case`, light stop-word list. |
| `citerag/bm25.py` | Okapi BM25 (k1=1.5, b=0.75), sparse postings, no dependencies. |
| `citerag/embeddings.py` | `Embedder` protocol; `FastEmbedEmbedder` (BAAI/bge-small-en-v1.5, 384-d, local ONNX) and `HashingEmbedder` (deterministic, zero download, for tests). |
| `citerag/store.py` | Persisted index + **incremental re-indexing** keyed by file content hash; refuses to load an index built with a different embedder. |
| `citerag/retriever.py` | Hybrid retrieval with RRF (k=60), optional MMR diversity, per-stage scores kept for the debug view. |
| `citerag/generation.py` | `ClaudeGenerator` (streaming, citations) and `ExtractiveGenerator` (offline). Same event stream for both. |
| `citerag/pipeline.py` | Wires retrieve → abstain gate → generate. |
| `citerag/api.py` + `web/` | FastAPI, Server-Sent Events, single-page UI with citation chips and a retrieval inspector. |
| `citerag/evaluation.py` | Golden-set retrieval metrics (Hit@k, Recall@k, MRR, nDCG@k) for BM25 vs dense vs hybrid. |
| `citerag/cli.py` | `ingest`, `search`, `ask`, `eval`, `serve`. |

## 4. Key decisions

**Hybrid retrieval, fused by rank, not score.** BM25 is strong on identifiers (`calcLaborDays`, `includeWaste`)
that embeddings blur; embeddings are strong on paraphrase ("how is the price worked out"). Their scores live on
different scales, so I fuse **ranks** with RRF (`Σ 1/(60 + rank)`), which needs no tuning or normalization.
The eval reports all three so the choice is justified by numbers.

*Outcome:* the answer depended on the corpus. With two projects indexed, dense retrieval was best or tied on
every slice, identifiers included, because chunks are embedded with their heading path
(`calc.js > calcLaborDays`), so dense shipped as the default. With six projects (78 files, much more code), dense
and hybrid tie on Hit@5 (0.98), hybrid leads on nDCG@5 (0.850 vs 0.831), and dense misses 1 of 10 identifier
questions that hybrid answers. The default is now `mode = "hybrid"`. With 50 answerable questions these gaps are
close to noise, so this is a default to re-check as the corpus grows.

**Claude's native `search_result` blocks for citations.** Each retrieved chunk is sent as a `search_result`
content block (`source` = path#lines, `title` = heading path). Claude returns `search_result_location`
citations that point to an exact chunk index, so citations are produced by the API, not parsed out of free text,
and cannot reference a chunk that wasn't retrieved.

**Abstain before calling the model.** If the best fused hit has no lexical overlap and low cosine similarity, the
pipeline answers "not in the indexed sources" without an LLM call. Cheaper, and it removes the most common
hallucination path (answering an off-topic question from general knowledge).

**Local embeddings.** fastembed runs bge-small via ONNX on CPU: no second API key, no per-query cost,
~10 ms per query. The embedder name is stored in the index manifest so a model change forces a rebuild
rather than silently mixing vector spaces.

**Offline-first.** Without `ANTHROPIC_API_KEY` the app still works end to end with an extractive generator
(best-matching sentences, each with a citation), and the UI says it is in offline mode. All tests run offline;
the Claude path is tested against a fake client that checks the exact request shape and the citation mapping.

**Model settings.** `claude-opus-5-5`, adaptive thinking, `effort: low` (grounded Q&A over short context does
not benefit from deep reasoning, and latency matters in a chat UI), streaming, and the server-side refusal
fallback (`fallbacks: "default"`).

## 5. Evaluation

`eval/golden.jsonl`: hand-written questions, each with the source file(s) that contain the answer.
`citerag eval` reports, per retriever (bm25 / dense / hybrid):

- **Hit@k**: at least one relevant source in the top k.
- **Recall@k**: fraction of relevant sources found in the top k.
- **MRR**: 1 / rank of the first relevant hit.
- **nDCG@k**: rank-aware gain.

The test suite asserts a floor on Hit@5 and MRR for the configured retriever so a regression in chunking or retrieval fails CI.

## 6. Testing plan

- Unit: tokenizer, chunker boundaries and line numbers, BM25 against hand-computed values, RRF, MMR,
  store round-trip and incremental update, abstain gate.
- Generator: fake Anthropic client asserts `search_result` blocks, model, streaming, and that
  `citations_delta` events map back to the right chunk.
- API: FastAPI TestClient, SSE event order (`sources` → `token`… → `citation`… → `done`), validation errors.
- End to end: build the real index over the demo corpus, run the golden set, drive the UI in a browser.

## 7. Future work

Cross-encoder re-ranking, query rewriting for follow-up questions, LLM-as-judge faithfulness eval, a hosted
vector store behind the same interface, Docker deploy.
