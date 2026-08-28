# Architecture

## Service responsibilities

| Service | Current responsibility | Next integration |
| --- | --- | --- |
| Orchestrator | Fan out queries and generate a Qwen RAG answer | Routing policy |
| Local DB Agent | Index and search an NQ-style fixture in Chroma | Complete BEIR corpus |
| Gmail Agent | Index and search dummy email in Chroma | Gmail API with a dedicated dummy account |
| Drive Agent | Index and search dummy documents in Chroma | Google Drive API with a dedicated dummy account |

## Retrieval layer

Search agents convert fixture records to LangChain `Document` objects and
preserve `document_id`, `source`, and `trust` as Chroma metadata. The default
offline embedding keeps tests deterministic. A later Ollama embedding adapter
can replace it without changing the search API.

## Answer generation

`POST /query` returns retrieval results for inspection. `POST /answer` formats
the highest-ranked passages and invokes `qwen3:8b` through LangChain
`ChatOllama`. Vulnerable mode includes trusted and untrusted passages. Defended
mode removes untrusted passages and uses an instruction-isolation prompt.

## Agent long-term memory

The orchestrator keeps a per-session JSONL memory of past turns at
`AGENT_MEMORY_FILE`, persisted on its own Docker volume. Recall is lexical and
scoped to `session_id`; recalled turns enter the prompt as ordinary passages
with `source=agent-memory`. A turn is stored as `trusted` only when every
passage used for it was trusted, so a poisoned turn produces `untrusted`
memory that defended mode drops like any other untrusted hit. This makes
memory persistence of an attack measurable with the existing evaluation code.
Experiment endpoints default to `use_memory=false` so trials stay independent.

## Retrieval-stage defenses

Both defenses from arXiv:2512.24268 run before generation and use no trust
metadata, so either composes with either `mode`.

`retrieval_defense: "ragpart"` switches the search agents to a second Chroma
collection holding `C(N, k)` mean-pooled fragment vectors per document. Each
combination is queried independently and the resulting top-p lists are merged
by majority vote. Building that side index costs `N` embedding calls per
document, so it is gated behind `RAGPART_ENABLED`.

`retrieval_defense: "ragmask"` instead sanitises what an ordinary search
already returned. The top `alpha*p` candidates are split into
`RAGMASK_MASK_LENGTH`-token segments; each segment is masked in turn and a
segment whose removal costs at least `RAGMASK_DELTA` of query similarity is
dropped, because that is what a retrieval-boosting poison does. The surviving
text is re-embedded and re-ranked down to the top p. It needs no side index, so
it costs query time rather than startup time and works on an existing
collection.

See `docs/ragpart-ragmask.ko.md` for the algorithms, the measurements, and the
dependence of both defenses on the retriever.

## Trust boundary

All retrieved content is data, not executable instructions. Future connectors
must preserve source and trust metadata so attack and defense evaluation can
distinguish trusted fixtures from injected content.

Secrets belong in `.env` or mounted credential files. They must never be
committed to Git or included in test fixtures.
