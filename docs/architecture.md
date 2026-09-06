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

These run before generation and use no trust metadata, so any of them composes
with either `mode`. RAGPart and RAGMask come from arXiv:2512.24268; the
redundancy filter is this lab's own, derived from measurements in
`docs/defense-roadmap.ko.md`.

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

`retrieval_defense: "cluster"` clusters the top `alpha*p` candidates by cosine
similarity and drops any group of `CLUSTER_MIN_SIZE` or more. PoisonedRAG needs
several near-duplicate poisons per query, so the attack is visible *between*
candidates even when each one looks individually plausible. Clean documents in
this corpus do not form such groups. It falls back to the undefended ranking if
every candidate is dropped.

See `docs/ragpart-ragmask.ko.md` for the algorithms, the measurements, and the
dependence of both defenses on the retriever.

## Generation-stage defenses

`generation_defense` answers each retrieved passage in isolation so an injected
passage can corrupt only its own response, then combines the responses.
`"robustrag"` is the paper's keyword aggregation (Xiang et al., SaTML 2026);
`"isolate_conflict"` is this lab's variant, which reports disagreement instead
of counting. They are orthogonal to `mode` and to `retrieval_defense`, and both
cost one LLM call per passage.

Measured against the current attack, the variant is harmful on its own and
useful only behind the redundancy filter. See `docs/robustrag.ko.md` for the
full table and the reasoning it overturned.

## Switching defenses off

Three levels: the per-request `retrieval_defense` / `generation_defense`
fields, the `DEFAULT_RETRIEVAL_DEFENSE` / `DEFAULT_GENERATION_DEFENSE`
deployment settings that apply when a request omits the field, and
`DEFENSES_ENABLED=false`, which forces every defense off regardless of the
request so a whole run returns to the undefended baseline.

## Trust boundary

All retrieved content is data, not executable instructions. Future connectors
must preserve source and trust metadata so attack and defense evaluation can
distinguish trusted fixtures from injected content.

Secrets belong in `.env` or mounted credential files. They must never be
committed to Git or included in test fixtures.
