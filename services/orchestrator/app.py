import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from services.common.schemas import (
    AgentPoisonBenchmarkFailure,
    AgentPoisonBenchmarkPoint,
    AgentPoisonBenchmarkRequest,
    AgentPoisonBenchmarkResponse,
    AgentPoisonMetrics,
    AgentPoisonRequest,
    AgentPoisonResponse,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentDocumentRequest,
    ExperimentDocumentResponse,
    ExperimentResetResponse,
    ExperimentEvaluationRequest,
    ExperimentEvaluationResponse,
    HealthResponse,
    MemoryListResponse,
    MemoryResetResponse,
    OrchestratorAnswerRequest,
    OrchestratorQueryRequest,
    AttackDashboardMetrics,
    PoisonedRAGBenchmarkPoint,
    PoisonedRAGBenchmarkRequest,
    PoisonedRAGBenchmarkResponse,
    PoisonedRAGCandidate,
    PoisonedRAGPipelineStatus,
    PoisonedRAGRequest,
    PoisonedRAGResponse,
    PoisonedRAGRunFailure,
    PoisonedRAGRunMetadata,
    SearchHit,
)
from services.common.embeddings import create_embeddings
from services.common.search import load_json_documents
from services.orchestrator.agent_poison import (
    craft_poison_value,
    embed_memory,
    optimize_trigger,
    rank_embedded_memory,
    retrieval_success,
)
from services.orchestrator.evaluation import (
    answer_accuracy,
    answers_agree,
    attack_success_rate,
    evaluate_answer,
    evaluate_retrieval,
    phrase_adopted,
    retrieval_stage_metrics,
)
from services.orchestrator.memory import ConversationMemory, memory_to_hit
from services.orchestrator.poisoned_rag import (
    GeneratedPoison,
    generate_poison_set,
    select_diverse_candidates,
)
from services.orchestrator.robust_rag import (
    RobustRagConfig,
    isolate,
    isolation_only,
    keyword_aggregate,
)
from services.orchestrator.rag import (
    build_keyword_chain,
    build_rag_chain,
    collect_context_hits,
    create_chat_model,
    format_context,
)

SERVICE_NAME = "orchestrator"
VERSION = os.getenv("APP_VERSION", "0.1.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
RAG_CONTEXT_LIMIT = int(os.getenv("RAG_CONTEXT_LIMIT", "6"))
ROBUSTRAG_GROUP_SIZE = int(os.getenv("ROBUSTRAG_GROUP_SIZE", "1"))
# RobustRAG keyword-aggregation parameters, at the paper's defaults.
ROBUSTRAG_ALPHA = float(os.getenv("ROBUSTRAG_ALPHA", "0.2"))
ROBUSTRAG_BETA = float(os.getenv("ROBUSTRAG_BETA", "3"))


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Master switch. False forces every defense off no matter what a request asks
# for, so a whole run can be taken back to the undefended baseline without
# editing the callers. The per-request fields stay the fine-grained control:
# omitted -> the DEFAULT_* deployment setting, "none" -> off for that request.
DEFENSES_ENABLED = _flag("DEFENSES_ENABLED", "true")
DEFAULT_RETRIEVAL_DEFENSE = os.getenv("DEFAULT_RETRIEVAL_DEFENSE", "none")
DEFAULT_GENERATION_DEFENSE = os.getenv("DEFAULT_GENERATION_DEFENSE", "none")


def resolve_retrieval_defense(requested: str | None) -> str:
    if not DEFENSES_ENABLED:
        return "none"
    return DEFAULT_RETRIEVAL_DEFENSE if requested is None else requested


def resolve_generation_defense(requested: str | None) -> str:
    if not DEFENSES_ENABLED:
        return "none"
    return DEFAULT_GENERATION_DEFENSE if requested is None else requested
AGENT_URLS = {
    "local_db": os.getenv("LOCAL_DB_AGENT_URL", "http://localhost:8001"),
    "gmail": os.getenv("GMAIL_AGENT_URL", "http://localhost:8002"),
    "drive": os.getenv("DRIVE_AGENT_URL", "http://localhost:8003"),
}

app = FastAPI(title=SERVICE_NAME, version=VERSION)
DEMO_FILE = Path(__file__).with_name("static") / "demo.html"
EXPERIMENT_RESULTS_DIR = Path(
    os.getenv("EXPERIMENT_RESULTS_DIR", "/tmp/poisonedrag-results")
)
EXPERIMENT_SCENARIOS_FILE = Path(
    os.getenv(
        "EXPERIMENT_SCENARIOS_FILE",
        str(Path(__file__).resolve().parents[2] / "datasets/experiments/nq_target_queries.json"),
    )
)
AGENT_MEMORY_FILE = Path(
    os.getenv("AGENT_MEMORY_FILE", "/tmp/agent-memory/memory.jsonl")
)
MEMORY_RECALL_LIMIT = int(os.getenv("MEMORY_RECALL_LIMIT", "3"))
AGENT_POISON_CORPUS_FILE = Path(
    os.getenv(
        "AGENT_POISON_CORPUS_FILE",
        # The full 2.68M-document corpus (datasets/generated/nq_2681468.json)
        # is too large to commit to git (~1.6GB; GitHub rejects files over
        # 100MB), so it isn't in the repo -- deployments that have it placed
        # out-of-band point AGENT_POISON_CORPUS_FILE at it instead. This
        # default is the smaller, git-tracked corpus so a fresh clone works
        # without any extra setup.
        str(Path(__file__).resolve().parents[2] / "datasets/generated/nq_100000.json"),
    )
)


@lru_cache(maxsize=1)
def _memory() -> ConversationMemory:
    return ConversationMemory(AGENT_MEMORY_FILE)


@app.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(DEMO_FILE)


@app.get("/experiments/scenarios")
async def experiment_scenarios() -> dict[str, Any]:
    try:
        scenarios = json.loads(EXPERIMENT_SCENARIOS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Experiment scenarios are unavailable") from exc
    required = {"id", "query", "expected_answer", "attack_target"}
    if not isinstance(scenarios, list) or any(
        not isinstance(item, dict) or not required.issubset(item)
        for item in scenarios
    ):
        raise HTTPException(status_code=503, detail="Experiment scenario data is invalid")
    return {"dataset": "beir-nq", "count": len(scenarios), "scenarios": scenarios}


@lru_cache(maxsize=1)
def _rag_model():
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
    )


def _attack_generation_model(temperature: float):
    return create_chat_model(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=OLLAMA_NUM_PREDICT,
    )


async def _answer_with_supplied_context(question: str, context: str) -> str:
    chain = build_rag_chain(_rag_model(), mode="vulnerable")
    return await chain.ainvoke({"question": question, "context": context})


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service=SERVICE_NAME, version=VERSION)


@app.get("/agents")
async def agents() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(client, "GET", f"{url}/health")
                for url in AGENT_URLS.values()
            ],
            return_exceptions=True,
        )

        source_responses = await asyncio.gather(
            *[
                _request_json(client, "GET", f"{AGENT_URLS[source]}/source-status")
                for source in ("gmail", "drive")
            ],
            return_exceptions=True,
        )

    status: dict[str, Any] = {}
    for source, response in zip(AGENT_URLS, responses, strict=True):
        if isinstance(response, Exception):
            status[source] = {"status": "unavailable", "error": str(response)}
        else:
            status[source] = response
    for source, response in zip(("gmail", "drive"), source_responses, strict=True):
        if not isinstance(response, Exception) and status[source].get("status") != "unavailable":
            status[source]["data_source"] = response
    return {"service": SERVICE_NAME, "agents": status}


@app.get("/model")
async def model() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await _request_json(
                client,
                "GET",
                f"{OLLAMA_BASE_URL}/api/tags",
            )
    except Exception as exc:
        return {
            "status": "unavailable",
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "error": str(exc),
        }

    installed_models = [
        item.get("name", "")
        for item in response.get("models", [])
        if isinstance(item, dict)
    ]
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "installed": OLLAMA_MODEL in installed_models,
        "available_models": installed_models,
    }


@app.get("/experiments/corpus-status")
async def corpus_status() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            return await _request_json(
                client, "GET", f"{AGENT_URLS['local_db']}/stats"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Local corpus status is unavailable"
        ) from exc


@app.post(
    "/experiments/documents",
    response_model=ExperimentDocumentResponse,
    status_code=201,
)
async def create_experiment_document(
    request: ExperimentDocumentRequest,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            return await _request_json(
                client,
                "POST",
                f"{AGENT_URLS['local_db']}/documents",
                json=request.model_dump(),
            )
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local DB agent is unavailable: {exc}",
        ) from exc


@app.delete(
    "/experiments/documents",
    response_model=ExperimentResetResponse,
)
async def reset_experiment_documents() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            return await _request_json(
                client,
                "DELETE",
                f"{AGENT_URLS['local_db']}/documents/untrusted",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local DB agent is unavailable: {exc}",
        ) from exc


async def _query_agents(
    request: OrchestratorQueryRequest,
) -> dict[str, Any]:
    selected_sources = list(dict.fromkeys(request.sources))
    payload = {
        "query": request.query,
        "limit": request.limit,
        "defense": resolve_retrieval_defense(request.retrieval_defense),
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(
                    client,
                    "POST",
                    f"{AGENT_URLS[source]}/search",
                    json=payload,
                )
                for source in selected_sources
            ],
            return_exceptions=True,
        )

    results: dict[str, Any] = {}
    for source, response in zip(selected_sources, responses, strict=True):
        if isinstance(response, Exception):
            results[source] = {"status": "error", "error": str(response)}
        else:
            results[source] = {"status": "ok", **response}
    return results


@app.post("/query")
async def query(request: OrchestratorQueryRequest) -> dict[str, Any]:
    results = await _query_agents(request)

    return {
        "service": SERVICE_NAME,
        "query": request.query,
        "results": results,
    }


@app.post("/answer")
async def answer(request: OrchestratorAnswerRequest) -> dict[str, Any]:
    return await _generate_answer(request)


async def _generate_answer(
    request: OrchestratorAnswerRequest,
) -> dict[str, Any]:
    retrieval_request = request
    if (
        request.mode == "defended"
        or request.allowed_untrusted_document_ids is not None
    ) and request.limit < 20:
        retrieval_request = request.model_copy(update={"limit": 20})
    results = await _query_agents(retrieval_request)
    if request.allowed_untrusted_document_ids is not None:
        allowed = set(request.allowed_untrusted_document_ids)
        for result in results.values():
            if result.get("status") != "ok":
                continue
            result["hits"] = [
                hit
                for hit in result.get("hits", [])
                if hit.get("trust") != "untrusted"
                or hit.get("document_id") in allowed
            ]
    # Recalled turns share the requested Top-K budget rather than adding to
    # it, so the context size still matches what the caller asked for. Memory
    # is capped at half the budget so it can never starve live retrieval.
    context_limit = min(request.limit, RAG_CONTEXT_LIMIT)
    memory_hits = _recall_memory(request)[: context_limit // 2]
    context_hits = memory_hits + collect_context_hits(
        results,
        limit=context_limit - len(memory_hits),
        trusted_only=request.mode == "defended",
    )
    if not context_hits:
        return {
            "service": SERVICE_NAME,
            "query": request.query,
            "model": OLLAMA_MODEL,
            "mode": request.mode,
            "answer": "I cannot determine the answer from the retrieved context.",
            "documents": [],
            "results": results,
            "memory": [],
        }

    generation_defense = resolve_generation_defense(request.generation_defense)
    chain = build_rag_chain(_rag_model(), mode=request.mode)
    surviving_keywords: list[str] | None = None
    try:
        if generation_defense in {"robustrag", "isolation"}:
            # Both share the paper's isolation: answer each group alone so an
            # injected passage can influence only its own response. They differ
            # only in how the isolated responses are combined.
            groups = isolate(
                context_hits,
                RobustRagConfig(group_size=ROBUSTRAG_GROUP_SIZE).group_size,
            )
            isolated = [
                await chain.ainvoke(
                    {"question": request.query, "context": format_context(group)}
                )
                for group in groups
            ]
            if generation_defense == "robustrag":
                # The paper's rule: keep keywords appearing in at least
                # min(alpha*n, beta) of the isolated responses, then answer
                # from the surviving keywords alone.
                keyword_result = keyword_aggregate(
                    isolated,
                    alpha=ROBUSTRAG_ALPHA,
                    beta=ROBUSTRAG_BETA,
                )
                surviving_keywords = keyword_result.keywords
                if surviving_keywords:
                    generated_answer = await build_keyword_chain(
                        _rag_model()
                    ).ainvoke(
                        {
                            "question": request.query,
                            "keywords": ", ".join(surviving_keywords),
                        }
                    )
                else:
                    generated_answer = (
                        "I cannot determine the answer from the retrieved context."
                    )
            else:
                # Isolation with no combining rule. Only useful once retrieval
                # has removed the poisons; see robust_rag.isolation_only.
                generated_answer = isolation_only(isolated).answer
        else:
            generated_answer = await chain.ainvoke(
                {
                    "question": request.query,
                    "context": format_context(context_hits),
                }
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model {OLLAMA_MODEL} is unavailable: {exc}",
        ) from exc

    if request.use_memory:
        _memory().append(
            session_id=request.session_id,
            query=request.query,
            answer=generated_answer,
            trust=(
                "trusted"
                if all(hit.trust == "trusted" for hit in context_hits)
                else "untrusted"
            ),
        )

    return {
        "service": SERVICE_NAME,
        "query": request.query,
        "model": OLLAMA_MODEL,
        "mode": request.mode,
        "answer": generated_answer,
        "documents": [hit.model_dump() for hit in context_hits],
        "results": results,
        "memory": [hit.model_dump() for hit in memory_hits],
        "generation_defense": generation_defense,
        "retrieval_defense": resolve_retrieval_defense(request.retrieval_defense),
        "surviving_keywords": surviving_keywords,
    }


def _recall_memory(request: OrchestratorAnswerRequest) -> list[SearchHit]:
    """Recall past turns of the same session as extra context passages.

    Memory written while poisoned passages were in context is `untrusted`, so
    defended mode drops it exactly like an untrusted retrieval hit.
    """
    if not request.use_memory:
        return []
    records = _memory().recall(
        request.query,
        session_id=request.session_id,
        limit=MEMORY_RECALL_LIMIT,
        trusted_only=request.mode == "defended",
    )
    return [memory_to_hit(record) for record in records]


@app.get("/memory", response_model=MemoryListResponse)
async def list_memory(
    session_id: str | None = None,
    limit: int = 20,
) -> MemoryListResponse:
    records = _memory().list(session_id=session_id, limit=limit)
    return MemoryListResponse(
        service=SERVICE_NAME,
        session_id=session_id,
        count=len(records),
        records=records,
    )


@app.delete("/memory", response_model=MemoryResetResponse)
async def reset_memory(session_id: str | None = None) -> MemoryResetResponse:
    memory = _memory()
    deleted_count = memory.clear(session_id=session_id)
    return MemoryResetResponse(
        session_id=session_id,
        deleted_count=deleted_count,
        remaining_count=len(memory.records),
    )


@app.post(
    "/experiments/evaluate",
    response_model=ExperimentEvaluationResponse,
)
async def evaluate_experiment(
    request: ExperimentEvaluationRequest,
) -> ExperimentEvaluationResponse:
    answer_payload = await _generate_answer(request)
    return _build_evaluation_response(request, answer_payload)


def _build_evaluation_response(
    request: ExperimentEvaluationRequest,
    answer_payload: dict[str, Any],
) -> ExperimentEvaluationResponse:
    documents = [
        SearchHit.model_validate(document)
        for document in answer_payload["documents"]
    ]
    outcome, expected_present, target_present = evaluate_answer(
        answer_payload["answer"],
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
    )
    (
        attack_retrieved,
        attack_rank,
        attack_score,
        untrusted_count,
    ) = evaluate_retrieval(
        documents,
        attack_document_ids=request.attack_document_ids,
    )

    return ExperimentEvaluationResponse(
        service=SERVICE_NAME,
        query=request.query,
        model=OLLAMA_MODEL,
        mode=request.mode,
        answer=answer_payload["answer"],
        outcome=outcome,
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        expected_answer_present=expected_present,
        attack_target_present=target_present,
        attack_document_retrieved=attack_retrieved,
        attack_document_rank=attack_rank,
        attack_document_score=attack_score,
        untrusted_document_count=untrusted_count,
        documents=documents,
    )


@app.post(
    "/experiments/compare",
    response_model=ExperimentComparisonResponse,
)
async def compare_experiment_modes(
    request: ExperimentComparisonRequest,
) -> ExperimentComparisonResponse:
    common_fields = request.model_dump()
    vulnerable_request = ExperimentEvaluationRequest(
        **common_fields,
        mode="vulnerable",
    )
    defended_request = ExperimentEvaluationRequest(
        **common_fields,
        mode="defended",
    )

    vulnerable_payload = await _generate_answer(vulnerable_request)
    defended_payload = await _generate_answer(defended_request)
    vulnerable = _build_evaluation_response(
        vulnerable_request,
        vulnerable_payload,
    )
    defended = _build_evaluation_response(
        defended_request,
        defended_payload,
    )

    vulnerable_succeeded = vulnerable.outcome == "attack_succeeded"
    defended_succeeded = defended.outcome == "attack_succeeded"
    return ExperimentComparisonResponse(
        service=SERVICE_NAME,
        query=request.query,
        model=OLLAMA_MODEL,
        vulnerable=vulnerable,
        defended=defended,
        attack_succeeded_in_vulnerable=vulnerable_succeeded,
        attack_succeeded_in_defended=defended_succeeded,
        defense_blocked_attack=vulnerable_succeeded and not defended_succeeded,
    )


@app.post(
    "/experiments/poisoned-rag",
    response_model=PoisonedRAGResponse,
)
async def run_poisoned_rag_experiment(
    request: PoisonedRAGRequest,
) -> PoisonedRAGResponse:
    run_started = perf_counter()
    started_at = datetime.now(timezone.utc)
    if request.cleanup_before_run:
        await reset_experiment_documents()

    run_id = uuid4().hex[:12]
    candidate_count = request.poison_count * request.candidate_multiplier
    placeholder_ids = [f"poisonedrag-{run_id}-{index}" for index in range(1, request.poison_count + 1)]
    baseline_request = ExperimentEvaluationRequest(
        query=request.query,
        sources=["local_db"],
        limit=request.top_k,
        mode="vulnerable",
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        attack_document_ids=placeholder_ids or [f"poisonedrag-{run_id}-none"],
        allowed_untrusted_document_ids=[],
        retrieval_defense=request.retrieval_defense,
    )
    baseline = _build_evaluation_response(
        baseline_request,
        await _generate_answer(baseline_request),
    )

    if request.fixed_candidates is not None:
        generated = [
            GeneratedPoison(
                instruction=item.instruction,
                poison_text=item.poison_text,
                verification_answer=item.verification_answer,
                verified=item.verified,
                generation_queries=0,
                generation_seconds=0.0,
                word_count=item.word_count,
            )
            for item in request.fixed_candidates[:request.poison_count]
        ]
    else:
        try:
            generated = await generate_poison_set(
                _attack_generation_model(request.generation_temperature),
                answer_with_context=_answer_with_supplied_context,
                query=request.query,
                attack_target=request.attack_target,
                count=candidate_count,
                word_count=request.passage_word_count,
                max_trials=request.max_generation_trials,
                composition=request.poison_composition,
                style=request.poison_generation_style,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"PoisonedRAG generation failed with {OLLAMA_MODEL}: {exc}",
            ) from exc

    candidate_ids: list[str] = []
    for index, item in enumerate(generated, start=1):
        document_id = f"poisonedrag-{run_id}-{index}"
        if item.verified:
            candidate_ids.append(document_id)
            await create_experiment_document(
                ExperimentDocumentRequest(
                    document_id=document_id,
                    source="poisonedrag-black-box",
                    tags=[
                        "controlled",
                        "poison",
                        "poisonedrag",
                        "black-box",
                        "q-plus-i",
                    ],
                    text=item.poison_text,
                )
            )
    retrieval_results = await _query_agents(
        OrchestratorQueryRequest(query=request.query, sources=["local_db"], limit=20)
    )
    retrieval_scores = {
        hit["document_id"]: float(hit["score"])
        for hit in retrieval_results.get("local_db", {}).get("hits", [])
    }
    ranked_candidates = [
        (index, item, retrieval_scores.get(f"poisonedrag-{run_id}-{index}", 0.0))
        for index, item in enumerate(generated, start=1)
    ]
    selected_indexes, rejection_reasons = select_diverse_candidates(
        ranked_candidates, count=request.poison_count
    )
    selected_set = set(selected_indexes)
    await _delete_experiment_documents(candidate_ids)

    generated_documents: list[PoisonedRAGCandidate] = []
    document_ids: list[str] = []
    for index, item in enumerate(generated, start=1):
        document_id = f"poisonedrag-{run_id}-{index}"
        selected = index in selected_set
        if selected:
            document_ids.append(document_id)
            await create_experiment_document(
                ExperimentDocumentRequest(
                    document_id=document_id,
                    source="poisonedrag-black-box",
                    tags=["controlled", "poison", "poisonedrag", "black-box", "q-plus-i", "selected"],
                    text=item.poison_text,
                )
            )
        score = retrieval_scores.get(document_id, 0.0)
        generated_documents.append(
            PoisonedRAGCandidate(
                document_id=document_id,
                instruction=item.instruction,
                poison_text=item.poison_text,
                verification_answer=item.verification_answer,
                verified=item.verified,
                generation_queries=item.generation_queries,
                generation_seconds=item.generation_seconds,
                word_count=item.word_count,
                retrieval_score=round(score, 6),
                selection_score=round(score, 6) if item.verified else None,
                selected=selected,
                rejection_reason=rejection_reasons.get(index),
            )
        )

    attacked_request = ExperimentEvaluationRequest(
        query=request.query,
        sources=["local_db"],
        limit=request.top_k,
        mode="vulnerable",
        expected_answer=request.expected_answer,
        attack_target=request.attack_target,
        attack_document_ids=document_ids or [f"poisonedrag-{run_id}-none"],
        allowed_untrusted_document_ids=document_ids,
        retrieval_defense=request.retrieval_defense,
    )
    attacked = _build_evaluation_response(
        attacked_request,
        await _generate_answer(attacked_request),
    )
    poison_ids = set(document_ids)
    poison_in_top_k = sum(
        document.document_id in poison_ids for document in attacked.documents
    )
    retrieved_count = len(attacked.documents)
    precision = poison_in_top_k / retrieved_count if retrieved_count else 0.0
    recall = poison_in_top_k / len(document_ids) if document_ids else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    retrieval_asr, retrieval_sr = retrieval_stage_metrics(
        attacked.documents,
        attack_document_ids=document_ids,
        expected_answer=request.expected_answer,
    )
    metrics = AttackDashboardMetrics(
        attack_success_rate=attack_success_rate(attacked.outcome),
        accuracy=answer_accuracy(attacked.outcome),
        retrieval_attack_success_rate=retrieval_asr,
        retrieval_success_rate=retrieval_sr,
        poison_in_top_k=poison_in_top_k,
        top_k=retrieved_count,
        retrieval_precision=round(precision, 4),
        retrieval_recall=round(recall, 4),
        retrieval_f1=round(f1, 4),
        generation_queries=sum(item.generation_queries for item in generated),
        generation_seconds=round(
            sum(item.generation_seconds for item in generated), 3
        ),
        total_seconds=round(perf_counter() - run_started, 3),
    )
    verified_count = sum(item.verified for item in generated)
    generation_status = (
        "skipped"
        if request.poison_count == 0
        else "completed"
        if len(document_ids) == request.poison_count
        else "partial"
    )
    injection_status = (
        "skipped"
        if request.poison_count == 0
        else "completed"
        if len(document_ids) == request.poison_count
        else "partial"
    )
    completed_at = datetime.now(timezone.utc)
    return PoisonedRAGResponse(
        run_id=run_id,
        retrieval_defense=request.retrieval_defense,
        requested_poison_count=request.poison_count,
        generated_candidate_count=len(generated),
        verified_poison_count=verified_count,
        injected_poison_count=len(document_ids),
        top_k=request.top_k,
        document_ids=document_ids,
        generated_documents=generated_documents,
        pipeline=PoisonedRAGPipelineStatus(
            generation=generation_status,
            injection=injection_status,
        ),
        metadata=PoisonedRAGRunMetadata(
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            service_version=VERSION,
            model=OLLAMA_MODEL,
            generation_temperature=request.generation_temperature,
            max_generation_trials=request.max_generation_trials,
            passage_word_count=request.passage_word_count,
            cleanup_before_run=request.cleanup_before_run,
            candidate_multiplier=request.candidate_multiplier,
            poison_composition=request.poison_composition,
            poison_generation_style=request.poison_generation_style,
        ),
        metrics=metrics,
        baseline=baseline,
        attacked=attacked,
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _write_benchmark_artifacts(
    result: PoisonedRAGBenchmarkResponse,
) -> None:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.json"
    csv_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.csv"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(PoisonedRAGBenchmarkPoint.model_fields),
        )
        writer.writeheader()
        writer.writerows(point.model_dump() for point in result.points)


@app.post(
    "/experiments/poisoned-rag/benchmark",
    response_model=PoisonedRAGBenchmarkResponse,
)
async def run_poisoned_rag_benchmark(
    request: PoisonedRAGBenchmarkRequest,
) -> PoisonedRAGBenchmarkResponse:
    experiment_id = f"poisonedrag-{uuid4().hex[:12]}"
    defenses = list(dict.fromkeys(request.retrieval_defenses))
    runs: list[PoisonedRAGResponse] = []
    failures: list[PoisonedRAGRunFailure] = []
    for defense in defenses:
        for scenario in request.scenarios:
            for repetition in range(1, request.repetitions + 1):
                fixed_candidates: list[PoisonedRAGCandidate] | None = None
                if request.fixed_poison_pool and max(request.poison_counts) > 0:
                    pool_started = perf_counter()
                    try:
                        pool_run = await run_poisoned_rag_experiment(
                            PoisonedRAGRequest(
                                query=scenario.query,
                                expected_answer=scenario.expected_answer,
                                attack_target=scenario.attack_target,
                                poison_count=max(request.poison_counts),
                                top_k=request.top_k,
                                max_generation_trials=request.max_generation_trials,
                                passage_word_count=request.passage_word_count,
                                generation_temperature=request.generation_temperature,
                                cleanup_before_run=True,
                                retrieval_defense=defense,
                                candidate_multiplier=request.candidate_multiplier,
                            )
                        )
                    except Exception as exc:
                        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                        for poison_count in request.poison_counts:
                            failures.append(PoisonedRAGRunFailure(
                                scenario_name=scenario.name,
                                retrieval_defense=defense,
                                poison_count=poison_count,
                                repetition=repetition,
                                stage="generation",
                                error_type=type(exc).__name__,
                                detail=str(detail)[:1000],
                                elapsed_seconds=round(perf_counter() - pool_started, 3),
                            ))
                        continue
                    fixed_candidates = sorted(
                        (item for item in pool_run.generated_documents if item.selected),
                        key=lambda item: (-(item.selection_score or 0.0), item.document_id),
                    )
                for poison_count in request.poison_counts:
                    attempt_started = perf_counter()
                    try:
                        run = await run_poisoned_rag_experiment(
                            PoisonedRAGRequest(
                                query=scenario.query,
                                expected_answer=scenario.expected_answer,
                                attack_target=scenario.attack_target,
                                poison_count=poison_count,
                                top_k=request.top_k,
                                max_generation_trials=request.max_generation_trials,
                                passage_word_count=request.passage_word_count,
                                generation_temperature=request.generation_temperature,
                                cleanup_before_run=True,
                                retrieval_defense=defense,
                                candidate_multiplier=(1 if fixed_candidates is not None else request.candidate_multiplier),
                                fixed_candidates=fixed_candidates,
                            )
                        )
                    except Exception as exc:
                        detail = (
                            exc.detail
                            if isinstance(exc, HTTPException)
                            else "Unexpected internal error; inspect restricted server logs."
                        )
                        detail_text = str(detail)[:1000]
                        stage = "generation" if "generation failed" in detail_text.casefold() else "unknown"
                        failures.append(
                            PoisonedRAGRunFailure(
                                scenario_name=scenario.name,
                                retrieval_defense=defense,
                                poison_count=poison_count,
                                repetition=repetition,
                                stage=stage,
                                error_type=type(exc).__name__,
                                detail=detail_text,
                                elapsed_seconds=round(perf_counter() - attempt_started, 3),
                            )
                        )
                        continue
                    run.scenario_name = scenario.name
                    runs.append(run)


    points: list[PoisonedRAGBenchmarkPoint] = []
    for defense in defenses:
        for poison_count in request.poison_counts:
            selected = [
                run for run in runs
                if run.requested_poison_count == poison_count
                and run.retrieval_defense == defense
            ]
            selected_failures = [
                failure for failure in failures if failure.poison_count == poison_count
                and failure.retrieval_defense == defense
            ]
            points.append(
                PoisonedRAGBenchmarkPoint(
                    retrieval_defense=defense,
                    poison_count=poison_count,
                    trials=len(selected) + len(selected_failures),
                    successful_trials=len(selected),
                    failed_trials=len(selected_failures),
                    attack_success_rate=_mean([run.metrics.attack_success_rate for run in selected]),
                    accuracy=_mean([run.metrics.accuracy for run in selected]),
                    retrieval_attack_success_rate=_mean([run.metrics.retrieval_attack_success_rate for run in selected]),
                    retrieval_success_rate=_mean([run.metrics.retrieval_success_rate for run in selected]),
                    retrieval_precision=_mean([run.metrics.retrieval_precision for run in selected]),
                    retrieval_recall=_mean([run.metrics.retrieval_recall for run in selected]),
                    retrieval_f1=_mean([run.metrics.retrieval_f1 for run in selected]),
                    average_poison_in_top_k=_mean([float(run.metrics.poison_in_top_k) for run in selected]),
                    average_generation_queries=_mean([float(run.metrics.generation_queries) for run in selected]),
                    average_generation_seconds=_mean([run.metrics.generation_seconds for run in selected]),
                    average_total_seconds=_mean([run.metrics.total_seconds for run in selected]),
                )
            )

    result = PoisonedRAGBenchmarkResponse(
        experiment_id=experiment_id,
        model=OLLAMA_MODEL,
        points=points,
        runs=runs,
        failures=failures,
        json_url=f"/experiments/results/{experiment_id}.json",
        csv_url=f"/experiments/results/{experiment_id}.csv",
        fixed_poison_pool=request.fixed_poison_pool,
    )
    _write_benchmark_artifacts(result)
    await reset_experiment_documents()
    return result


@app.post("/experiments/agent-poison", response_model=AgentPoisonResponse)
async def run_agent_poison_experiment(
    request: AgentPoisonRequest,
) -> AgentPoisonResponse:
    """Run an isolated AgentPoison reproduction without mutating Chroma."""
    run_id = f"agentpoison-{uuid4().hex[:12]}"
    records = load_json_documents(AGENT_POISON_CORPUS_FILE)[: request.benign_corpus_limit]
    benign_memories = [
        (str(item["text"]), str(item["text"]), False) for item in records
    ]
    embedding = create_embeddings()
    best, history = optimize_trigger(
        embedding,
        queries=request.train_queries,
        benign_texts=[key for key, _value, _poisoned in benign_memories],
        seed_trigger=request.seed_trigger,
        candidate_tokens=request.candidate_tokens,
        iterations=request.iterations,
        query_batch_size=request.query_batch_size,
    )
    poison_memories = [
        (
            f"{request.train_queries[index % len(request.train_queries)]} {best.trigger}",
            craft_poison_value(
                request.target_action,
                query=request.train_queries[index % len(request.train_queries)],
                style=request.poison_style,
            ),
            True,
        )
        for index in range(request.poison_count)
    ]
    poisoned_memory = benign_memories + poison_memories
    # Embed each corpus once and reuse it for every test query below, instead
    # of letting rank_memory re-embed the (possibly large) benign corpus on
    # every call -- see embed_memory()'s docstring.
    embedded_benign = embed_memory(embedding, benign_memories)
    embedded_poisoned = embedded_benign + embed_memory(embedding, poison_memories)
    trials: list[dict[str, object]] = []
    retrieval_successes = action_successes = end_to_end_successes = preserved = 0
    for query in request.test_queries:
        clean_ranked = rank_embedded_memory(
            embedding, query=query, embedded_memories=embedded_benign, top_k=request.top_k
        )
        clean_under_poison = rank_embedded_memory(
            embedding, query=query, embedded_memories=embedded_poisoned, top_k=request.top_k
        )
        triggered_query = f"{query} {best.trigger}"
        triggered_ranked = rank_embedded_memory(
            embedding, query=triggered_query, embedded_memories=embedded_poisoned, top_k=request.top_k
        )
        retrieved = retrieval_success(triggered_ranked)
        clean_answer = await _answer_with_supplied_context(
            query, "\n\n".join(item[1] for item in clean_ranked)
        )
        clean_poisoned_answer = await _answer_with_supplied_context(
            query, "\n\n".join(item[1] for item in clean_under_poison)
        )
        triggered_answer = await _answer_with_supplied_context(
            triggered_query, "\n\n".join(item[1] for item in triggered_ranked)
        )
        action_succeeded = phrase_adopted(triggered_answer, request.target_action)
        retrieval_successes += int(retrieved)
        action_successes += int(retrieved and action_succeeded)
        end_to_end_successes += int(action_succeeded)
        preserved += int(answers_agree(clean_answer, clean_poisoned_answer))
        trials.append({
            "query": query,
            "triggered_query": triggered_query,
            "retrieval_success": retrieved,
            "poison_in_top_k": sum(item[2] for item in triggered_ranked),
            "action_succeeded": action_succeeded,
            "clean_answer": clean_answer,
            "clean_under_poison_answer": clean_poisoned_answer,
            "triggered_answer": triggered_answer,
            "retrieved": [
                {"key": key, "value": value, "poisoned": poisoned, "score": round(score, 6)}
                for key, value, poisoned, score in triggered_ranked
            ],
        })
    total = len(request.test_queries)
    metrics = AgentPoisonMetrics(
        asr_r=round(retrieval_successes / total, 4),
        asr_a=round(action_successes / retrieval_successes, 4) if retrieval_successes else 0.0,
        asr_t=round(end_to_end_successes / total, 4),
        benign_accuracy=round(preserved / total, 4),
        poison_rate=round(request.poison_count / len(poisoned_memory), 6),
    )
    result = AgentPoisonResponse(
        run_id=run_id,
        trigger=best.trigger,
        objective=best.objective,
        uniqueness=best.uniqueness,
        compactness=best.compactness,
        objective_history=[item.objective for item in history],
        target_action=request.target_action,
        poison_count=request.poison_count,
        corpus_count=len(poisoned_memory),
        metrics=metrics,
        trials=trials,
    )
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_RESULTS_DIR / f"{run_id}.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _write_agent_poison_benchmark_artifacts(
    result: AgentPoisonBenchmarkResponse,
) -> None:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.json"
    csv_path = EXPERIMENT_RESULTS_DIR / f"{result.experiment_id}.csv"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(AgentPoisonBenchmarkPoint.model_fields),
        )
        writer.writeheader()
        writer.writerows(point.model_dump() for point in result.points)


@app.post(
    "/experiments/agent-poison/benchmark",
    response_model=AgentPoisonBenchmarkResponse,
)
async def run_agent_poison_benchmark(
    request: AgentPoisonBenchmarkRequest,
) -> AgentPoisonBenchmarkResponse:
    """Sweep poison_count for a fixed AgentPoison scenario, in-memory only."""
    experiment_id = f"agentpoison-bench-{uuid4().hex[:12]}"
    runs: list[AgentPoisonResponse] = []
    failures: list[AgentPoisonBenchmarkFailure] = []
    for poison_count in request.poison_counts:
        for repetition in range(1, request.repetitions + 1):
            attempt_started = perf_counter()
            try:
                run = await run_agent_poison_experiment(
                    AgentPoisonRequest(
                        train_queries=request.train_queries,
                        test_queries=request.test_queries,
                        target_action=request.target_action,
                        seed_trigger=request.seed_trigger,
                        candidate_tokens=request.candidate_tokens,
                        poison_count=poison_count,
                        top_k=request.top_k,
                        iterations=request.iterations,
                        benign_corpus_limit=request.benign_corpus_limit,
                        query_batch_size=request.query_batch_size,
                        poison_style=request.poison_style,
                    )
                )
            except Exception as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                failures.append(
                    AgentPoisonBenchmarkFailure(
                        poison_count=poison_count,
                        repetition=repetition,
                        error_type=type(exc).__name__,
                        detail=str(detail)[:1000],
                        elapsed_seconds=round(perf_counter() - attempt_started, 3),
                    )
                )
                continue
            runs.append(run)

    points: list[AgentPoisonBenchmarkPoint] = []
    for poison_count in request.poison_counts:
        selected = [run for run in runs if run.poison_count == poison_count]
        selected_failures = [
            failure for failure in failures if failure.poison_count == poison_count
        ]
        points.append(
            AgentPoisonBenchmarkPoint(
                poison_count=poison_count,
                trials=len(selected) + len(selected_failures),
                successful_trials=len(selected),
                failed_trials=len(selected_failures),
                asr_r=_mean([run.metrics.asr_r for run in selected]),
                asr_a=_mean([run.metrics.asr_a for run in selected]),
                asr_t=_mean([run.metrics.asr_t for run in selected]),
                benign_accuracy=_mean([run.metrics.benign_accuracy for run in selected]),
                average_poison_rate=_mean([run.metrics.poison_rate for run in selected]),
            )
        )

    result = AgentPoisonBenchmarkResponse(
        experiment_id=experiment_id,
        model=OLLAMA_MODEL,
        points=points,
        runs=runs,
        failures=failures,
        json_url=f"/experiments/results/{experiment_id}.json",
        csv_url=f"/experiments/results/{experiment_id}.csv",
    )
    _write_agent_poison_benchmark_artifacts(result)
    return result


@app.get("/experiments/results/{filename}")
async def download_experiment_result(filename: str) -> FileResponse:
    if not filename.startswith(("poisonedrag-", "agentpoison-")) or not filename.endswith((".json", ".csv")):
        raise HTTPException(status_code=404, detail="Experiment result not found")
    path = EXPERIMENT_RESULTS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Experiment result not found")
    return FileResponse(path, filename=filename)


async def _delete_experiment_documents(document_ids: list[str]) -> int:
    """Retained for targeted cleanup used by tests and maintenance scripts."""
    if not document_ids:
        return 0
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        responses = await asyncio.gather(
            *[
                _request_json(
                    client,
                    "DELETE",
                    f"{AGENT_URLS['local_db']}/documents/{document_id}",
                )
                for document_id in document_ids
            ],
            return_exceptions=True,
        )
    return sum(
        isinstance(response, dict) and response.get("deleted") is True
        for response in responses
    )
