"""What does the redundancy filter cost when there is no attack?

Every measurement so far injected poisons. A defense that is only ever
measured under attack has an unmeasured false-positive rate: if it discards
clean passages on ordinary queries it is not free to deploy, however well it
scores against the attack.
"""
import os
if os.getenv("EMBEDDING_BACKEND") != "ollama":
    raise SystemExit("EMBEDDING_BACKEND=ollama required.")

import asyncio, json, random
from pathlib import Path
from uuid import uuid4

from services.common.chroma_store import ChromaDocumentStore
from services.common.embeddings import create_embeddings
from services.common.ragpart import RagPartConfig
from services.common.cluster_filter import (
    ClusterFilterConfig, cluster, redundant_indices,
)
from services.common.ragmask import candidate_count
from services.orchestrator.rag import (
    build_rag_chain, create_chat_model, format_context,
)
from services.orchestrator.evaluation import evaluate_answer

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5
scen_all = json.loads(Path('datasets/experiments/nq_target_queries.json').read_text())
corpus = json.loads(Path('datasets/generated/nq_100000.json').read_text())
by_id = {d['id']: d for d in corpus}

# 오염문 8개 시나리오와 겹치지 않는, 더 넓은 표본으로 무공격 비용을 본다.
scen = [s for s in scen_all if [i for i in s['relevant_document_ids'] if i in by_id]][:20]
keep = {i for s in scen for i in s['relevant_document_ids'] if i in by_id}
random.seed(0)
subset = [by_id[i] for i in keep] + random.sample(
    [d for d in corpus if d['id'] not in keep], 400)
work = Path('datasets/generated/_clean.json'); work.write_text(json.dumps(subset))

emb = create_embeddings()
store = ChromaDocumentStore(work, collection_name=f"clean-{uuid4().hex[:8]}",
                            embedding=emb, ragpart=RagPartConfig(enabled=False),
                            cluster_filter=ClusterFilterConfig())
cfg = ClusterFilterConfig()
model = create_chat_model(model="qwen3:8b", base_url="http://localhost:11434",
                          temperature=0.0, num_predict=200)
chain = build_rag_chain(model, mode="vulnerable")

async def run():
    dropped_total = fired = 0
    stats = {}
    for defense in ("none", "cluster"):
        gold_hits = correct = 0
        for s in scen:
            gold = set(s['relevant_document_ids'])
            if defense == "cluster":
                hits = store.search_cluster(s['query'], TOP_K)
            else:
                hits = store.search(s['query'], TOP_K)
            gold_hits += any(h.document_id in gold for h in hits)
            answer = await chain.ainvoke(
                {"question": s['query'], "context": format_context(hits)})
            outcome, expected_present, _ = evaluate_answer(
                answer, expected_answer=s['expected_answer'],
                attack_target=s['attack_target'])
            correct += expected_present
        stats[defense] = (gold_hits, correct)
        print(f"{defense:8s} 정답문서 top-5 포함 {gold_hits}/{len(scen)}  "
              f"정답 언급 {correct}/{len(scen)}", flush=True)

    # 필터가 무공격 상태에서 실제로 무엇을 버리는가
    for s in scen:
        cands = store.search(s['query'], candidate_count(TOP_K, cfg.overfetch))
        vecs = emb.embed_documents([h.text for h in cands])
        groups = cluster(vecs, cfg.similarity_threshold)
        dropped = redundant_indices(groups, cfg.min_cluster_size)
        dropped_total += len(dropped)
        fired += bool(dropped)
    print(f"\n무공격 후보 {candidate_count(TOP_K, cfg.overfetch)}개 중 "
          f"평균 {dropped_total/len(scen):.2f}개 제거, "
          f"{fired}/{len(scen)} 질의에서 필터가 발동")
    work.unlink()

asyncio.run(run())
