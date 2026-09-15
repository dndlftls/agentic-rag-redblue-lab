# 선행연구 조사 — PoisonedRAG 방어

조사일 **2026-09-15**. 이 문서는 다음 조사가 반복이 아니라 확장이 되도록 검색어,
색인, 확인 수준을 함께 기록한다.

> **먼저 결론 — 이 랩의 중복성 필터가 내세운 차별점은 선행연구에 선점되어 있다.**
> `final-report.ko.md` §7은 선행연구로 TrustRAG만 들고 "군집 판정 방식(k=2 고정 → 절대
> 임계값)"과 "오탐률 정량화"를 기여로 적었다. 그러나
> **RAGDefender(ACSAC 2025)** 가 이미 검색된 패시지에 계층적 응집 군집화를 쓰고, 같은
> 공격·같은 데이터·같은 top-5에서 무공격 비용을 보고했다. **SeCon-RAG(NeurIPS 2025)** 는
> "촘촘히 뭉친 문서 = 오염" 가설에 절대 유사도 임계값을 쓴다. 이 랩의 구체적 조합
> (질의별 · 평균연결 · 쌍 유사도 절대 임계값 · 크기 2 이상 군집 통째 제거)과 완전히
> 같은 논문은 찾지 못했지만, **각 구성요소는 모두 존재한다.** 해당 절은 고쳐야 한다.

---

## 1. 중복 (Duplicates)

### RAGDefender — 중복성 필터와 부분 중복

- **인용**: Minseok Kim, Hankook Lee, Hyungjoon Koo. *Rescuing the Unpoisoned: Efficient
  Defense against Knowledge Corruption Attacks on RAG Systems.* **ACSAC 2025.**
- **URL**: https://arxiv.org/abs/2511.01268 · 코드 https://github.com/SecAI-Lab/RAGDefender
- **게재 근거**: arXiv 초록 페이지 "To appear in the Proceedings of the 2025 ACSAC" +
  dblp `conf/acsac/acsac2025` 목록에서 제목 확인 (둘 다 직접 열어봄)
- **방법 (전문 확인)**: 검색 후 단계에서 동작, LLM 호출 없음.
  "we adopt a hierarchical agglomerative clustering to partition the embeddings of
  retrieved passages." 두 군집으로 나눈 뒤 TF-IDF로 오염 개수 `N_adv`를 추정하고,
  고유사도 쌍에 자주 등장한 패시지 상위 `N_adv`개를 제거한다. 고정 임계값이 아니라
  순위 기반이다.
- **평가**: PoisonedRAG · GARAG · Tan et al., **NQ에서 k=5**, 오염 비율 1×/2×/4×/6×.
  무공격 시 "Gemini shows only a 2% drop in accuracy, while other language models exhibit
  no measurable change"; 오분할 0.54%.
- **이 랩과의 관계**: **공격 · 데이터셋 · top-k · 알고리즘 계열 · 적용 단계 · 무공격 비용
  보고가 모두 같다.** 다른 것은 결정 규칙 하나 — RAGDefender는 오염 개수를 추정해 상위
  N개를 버리고, 이 랩은 쌍 유사도 0.90 절대 임계값을 넘는 군집을 통째로 버린다.

### SeCon-RAG — 중복성 가설 + 충돌 탐지와 부분 중복

- **인용**: *SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for
  Trustworthy RAG.* **NeurIPS 2025.**
- **URL**: https://arxiv.org/abs/2510.09710 · https://neurips.cc/virtual/2025/poster/115589
- **게재 근거**: dblp `conf/nips/neurips2025` 목록(5,823편 전수 중)에서 제목 확인 +
  NeurIPS 포스터 페이지
- **방법 (전문 확인)**: 1단계는 **질의 시점이 아니라 말뭉치 전체에 오프라인으로**
  K-means를 돌려 "documents that cluster too tightly around a centroid" 를 오염으로 보고
  중심 유사도 **절대 임계값 τ_cluster**로 거른다. 2단계는 LLM이 질의·말뭉치·모델 내부
  지식 일관성을 판정하는 충돌 인지 필터(질의당 LLM 호출 3회 이상).
- **이 랩과의 관계**: 1단계는 중복성 필터와 **같은 가설 + 절대 임계값**(적용 대상과
  시점은 다름). 2단계는 폐기한 `isolate_conflict`와 같은 계열이며, **검색 결과가 전부
  오염돼 서로 일치하는 경우를 다루지 않는다** — 이 랩이 측정으로 확인한 그 사각지대.

---

## 2. 기반 — 이 랩이 적용을 검토할 방어

이 랩에 남은 공백은 **오염문 1개 구간(공격 성공률 60~70%)** 이다. 중복성 필터는 원리적으로
닿지 않는다. 아래는 그 구간을 다룰 수 있는지를 기준으로 정리했다.

| 방어 | 게재 | 단일 오염문 | 모델 내부 필요 | 이 랩 적용성 |
| --- | --- | --- | --- | --- |
| **AV Filter** | **ICML 2026** | **주 평가 설정** | 어텐션 (보조 모델 가능) | **1순위** |
| SDAG | 프리프린트 | 명시 평가 | 어텐션 마스크 | 2순위 |
| RevPRAG | Findings EMNLP 2025 | 1~5개 절제 | 활성값 + 학습 필요 | 탐지만 |
| Astute RAG | ACL 2025 | 공격 특화 아님 | 없음 | 인접 |
| RAGDefender | ACSAC 2025 | 1× 비율 평가 | 없음 | 중복 (위) |

### AV Filter — 1순위 후보

- **인용**: Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Krishnamurthy Dj Dvijotham,
  Somesh Jha. *Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG.*
- **URL**: https://arxiv.org/abs/2506.04390 · https://openreview.net/forum?id=PS43wqCSME
- **게재 근거**: arXiv 초록 페이지 "Accepted at ICML 2026". **dblp `conf/icml/icml2026`은
  아직 없음(404)이라 색인으로는 재확인 못 함.**
- **방법 (전문 확인)**: 패시지별 어텐션 점수(NPAS)를 계산하고 분산 기반 이상치 탐지로
  제거. "iteratively removes the highest-scoring passage until either the NPAS variance
  drops below δ or an ϵ-fraction of passages have been removed".
- **단일 오염문**: 실험 설정 ϵ=0.1, k=10 → **오염문 정확히 1개**가 주 평가 조건.
  ASR 2.4~7.2%. **주의: 초록 요약 한 번은 "단일 패시지 공격이 아니다"로 읽었으나 전문은
  정반대였다.** 근접 논문을 전문으로 읽어야 하는 이유의 실례.
- **이 랩 적용성**: Ollama는 어텐션을 노출하지 않는다. 그러나 논문이
  "For GPT-4o, which lacks accessible internals, we use Mistral-7B as an auxiliary model"
  라고 하므로, **생성은 Ollama로 두고 어텐션 계산만 별도 오픈 모델(HF transformers)로**
  할 수 있다.
- **중복성 필터와의 관계 (추론, 미측정)**: 분산 기반 이상치 탐지는 이상치가 있어야
  작동하므로, 전부 오염된 경우엔 약할 것이다. 반대로 중복성 필터는 1개일 때 무력하다.
  **두 방어가 서로의 사각지대를 덮을 가능성**이 있으나 측정 전까지는 가설이다.
- **한계**: "adaptive attacks can evade AV Filter, achieving an attack success rate up to
  35%" (단, 질의당 최대 10⁴초의 최적화와 모델 전체 접근 필요).

### SDAG — 2순위, 격리의 저비용 구현

- **인용**: Sagie Dekel, Moshe Tennenholtz, Oren Kurland. *Addressing Corpus Knowledge
  Poisoning Attacks on RAG Using Sparse Attention.* arXiv:2602.04711 (v3 2026-08-26).
- **게재**: **확인된 게재지 없음 (프리프린트).**
- **방법 (전문 확인)**: 서로 다른 검색 문서 간 어텐션만 차단, 질의는 모든 문서를 봄.
  이 랩이 권장하는 **격리(`isolation`)를 LLM 호출 k번이 아니라 1번으로** 구현하는 셈.
  RobustRAG를 인용·비교하지 않는다.
- **단일 오염문**: 명시적으로 평가 — k=5, NQ에서 ASR 0.17 (기준선 CARG 0.41).
  오염문이 다수·전부인 경우는 평가하지 않음.
- **적용성**: 어텐션 마스크 제어 필요 → Ollama로 불가, HF transformers 필요.

### RevPRAG — 탐지 전용

- **인용**: Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai. *RevPRAG:
  Revealing Poisoning Attacks in RAG through LLM Activation Analysis.* **Findings of EMNLP
  2025**, pp. 12999–13011.
- **URL**: https://aclanthology.org/2025.findings-emnlp.698/
- **게재 근거**: ACL Anthology 인용 블록 직접 확인 (ID 2025.findings-emnlp.698)
- **방법 (전문 확인)**: 마지막 토큰의 전 층 활성값으로 분류기 학습. 데이터셋마다 정상
  1,500 + 오염 1,500 필요. **답변을 고치지 않고 표시만** 한다.
  오픈 모델만 평가(GPT2-XL, Llama2-7B/13B, Mistral-7B, Llama3-8B). 오염문 1~5개 절제.
- **적용성**: 학습 데이터 구축 비용이 크고 데이터셋 간 전이 결과가 없음.

### Astute RAG — 인접 (공격 특화 아님)

- **인용**: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan O Arik. *Astute RAG.*
  **ACL 2025 (Long)**, pp. 30553–30571. https://aclanthology.org/2025.acl-long.1476/
- 모델 내부 지식과 외부 문서를 출처를 구분해 통합. 초록 기준 **의도적 오염이 아니라
  일반적인 불완전 검색**을 다룬다. 단일 오염문이 모델이 아는 사실과 충돌할 때 이론상
  유리하나, 공격 평가가 없어 근거로 쓰기 어렵다.

### 사후 추적 (방어가 아니라 포렌식)

| 논문 | 게재 | 내용 |
| --- | --- | --- |
| RAGForensics (Baolei Zhang 외) | **WWW 2025** ([ACM DL](https://dl.acm.org/doi/10.1145/3696410.3714756)) | LLM 프롬프트로 DB 내 오염문 역추적 |
| RAGOrigin — *Who Taught the Lie?* (Baolei Zhang 외) | **IEEE S&P 2026** (dblp 확인) | 블랙박스 책임 귀속, 비지도 군집으로 오염문 격리 |
| AttnTrace (Yanting Wang, Runpeng Geng, Ying Chen, Jinyuan Jia) | **IEEE S&P 2026** (dblp 확인) | 어텐션 기반 문맥 귀속, 질의당 약 10초. 탐지기와 결합하는 용법 제시 |

---

## 3. 동기 — 문제가 실재한다는 근거

- **One Shot Dominance** (Zhiyuan Chang 외), **Findings of EMNLP 2025**, pp. 18811–18825.
  https://aclanthology.org/2025.findings-emnlp.1023/ — 초록이 기존 공격을 이렇게 평가한다:
  "they either require injecting multiple poisoned documents (resulting in poor
  stealthiness)". **공격 연구가 다수 오염문 방식을 탐지되기 쉽다고 보고 단일 문서로
  옮겨가고 있다.** 중복성 필터의 전제를 정면으로 겨냥하는 흐름이다.
- **DenialRAG** (arXiv:2608.02678, 프리프린트) — 정답을 명시하고 부정하는 단일 문서 공격.
- **Benchmarking Poisoning Attacks against RAG** (Baolei Zhang 외, arXiv:2505.18543,
  프리프린트) — 공격 13종 × 방어 7종, "current defense techniques fail to provide robust
  protection".
- **OWASP Top 10 for LLM 2025, LLM08 Vector and Embedding Weaknesses**
  (https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/) — 원문 확인.
  완화책은 접근 통제 · 출처 인증 · 데이터 검토 · 로깅 4가지로, **군집·어텐션 같은 탐지
  기법은 언급하지 않는다.** 실무 표준이 이 공격면을 탐지 차원에서 다루지 않는다는 근거.
- MITRE ATLAS의 RAG 오염 기법 페이지는 추정한 URL이 404라 **확인하지 못했다.**

---

## 4. 확인한 공백과 조사 범위

### 학회 목록 전수 확인 (dblp, 실제 브라우저)

`li.entry` 수에서 볼륨 레코드(`editor` 클래스) 1개를 뺀 **논문 수**를 보고한다. 필터는
**제목** 정규식이므로, 제목에 RAG·retrieval·poison 계열 단어가 없는 방어 논문은 놓칠 수
있다.

| 학회 | 전수 | RAG 오염 방어 | 그 밖의 관련 |
| --- | --- | --- | --- |
| USENIX Security 2025 | 439 | 0 | 공격 3 (PoisonedRAG, Topic-FlipRAG, Machine Against the RAG) |
| IEEE S&P 2025 | 255 | 0 | — |
| IEEE S&P 2026 | 252 | 0 | 귀속 2 (RAGOrigin, AttnTrace) |
| ACM CCS 2025 | 395 | 0 | RAG 관련 6편, 모두 공격·멤버십 추론·워터마크·코드 RAG |
| NDSS 2025 | 211 | 0 | — |
| NDSS 2026 | 265 | 0 | 시맨틱 캐시 오염 (LLM 캐시, RAG 말뭉치 아님) |
| **ACSAC 2025** | 84 | **1 (RAGDefender)** | — |
| SaTML 2025 | 53 | 0 | — |
| EuroS&P 2025 / 2026 | 60 / 82 | 0 / 0 | — |
| ICML 2025 | 3,257 | 0 | 공격 1 (PoisonedEye, VLM) |
| **NeurIPS 2025** | 5,823 | **2 (ReliabilityRAG, SeCon-RAG)** | 공격 1, 데이터셋 1 |
| ICLR 2025 | 3,704 | 0 | 문맥 가지치기 1 (Provence, 인접) |

- 사용한 정규식(보안 학회):
  `/(\bRAG\b|retrieval[- ]augmented|knowledge (corruption|poison)|corpus poison|poison\w*.*(retriev|RAG|knowledge|LLM|agent)|(retriev|RAG|knowledge|LLM|agent).*poison|agent memory|memory poison)/i`
- 사용한 정규식(ML 학회):
  `/(\bRAG\b|retrieval[- ]augmented).*(poison|corrupt|attack|defen|robust|secur|adversar|trust)|(poison|corrupt|defen|robust|adversar).*(\bRAG\b|retrieval[- ]augmented)|knowledge (corruption|poison)|corpus poison|memory poison|agent memory/i`
- 보안 학회 스윕 중 도구 출력이 7번째 이후 학회명을 가렸다. **배열 순서로 대응시켰다.**

### 전수 확인하지 못한 곳

- **dblp에 아직 없음(404)**: USENIX Security 2026, SaTML 2026, ICML 2026
- **dblp가 연속 요청 후 응답을 멈춤(HEAD 요청도 45초 초과)**: ICLR 2026, ACL 2025,
  EMNLP 2025, NAACL 2025, WWW 2025, KDD 2025 → `site:aclanthology.org`,
  `site:openreview.net` 검색으로만 **훑었다(scan). 전수 확인 아님.**
- 비영어권 학회 색인은 조사하지 않았다.

### 사용한 웹 검색어 (2026-09-15)

1. `PoisonedRAG defense detect poisoned passages 2025 2026 knowledge corruption`
2. `RevPRAG detecting poisoned RAG responses LLM activations`
3. `traceback poisoning attacks retrieval-augmented generation RAGForensics RAGOrigin`
4. `Astute RAG knowledge conflict internal external imperfect retrieval`
5. `RAGDefender lightweight classifier poisoned passages retrieval augmented generation`
6. `attention variance filter poisoned passage RAG defense certified`
7. `FilterRAG ML-FilterRAG defending knowledge poisoning attacks retrieval augmented generation`
8. `LLM agent memory poisoning defense AgentPoison A-MemGuard 2025`
9. `"retrieval-augmented generation" poisoning defense NDSS 2026 OR "IEEE S&P 2026" OR "CCS 2025" accepted paper`
10. `site:aclanthology.org 2025 retrieval-augmented generation poisoning defense knowledge corruption`
11. `site:openreview.net ICLR 2026 RAG poisoning defense corpus poisoning robust`
12. `sparse attention defense corpus knowledge poisoning retrieval-augmented generation`
13. `defense single poisoned document RAG "one" poisoned passage detection without redundancy 2026`

### 초록만 확인한 장기 꼬리 (프리프린트)

게재지를 확인하지 못했거나 게재지가 없는 것들이다. 동료심사 근거로 쓰지 말 것.

| 논문 | arXiv | 내용 |
| --- | --- | --- |
| FilterRAG / ML-FilterRAG | 2508.02835 | 블랙박스 통계 속성(Freq-Density)으로 필터링 |
| A-MemGuard | 2510.02373 | 에이전트 메모리, 병렬 추론 경로 합의 기반. **검색 요약은 "게재됨"이라 했으나 arXiv에 게재 표기 없음** |
| PRA-RAG | 2607.00012 | 증명 가능한 견고 집계. RevPRAG와 저자 5명 겹침 |
| RAGuard | 2607.26339 | 다층 방어 (검색 확장 + 퍼플렉시티 필터) |
| RAGShield | 2604.00387 | 출처 검증 심층 방어 (정부 RAG) |
| ProGRank | 2603.22934 | 프로브 기울기 재순위화, 검색기 기울기 필요 |
| MEMSAD / MemAudit | 2605.03482 / 2605.23723 | 에이전트 메모리 오염 탐지·감사 |
| Secure RAG against Poisoning | 2510.25025 | 미확인 |

---

## 5. 이 랩 문서에 대한 영향

1. **`final-report.ko.md` §7과 `project-summary.ko.md` 한계절**: 선행연구가 TrustRAG
   하나로 적혀 있다. RAGDefender와 SeCon-RAG를 추가하고, "절대 임계값"과 "오탐률
   정량화"를 기여로 내세운 문장을 고쳐야 한다.
2. **ReliabilityRAG(NeurIPS 2025)** 는 `robustrag.ko.md`에서 **구현 없이 논증으로
   기각**되어 있다. RobustRAG에서 같은 방식의 기각이 측정으로 뒤집혔으므로, 이것도
   재보기 전까지는 "기각"이 아니라 "미측정"으로 적는 게 맞다.
3. **다음 방어 후보는 AV Filter**다. 게재지(ICML 2026)가 가장 강하고, 이 랩의 공백인
   단일 오염문이 주 평가 설정이며, 보조 오픈 모델로 어텐션을 계산할 수 있어 Ollama 스택과
   공존 가능하다.
