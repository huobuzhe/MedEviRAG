# 统一评测看板（P3）

生成时间：2026-11-12T11:25:24+00:00

## 各业务本次指标

| target | version | case_count | success_rate | error_rate | tokens | p95_latency_ms |
|---|---|---|---|---|---|---|
| ai_chef_agent | v3.0-stage10 | 12 | 0.5 | 0.5 | 2360 | 0.0554 |
| medevirag_rag | v0.3-eval | 12 | 0.0 | 1.0 | 1560 | 0.0057 |

## 版本对比（统一基线）

| target | baseline | current | gate | flags |
|---|---|---|---|---|
| ai_chef_agent | ai_chef_agent_20261112T112524_baseline | ai_chef_agent_20261112T112524_current | BLOCKED | SUCCESS_DROP success_rate: 1.0 -> 0.5; ERROR_RISE error_rate: 0.3333 -> 0.5 |
| medevirag_rag | medevirag_rag_20261112T112524_baseline | medevirag_rag_20261112T112524_current | BLOCKED | SUCCESS_DROP success_rate: 1.0 -> 0.0; ERROR_RISE error_rate: 0.0 -> 1.0 |

## bad case 回流

- 候选样本数：18
- 文件：`F:\MedEviRAG\data\eval\layers\reflow_candidates.jsonl`
- 隔离状态：`quarantine=True`，须人工审核后再冻结，禁止直接当 few-shot（防污染）

> 说明：两个业务共用同一套 EvalRecord 字段、同一份基线、同一套门禁与同一张看板；
> 这就是「复用/服务化」的中台思维，但项目命名仍不叫中台。
