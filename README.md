# MedEviRAG · 循证医疗问答与证据溯源

> Agent 应用工程师求职作品集项目之一 · 线上可交互快照：https://huobuzhe.github.io/showcase/

## 一句话介绍

一个以「可溯源」为核心的医疗知识库 RAG 系统：检索 → 重排 → 生成 → 引用校验 → 决策。强调证据分级、页码 / 版面定位、证据不足时拒答，并用分层评测 + CI 门禁证明「改动有没有让系统变好」。

## 这个仓库里有什么

- <code>src/medevirag/serve/</code>：FastAPI 服务（<code>routes.py</code> 路由、<code>schemas.py</code> 请求/响应模型、<code>app.py</code> 启动）
- <code>src/medevirag/eval/</code>：分层评测（layers / calibration / gate）与门禁逻辑
- <code>scripts/eval/</code>：run_core.py（跑核心集）、make_label_sheet.py（标注表）、degrade_probe.py（劣化探针）
- <code>tests/eval/</code>：P0 层的评测单测；<code>.github/workflows/eval.yml</code>：CI 回归门禁
- <code>docs/</code>：阶段文档与 <code>docs/evidence/</code>（本机评测报告 JSON，如 p0_degrade_probe.json 的 gate_status=BLOCKED、eval_*_p0base_core_80_k10.json 的 n=80 / llm_mode=fake / pipeline_selfcheck）
- <code>configs/</code>、<code>docker/</code>、<code>web/</code>、<code>requirements.txt</code>、<code>pyproject.toml</code>

## 流水线与关键设计

1. 检索（向量 + 混合检索）→ 2. 重排 → 3. 生成 → 4. 引用校验 → 5. 决策（answered / partial / refused）
- 每条回答带 <code>citations</code>：chunk_id、doc_id、doc_title、section_path、page、quote、url、evidence_level
- 回答带 <code>confidence</code> 与 <code>refusal_reason</code>；证据不足或过滤器空证据时拒答，而不是硬编一个答案
- <code>trace</code> 记录 trace_id、cache（hit / miss_answer / bypass）、retrieval_ms / rerank_ms / generation_ms / total_ms、各节点耗时
- 错误码：INVALID_REQUEST 400、NOT_FOUND 404、BACKEND_UNAVAILABLE 503、LLM_UPSTREAM_ERROR 502、TIMEOUT 504、INTERNAL 500

## 接口一览

| 接口 | 说明 |
|---|---|
| POST /v1/ask | AskRequest{question, top_k, filters, use_cache} → AskResponse{trace, status, answer, citations, confidence, refusal_reason, disclaimer} |
| GET /v1/citations/{chunk_id} | 回链原文：text、context_header、locator、evidence_level |
| POST /v1/feedback | 反馈 trace_id + rating(-1|1) + comment |
| GET /healthz | 存活探测 |
| GET /readyz | 探测 pg / redis / milvus / models，未就绪返回 503 |

## 评测闭环（为何它重要）

招聘方默认候选人会写 Agent，真正想问的是「你怎么证明它变好了」。本项目提供：160 条评测集、LLM-as-judge 三维打分、劣化探针、分层评测与 CI 回归门禁。展示站「评测门禁与性能回归」卡片的数据即来自这里的报告。

诚实说明：报告里的 <code>pipeline_selfcheck</code> / <code>llm_mode=fake</code> 是流水线自检口径，**不是**对系统质量或临床有效性的结论；压测数字只代表本机环境。

## 快速开始

1. 复制 <code>.env.example</code> 为 <code>.env</code> 并配置模型 / 向量库 / 数据库连接。
2. 按 <code>docker/</code> 起依赖（Postgres、Redis、Milvus），准备已索引语料。
3. 安装 <code>requirements.txt</code> 后启动服务（默认端口 8000）。
4. <code>POST /v1/ask</code> 提问，用返回的 chunk_id 调 <code>/v1/citations/{chunk_id}</code> 核对原文。
5. 跑 <code>python scripts/eval/run_core.py</code> 与 CI 门禁。

## 与展示站的关系

https://huobuzhe.github.io/showcase/ 里「循证医疗问答与证据溯源」卡片的 5 个实例（正常回答带 3 条引用、缓存命中、语料外拒答、过滤器空证据拒答、引用回链），来自本项目 2026-11-13 本机真实运行的 <code>/v1/ask</code>；原始报告见 <code>docs/evidence/</code>。展示页是静态快照，不连接本仓库的服务。

## 安全与免责声明

- <code>.env</code>、虚拟环境、缓存、模型权重、原始语料 / 索引均未入库；变量名见 <code>.env.example</code>。
- 仅供学习与工程演示，不构成医疗建议；任何健康问题请咨询专业人士。
