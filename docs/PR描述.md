# PR 描述：把 MedEviRAG（循证医疗问答与证据溯源） 完整工程导入 GitHub

> 本文件同时作为该仓库首个 Pull Request 的描述，便于招聘方一眼看清「改了什么、为什么、怎么验证」。

## 这个 PR 做了什么

把本地开发完成的 MedEviRAG（循证医疗问答与证据溯源） 工程导入此仓库：源码、接口契约、阶段文档、评测 / 压测证据与 CI 配置一次性纳入版本管理，作为 Agent 应用工程师求职作品集的可核查证据。

## 为什么

招聘方能打开、能读、能复现的工程证据，比截图和口头描述更有说服力；尤其是「评测怎么做」「数据从哪来」「服务怎么起」这三件事，需要落在仓库里而不是聊天记录里。

## 包含内容

- RAG 服务 src/medevirag/serve/（/v1/ask、/v1/citations、/v1/feedback、/healthz、/readyz）
- 分层评测 src/medevirag/eval/、评测脚本 scripts/eval/、P0 单测与 CI 回归门禁 workflow
- docs/ 阶段文档与 docs/evidence/ 本机评测报告 JSON（含 gate_status=BLOCKED 的劣化探针）
- configs/、docker/、web/、requirements.txt、pyproject.toml 与 .env.example

## 不包含（安全与体积）

- <code>.env</code> 与任何真实密钥、令牌、口令
- 虚拟环境（.venv / .venv3）、Python / 测试缓存、本机日志
- 模型权重、大体积语料与向量索引（体积大且可再生）
- 本机临时文件与个人笔记

但保留了 <code>.env.example</code> 与文档，别人可以按说明自行复现环境。

## 如何验证

1. 复制 .env.example 为 .env 并配置模型与向量库 / 数据库连接
2. 按 docker/ 起 Postgres / Redis / Milvus 并准备已索引语料，启动服务（默认端口 8000）
3. POST /v1/ask 后用返回的 chunk_id 调 GET /v1/citations/{chunk_id} 核对原文与页码
4. 运行 python scripts/eval/run_core.py 与 CI 门禁，查看 docs/evidence/ 报告
5. 打开 https://huobuzhe.github.io/showcase/ 查看「循证医疗问答与证据溯源」与「评测门禁与性能回归」卡片

## 数据来源与诚实说明

展示站 https://huobuzhe.github.io/showcase/ 的 15 条运行实例来自 2026-11-13 本机真实运行，本仓库对应其中一部分；评测报告里的 <code>pipeline_selfcheck</code> / <code>llm_mode=fake</code> 只是自检口径，不是系统质量结论，压测数字只代表本机环境。静态展示页不连接后端。

## 关联链接

- 线上作品集快照：https://huobuzhe.github.io/showcase/
- 展示站源码：https://github.com/huobuzhe/showcase
- 本仓库：https://github.com/huobuzhe/MedEviRAG
- 同系列另一个项目 AI私厨：https://github.com/huobuzhe/ai_chief
