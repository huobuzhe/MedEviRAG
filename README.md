# MedEviRAG · 循证医疗问答与证据溯源

> Agent 应用工程师求职作品集项目之一 · [在线功能快照](https://huobuzhe.github.io/showcase/) · [本仓库](https://github.com/huobuzhe/MedEviRAG) · [导入 PR #1](https://github.com/huobuzhe/MedEviRAG/pull/1) · [USAGE.md（更详细的命令手册）](USAGE.md)

## 0. 一分钟了解

一个以「可溯源」为核心的中文医学指南 RAG 系统：检索 → 重排 → 生成 → 引用校验 → 决策。强调证据分级、页码 / 版面定位、证据不足时拒答，并用分层评测 + CI 门禁证明「改动有没有让系统变好」。

**默认零 API 消耗即可跑通全链路**：没有 LLM Key 时答案是链路自检桩（<code>llm_mode=fake-deterministic</code>），用来验证检索、引用、拒答、评测这些工程环节是否正常工作，**不具医学意义**。

### 能做什么

- 输入临床问题，从本地中文指南语料检索证据，生成带 <code>[C1]</code> 编号引用的回答。
- 每条引用可回链到指南原文：文档名、章节路径、页码、版面 bbox 坐标、证据等级（12 个字段）。
- 两级引用校验：模型编造的引用标记会被拦截，不带病输出。
- 安全拒答：越界 / 时效 / 个体化用药 / 证据不足四类情形明确拒答并给就医指引，而不是硬编答案。
- 混合检索：向量（bge-m3）+ BM25 双路召回、RRF 融合、元数据硬过滤（年份 / 语言 / 文献类型 / 证据等级）。
- 置信度四分量（排序分 / 引用覆盖率 / 证据等级 / 区分度），低于阈值自动降级为 partial 或拒答。
- 评测闭环：160 题分层评测集、三维 LLM judge、劣化探针、回归门禁、CI 工作流。
- Web 前端（Vite）：问答、点引用看原文卡片、反馈评分。

### 不做什么（边界）

- 不开处方、不给个体化剂量；不答急诊（引导拨 120）；语料外问题明确说「没有证据」。
- 报告里的 <code>pipeline_selfcheck</code> / <code>llm_mode=fake</code> 是流水线自检口径，**不是**系统质量或临床有效性结论。
- 仓库不含 <code>.env</code>、虚拟环境、模型权重、原始语料 / 索引、缓存，需要自己准备。

## 1. 项目功能详解（用这些能力做了什么）

### 1.1 八节点 RAG 流水线
<code>src/medevirag/</code> 用 LangGraph 串起 route → retrieve → rerank → generate → citation-check → decision 等节点；每次请求返回 <code>trace</code>（trace_id、cache hit / miss_answer / bypass、retrieval_ms / rerank_ms / generation_ms / total_ms、各节点耗时），便于定位慢在哪一段。

### 1.2 可溯源引用
<code>AskResponse.citations</code> 每项含 chunk_id / doc_id / doc_title / section_path / page / quote / url / evidence_level；<code>GET /v1/citations/{chunk_id}</code> 返回 12 字段详情（含 <code>text</code> 原文全文、<code>context_header</code>、<code>locator</code> 的 page 与 bbox、<code>evidence_level</code>）。前端点引用即可展开原文与版面位置。

### 1.3 拒答与置信度
<code>status</code> 取 answered / partial / refused，附 <code>refusal_reason</code> 与 <code>disclaimer</code>；<code>confidence</code> 是四分量复合分，低于阈值降级。规则夹具矩阵（<code>cli refusal-matrix</code>）覆盖 54 条拒答情形。

### 1.4 检索与索引
PyMuPDF 解析 PDF → 章节感知切分（<code>cli chunk</code>）→ 向量化 + Milvus Lite + BM25 三建（<code>cli index</code>）；查询时向量 + BM25 双路召回、RRF 融合、元数据硬过滤、bge-reranker 重排（<code>configs/retrieval.yaml</code> 里 <code>enabled:false</code> 可关重排做消融）。

### 1.5 分层评测与 CI 门禁
<code>src/medevirag/eval/</code>（layers / calibration / gate）、<code>scripts/eval/</code>（run_core.py、run_p1.py、run_p2.py、run_p3.py、make_label_sheet.py、degrade_probe.py）、<code>tests/eval/</code> 的 P0 单测、<code>.github/workflows/eval.yml</code> 的 CI 回归门禁：160 题冻结评测集、LLM-as-judge 三维打分、指标跌超阈值非零退出。原始报告在 <code>docs/evidence/</code>，例如 <code>p0_degrade_probe.json</code> 的 <code>gate_status=BLOCKED</code>、<code>eval_*_p0base_core_80_k10.json</code> 的 <code>n=80 / llm_mode=fake / pipeline_selfcheck</code>。

### 1.6 服务与前端
<code>src/medevirag/serve/app.py</code> 装配 FastAPI（trace 中间件、异常处理器、CORS），<code>/docs</code> 自带交互式文档；<code>web/</code> 是 Vite 前端，问答、引用卡片、查看原文、反馈（落 <code>data/logs/feedback.jsonl</code>）。

## 2. 知识点与技术栈（以及用它们做了什么）

| 知识点 / 技术 | 在项目里的位置 | 用来做了什么 |
|---|---|---|
| Python 3.10、FastAPI 0.115、Uvicorn 0.31 | <code>src/medevirag/serve/</code>、<code>requirements.txt</code> | REST API（端口 8000）、trace 中间件、统一错误码 |
| LangGraph 0.2.35 | <code>src/medevirag/</code> 流水线 | 把检索 / 重排 / 生成 / 校验 / 决策串成可观测的状态图 |
| sentence-transformers + bge-m3 / bge-reranker | <code>requirements.txt</code>、<code>configs/models.yaml</code> | 向量化与重排；<code>low_cpu_mem_usage=True</code> 避免本机 fp32 爆内存 |
| FAISS / Milvus Lite / pymilvus、BM25（rank-bm25）、RRF | <code>src/medevirag/retrieval/</code> 相关代码、<code>scripts/s2_*.py</code> | 双路召回与融合、索引构建与消融实验 |
| PyMuPDF | 语料解析 | 抽取 PDF 文本、章节树、页码与 bbox 版面坐标，支撑引用回链 |
| Redis / PostgreSQL（psycopg） | <code>.env.example</code> 的 REDIS_URL / PG_DSN、<code>docker/docker-compose.yml</code> | 缓存、反馈与元数据存储 |
| pydantic-settings / structlog / tenacity / orjson | <code>src/medevirag/common/</code> | 配置、结构化日志、重试、快速 JSON |
| LLM-as-judge、ECE 校准、劣化探针、回归门禁 | <code>src/medevirag/eval/</code>、<code>configs/eval.yaml</code> | 证明「改动有没有让系统变好」，而不是凭感觉 |
| pytest / ruff / mypy | <code>pyproject.toml</code>、<code>tests/</code> | 单测、lint、严格类型检查 |
| Vite + Node.js 前端 | <code>web/</code> | 问答界面、引用卡片、原文展开、反馈 |
| Docker Compose（etcd / minio / milvus / postgres / redis / api） | <code>docker/docker-compose.yml</code> | 全栈一键起，healthcheck 决定启动顺序 |
| GitHub Actions | <code>.github/workflows/eval.yml</code> | 评测回归门禁进 CI |

## 3. 目录结构：该打开哪些文件

| 路径 | 是什么 | 什么时候打开 |
|---|---|---|
| <code>USAGE.md</code> | 更详细的使用文档（含全部 CLI 命令与验收口径） | 想查某条命令的确切参数 |
| <code>src/medevirag/serve/routes.py</code>、<code>schemas.py</code> | 路由与请求 / 响应模型 | 调 <code>/v1/ask</code> 前确认字段 |
| <code>src/medevirag/serve/app.py</code> | FastAPI 装配 | 改端口、中间件、CORS |
| <code>src/medevirag/eval/</code> | 分层评测、校准、门禁逻辑 | 理解「怎么证明系统变好」 |
| <code>scripts/eval/</code>、<code>scripts/</code> | run_core / make_label_sheet / degrade_probe / run_p1..p3、accept_s0..s6、loadtest | 跑评测、验收、压测 |
| <code>configs/retrieval.yaml</code> | topk / RRF k / 硬过滤阈值 / 重排开关 | 调检索效果 |
| <code>configs/eval.yaml</code> | 评测阈值唯一真值源、judge 路由、门禁档位 | 调门禁 |
| <code>configs/app.yaml</code>、<code>configs/models.yaml</code> | 服务端口 / 缓存 TTL / 分段耗时预算 / LLM 路由 | 调服务与模型 |
| <code>docs/</code>、<code>docs/evidence/</code> | 阶段文档与本机评测报告 JSON | 看证据与结论 |
| <code>web/</code> | Vite 前端 | 起界面 |
| <code>requirements.txt</code>、<code>.env.example</code> | 依赖与环境变量样例 | 安装与配置 |
| <code>docker/docker-compose.yml</code> | 全栈编排 | 一键起依赖 |
| <code>tests/</code>、<code>pyproject.toml</code> | 单测与工具配置 | 跑测试 / lint |

## 4. 运行前置做法（Prerequisites）

1. **Python 3.10+**（<code>pyproject.toml</code> 的 target-version=py310）；建议建独立虚拟环境 <code>.venv</code>。
2. **Node.js 18+ 与 npm**（仅前端 <code>web/</code> 需要）。
3. **Docker Desktop**（可选）：跑完整 Milvus / Postgres / Redis 全栈；本机只跑 CLI 与 API 时可用 Milvus Lite / FAISS，不强制 Docker。
4. **磁盘约 10GB**：两个本地模型（bge-m3 约 2.2GB + bge-reranker 约 2.2GB）与语料 / 索引。
5. **GPU 可选**：无 GPU 设 <code>MEDEVIRAG_DEVICE=cpu</code>；有 CUDA 可改 <code>cuda</code>。
6. **网络 / API Key 可选**：默认离线桩运行（<code>llm_mode=fake-deterministic</code>，零 API 消耗）。只有接真实生成模型时才把 <code>SILICONFLOW_API_KEY</code> / <code>DASHSCOPE_API_KEY</code> 填进 <code>.env</code>。
7. **端口**：API <code>8000</code>、前端 <code>5173</code>、Milvus <code>19530</code>；确认未被占用。
8. **不要提交 <code>.env</code>**：变量名见 <code>.env.example</code>（SILICONFLOW_API_KEY / DASHSCOPE_API_KEY / MILVUS_URI / PG_DSN / REDIS_URL / HF_HOME）。

## 5. 快速开始（一步一步，命令可直接复制）

### 5.1 克隆并建虚拟环境

~~~bash
git clone https://github.com/huobuzhe/MedEviRAG.git
cd MedEviRAG
python -m venv .venv
~~~

激活（二选一）：

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
source .venv/bin/activate
~~~

安装依赖（<code>requirements.txt</code> 注释明确写的是 pip 口径，不要用 <code>uv sync</code>，会裁剪继承来的 torch 等包）：

~~~bash
pip install -r requirements.txt
~~~

### 5.2 打开 <code>.env.example</code> 复制成 <code>.env</code> 并设环境变量

~~~powershell
Copy-Item .env.example .env
$env:PYTHONPATH = "src"
$env:MEDEVIRAG_DEVICE = "cpu"
$env:TRANSFORMERS_CACHE = "data/.hf_cache"
~~~

~~~bash
cp .env.example .env
export PYTHONPATH=src
export MEDEVIRAG_DEVICE=cpu
export TRANSFORMERS_CACHE=data/.hf_cache
~~~

<code>PYTHONPATH=src</code> 是必须的：包在 <code>src/medevirag</code> 下。若后续要用虚拟环境解释器显式调用，把 <code>python</code> 换成 <code>.venv\Scripts\python.exe</code>（Windows）或 <code>.venv/bin/python</code>（Linux/macOS）。

### 5.3 环境自检

~~~bash
python -X utf8 -m medevirag.cli doctor
~~~

逐项检查 GPU / 语料 / 索引 / 模型；缺哪一步它会点名。

### 5.4 命令行问一个问题（不用起服务也能验证全链路）

~~~bash
python -X utf8 -m medevirag.cli ask "成人高血压的诊断标准是什么"
~~~

输出含 <code>status=answered</code>、<code>answer=... [C1]</code>、引用列表与 <code>trace</code>；无 API key 时开头会标注 <code>llm_mode=fake-deterministic（答案仅为链路自检，不具医学意义）</code>。

### 5.5 查看一条引用的原文

把上一步输出的 <code>chunk_id</code> 填进来（含 <code>#</code> 的 id 必须加引号）：

~~~bash
python -X utf8 -m medevirag.cli citation "0108_135352_975#(一)#c03" --json
~~~

返回 12 个字段：chunk_id / doc_id / doc_title / section_path / chunk_type / page / bbox / block_index / text / context_header / evidence_level / url。

### 5.6 起 API 服务 + 前端（推荐体验方式）

终端 1（端口 8000）：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m uvicorn medevirag.serve.app:app --port 8000 --workers 1
~~~

~~~bash
.venv/bin/python -X utf8 -m uvicorn medevirag.serve.app:app --port 8000 --workers 1
~~~

终端 2（端口 5173）：

~~~bash
cd web
npm install
npm run dev
~~~

浏览器打开 [http://127.0.0.1:5173](http://127.0.0.1:5173) 用界面问答；打开 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 看交互式 API 文档。界面里输入问题 → 点答案里的 <code>[C1]</code> → 右栏引用卡片高亮 → 点「查看原文」展开 chunk 全文与 bbox → 底部可点「有用 / 无用」反馈。

### 5.7 用 curl 直接调 API

~~~bash
curl -X POST http://127.0.0.1:8000/v1/ask -H 'Content-Type: application/json' -d '{"question":"成人高血压的诊断标准是什么","top_k":10}'
curl 'http://127.0.0.1:8000/v1/citations/0108_135352_975%23(%E4%B8%80)%23c03'
curl -X POST http://127.0.0.1:8000/v1/feedback -H 'Content-Type: application/json' -d '{"trace_id":"<上一步返回的 trace_id>","rating":1}'
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
~~~

（Windows PowerShell 把 <code>curl</code> 换成 <code>curl.exe</code>；<code>/readyz</code> 会探测 pg / redis / milvus / models，未就绪返回 503 并点名。）

### 5.8 语料与索引全流程（要换自己的语料时）

~~~bash
python -X utf8 -m medevirag.cli fetch
python -X utf8 -m medevirag.cli parse
python -X utf8 -m medevirag.cli chunk
python -X utf8 -m medevirag.cli report
python -X utf8 -m medevirag.cli index
python -X utf8 -m medevirag.cli retrieve "成人高血压的诊断标准是什么" --topk 10
~~~

分别对应：登记 / 下载指南 PDF → 解析出章节树 → 章节感知切分 → 切分质量报告 → 向量 + Milvus Lite + BM25 三建 → 单条检索调试（看两路分数与来源）。

### 5.9 评测、验收与压测

~~~bash
python -X utf8 -m medevirag.cli refusal-matrix --out data/reports/refusal_matrix.json
python -X utf8 scripts/eval/run_core.py
python -X utf8 scripts/eval/degrade_probe.py
python -X utf8 -m medevirag.eval.runner --split dev
python -X utf8 -m medevirag.eval.regression --baseline <id1> --current <id2>
python -X utf8 scripts/loadtest.py --concurrency 20 --repeat 3
~~~

分阶段验收（PowerShell / bash 二选一）：

~~~powershell
0..6 | ForEach-Object { .\.venv\Scripts\python.exe -X utf8 scripts\accept_s$_.py }
~~~

~~~bash
for s in 0 1 2 3 4 5 6; do python -X utf8 scripts/accept_s$s.py; done
~~~

### 5.10 Docker 全栈（可选）

~~~bash
docker compose -f docker/docker-compose.yml up -d
~~~

按 healthcheck 顺序起 etcd / minio / milvus / postgres / redis / api。仅需要 Milvus 时也可只起对应服务。

## 6. 接口一览

| 接口 | 说明 |
|---|---|
| <code>POST /v1/ask</code> | AskRequest{question, top_k, filters, use_cache} → AskResponse{trace, status, answer, citations, confidence, refusal_reason, disclaimer} |
| <code>GET /v1/citations/{chunk_id}</code> | 回链原文：text、context_header、locator（page / bbox）、evidence_level |
| <code>POST /v1/feedback</code> | trace_id + rating(-1 或 1) + comment |
| <code>GET /healthz</code> | 存活探测 + 缓存命中率 |
| <code>GET /readyz</code> | 就绪探测（四项依赖，未就绪 503 并点名） |

错误码：<code>INVALID_REQUEST 400</code> / <code>NOT_FOUND 404</code> / <code>BACKEND_UNAVAILABLE 503</code> / <code>LLM_UPSTREAM_ERROR 502</code> / <code>TIMEOUT 504</code> / <code>INTERNAL 500</code>。

## 7. 评测闭环与证据（为何它重要）

招聘方默认候选人会写 Agent，真正想问的是「你怎么证明它变好了」。本项目提供：160 题冻结评测集、三维 LLM-as-judge、劣化探针、ECE 校准、分层评测与 CI 回归门禁（<code>.github/workflows/eval.yml</code>）。[在线功能快照](https://huobuzhe.github.io/showcase/)「评测门禁与性能回归」卡片的数据即来自 <code>docs/evidence/</code> 里的报告。

诚实说明：报告里的 <code>pipeline_selfcheck</code> / <code>llm_mode=fake</code> 是流水线自检口径，**不是**对系统质量或临床有效性的结论；压测数字只代表本机环境。

## 8. 与展示站的关系

[在线功能快照](https://huobuzhe.github.io/showcase/) 里「循证医疗问答与证据溯源」卡片的 5 个实例（正常回答带 3 条引用、缓存命中、语料外拒答、过滤器空证据拒答、引用回链），来自本项目 2026-11-13 本机真实运行的 <code>/v1/ask</code>；原始报告见 <code>docs/evidence/</code>。[展示站源码](https://github.com/huobuzhe/showcase) 是静态快照，不连接本仓库的服务。

## 9. 常见问题

- **<code>ModuleNotFoundError: medevirag</code>**：没设 <code>PYTHONPATH=src</code>，见 5.2。
- **首次跑很慢 / 下载模型**：bge-m3 与 bge-reranker 各约 2.2GB，首次会拉到 <code>TRANSFORMERS_CACHE</code> 指定目录。
- **显存 / 内存不足**：设 <code>MEDEVIRAG_DEVICE=cpu</code>；<code>requirements.txt</code> 注释说明 <code>low_cpu_mem_usage=True</code> 是必需的。
- **<code>/readyz</code> 503**：按返回体点名的依赖，起 Docker Compose 或改 <code>.env</code> 连接串。
- **答案标注 <code>llm_mode=fake</code>**：这是离线自检桩；接真实生成模型请在 <code>.env</code> 填 <code>SILICONFLOW_API_KEY</code> 或 <code>DASHSCOPE_API_KEY</code>。
- **中文乱码**：命令统一加 <code>-X utf8</code>。

## 10. 安全与免责声明

- <code>.env</code>、虚拟环境、缓存、模型权重、原始语料 / 索引均未入库；变量名见 <code>.env.example</code>。
- 仅供学习与工程演示，不构成医疗建议；任何健康问题请咨询专业人士。
