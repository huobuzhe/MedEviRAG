# MedEviRAG · 中文医学循证问答系统 — 使用文档

> **一句话**：输入一个临床问题，系统从本地中文医学指南语料中检索证据，生成**每句带可点击引用标记**的回答；每条引用可回链到指南原文的页码与版面坐标（bbox）；问题越界（个体化用药/急诊/超范围）时**明确拒答并说明原因**，而不是硬答。

---

## 1. 这个项目能做什么

| 能力 | 说明 | 实测指标 |
|---|---|---|
| **循证问答** | 基于已收录的中文指南/共识语料回答临床问题，答案带 `[C1]` 编号引用 | 20 并发 P95=1.9s，零错误 |
| **引用回链** | 点引用可查到：出自哪部指南、哪一章节、第几页、页面上哪块区域（bbox 坐标）、证据等级 | 12 字段详情，页码为整数、bbox 为 4 元素数组 |
| **两级引用校验** | 模型编造的引用标记会被自动拦截，绝不带病输出 | 注入 `[C9]` 漏出 0 次 |
| **安全拒答** | 17 条规则识别四类不可答情形：越界/时效/个体化用药/证据不足；拒答时说明原因并给就医指引 | 召回 1.00、误拒 0.00（夹具级） |
| **混合检索** | 向量（bge-m3）+ BM25 双路召回、RRF 融合、元数据硬过滤（年份/语言/文献类型/证据等级） | Recall@10 达标 |
| **置信度输出** | 四分量复合分（排序分/引用覆盖率/证据等级/区分度），低于阈值自动降级为 partial 或拒答 | ECE 实测输出 |
| **评测闭环** | 160 题分层评测集（冻结防篡改）、三维 LLM judge、回归门禁（指标跌超 5% 即非零退出） | 篡改一行 gold 立即被拦 |

### 不做什么（边界）

- **不开处方/不下个体化剂量**：「我该吃多少 mg」必拒，并建议面诊；
- **不答急诊**：「我现在胸痛」直接引导拨 120，不给出诊疗建议；
- **不答语料外问题**：诗歌、股票、语料未覆盖的罕见病，明确说"没有证据"而不是编。

---

## 2. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11 | 已实测；Linux 可跑但路径示例为 Windows |
| Python | 3.10（项目自带 `.venv`） | `D:\Anaconda\envs\pytorch\python.exe -m venv .venv --system-site-packages` 创建 |
| GPU | 可选 | 有 CUDA 则检索加速；无 GPU 自动降级 CPU（设 `MEDEVIRAG_DEVICE=cpu`） |
| 磁盘 | ~10GB | 语料+索引+两个本地模型（bge-m3 2.2GB + bge-reranker 2.2GB） |
| 网络 | 不需要 | **全流程零 API 消耗**（离线桩），接真实 LLM 时才需要 key |

---

## 3. 五分钟上手

### 3.1 初始化环境（每次打开新终端执行一次）

```powershell
Set-Location F:\MedEviRAG
$env:PYTHONPATH = "src"
$env:MEDEVIRAG_DEVICE = "cpu"            # 有 GPU 改 cuda
$env:TRANSFORMERS_CACHE = "data/.hf_cache"
.\.venv\Scripts\python.exe -X utf8 -m medevirag.cli doctor   # 环境自检
```

### 3.2 问一个问题（命令行）

```powershell
.\.venv\Scripts\python.exe -X utf8 -m medevirag.cli ask "成人高血压的诊断标准是什么"
```

输出示例（无 API key 时答案为链路自检桩，标注 llm_mode）：

```
llm_mode=fake-deterministic（无 API key：答案仅为链路自检，不具医学意义）
status=answered  reason=None
answer=... [C1]
  [C1] <chunk_id>  p4  medevirag://doc/...?page=4&bbox=...
trace={"route":0,"retrieve":...,...}
```

### 3.3 查看一条引用的原文

```powershell
# chunk_id 从上一步输出取；含 # 的 id 必须加双引号
.\.venv\Scripts\python.exe -X utf8 -m medevirag.cli citation "0108_135352_975#(一)#c03" --json
```

返回 12 个字段：chunk_id / doc_id / doc_title / section_path / chunk_type / page / bbox / block_index / text（原文全文）/ context_header / evidence_level / url（深链）。

### 3.4 起服务 + 开前端（推荐体验方式）

```powershell
# 终端 1：起 API（端口 8000）
.\.venv\Scripts\python.exe -X utf8 -m uvicorn medevirag.serve.app:app --port 8000 --workers 1

# 终端 2：起前端（端口 5173，首次需 npm install）
Set-Location web; npm install; npm run dev
```

浏览器打开 **http://127.0.0.1:5173**：

1. 输入问题 → 看到带 `[C1]` 标号的答案；
2. 点答案里的标号 → 右栏对应引用卡片高亮（文档名/章节/页码/证据等级/原文片段）；
3. 点卡片上的「查看原文」→ 就地展开该 chunk 全文与 bbox；
4. 底部可点「有用/无用」反馈（落 `data/logs/feedback.jsonl`）。

API 端点（浏览器打开 http://127.0.0.1:8000/docs 有交互式文档）：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | /healthz | 存活探针 + 缓存命中率 |
| GET | /readyz | 就绪探针（四项依赖探测，未就绪 503 并点名） |
| POST | /v1/ask | 问答（body: question/top_k/filters/use_cache） |
| GET | /v1/citations/{chunk_id} | 引用详情 |
| POST | /v1/feedback | 评分反馈（-1/1） |

---

## 4. 全部命令一览

| 阶段 | 命令 | 用途 |
|---|---|---|
| 自检 | `cli doctor` | GPU/语料/索引/模型逐项检查 |
| 语料 | `cli fetch [--url]` | 登记或下载指南 PDF |
| | `cli parse` | 解析 PDF → 章节树（落 data/interim） |
| | `cli chunk` | 章节感知切分 → data/processed/chunks.jsonl |
| | `cli report` | 重算切分质量指标 |
| 索引 | `cli index [--drop]` | 向量化 + Milvus Lite + BM25 三建 |
| 检索 | `cli retrieve "问句" --topk 10` | 单条检索调试（看两路分数与来源） |
| 问答 | `cli ask "问句" [--json]` | 8 节点全链路问答 |
| 溯源 | `cli citation "<chunk_id>" [--json]` | 引用回链详情 |
| 拒答 | `cli refusal-matrix --out data/reports/refusal_matrix.json` | 跑 54 条规则夹具矩阵 |

## 5. 评测与验收

```powershell
# 拒答矩阵（S4）
.\.venv\Scripts\python.exe -X utf8 -m medevirag.cli refusal-matrix

# 全量评测（S5：160 题 dev 集，冻结校验）
.\.venv\Scripts\python.exe -X utf8 -m medevirag.eval.runner --split dev

# 回归比对（两次 run 的 delta，门禁非零退出）
.\.venv\Scripts\python.exe -X utf8 -m medevirag.eval.regression --baseline <id1> --current <id2>

# 压测（S6：20 并发 P95）
.\.venv\Scripts\python.exe -X utf8 scripts\loadtest.py --concurrency 20 --repeat 3

# 分阶段验收 s0-s6
0..6 | ForEach-Object { .\.venv\Scripts\python.exe -X utf8 scripts\accept_s$_.py }
```

## 6. 配置速查

| 文件 | 控制什么 |
|---|---|
| configs/retrieval.yaml | topk / RRF k / 硬过滤阈值 / 重排开关（enabled:false=否决直通） |
| configs/eval.yaml | 评测阈值（00 §9 唯一真值源）/ judge 路由 / 回归门禁档位 |
| configs/app.yaml | 服务端口 / 缓存 TTL / 分段耗时预算 / 并发上限 |
| configs/models.yaml | LLM 路由（按 provider 配代理）与 token 预算护栏 |

## 7. 接真实大模型（可选）

系统默认零 API 离线运行。要接真实生成模型：

1. `Copy-Item .env.example .env`，填入 `SILICONFLOW_API_KEY`（生成）与 `DASHSCOPE_API_KEY`（判分，必须与生成模型不同族）；
2. 评测跑 `--real`；此时 s3/s5 里的 SKIP 项即可复核（引用落点/句级支撑/ECE）。

## 8. 常见问题

| 症状 | 处置 |
|---|---|
| ask 提示模型/索引找不到 | 先跑 `cli index` 建索引；模型目录 models/ 缺失见 docs/09 |
| 8000 端口占用 | `netstat -ano | findstr :8000` 找 PID 后 `taskkill /PID <pid> /F` |
| 前端 CORS 报错 | 用 `npm run dev`（vite 代理转发，天然同源）；直连 8000 时 cors_origins 已含 5173 |
| 检索突然全崩 | 多半是孤儿 uvicorn 进程持有 Milvus Lite 文件锁；杀掉命令行含 uvicorn 的进程 |
| 答案显示 fake-deterministic | 正常：无 API key 时为离线桩，接 key 后消失 |

更多故障处置见 **docs/09-运维手册.md**；架构与决策见 docs/00（ADR 表）；各阶段设计/实测见 docs/01-07。
