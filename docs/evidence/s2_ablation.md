# S2 索引与混合召回 · 消融报告

- generated_at: `2026-10-24T09:37:43+08:00`
- config_hash: `sha256:b18ced496e95`
- eval_set: `data/eval/retrieval_dev60.jsonl`  split=`dev`  cases=48
- index.source: `milvus-lite`  milvus_uri=`./data/indexes/milvus_lite.db`  N=4023
- 冻结参数: dense_topk=50 bm25_topk=50 rrf_k=60 topk=10
- 阈值(docs/00 §9): Recall@10 ≥ 0.85 / MRR ≥ 0.62 / nDCG@10 ≥ 0.70

## 主表：四组变体 × 三指标

| variant | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense-only | 0.9062 | 0.7951 | 0.7312 |
| bm25-only | 0.8646 | 0.7431 | 0.6775 |
| rrf | 0.9271 | 0.8351 | 0.7552 |
| weighted-sum | 0.9271 | 0.8538 | 0.7648 |

## 分组小表：按 case_type

### case_type = exact_term

| variant | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense-only | 0.9375 | 0.7760 | 0.7553 |
| bm25-only | 0.9375 | 0.7917 | 0.7583 |
| rrf | 1.0000 | 0.8802 | 0.8200 |
| weighted-sum | 1.0000 | 0.8771 | 0.8149 |

### case_type = semantic

| variant | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense-only | 1.0000 | 0.8281 | 0.7892 |
| bm25-only | 1.0000 | 0.7641 | 0.7239 |
| rrf | 1.0000 | 0.8125 | 0.7728 |
| weighted-sum | 1.0000 | 0.8802 | 0.8020 |

### case_type = multi_hop

| variant | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense-only | 0.7812 | 0.7812 | 0.6491 |
| bm25-only | 0.6562 | 0.6736 | 0.5502 |
| rrf | 0.7812 | 0.8125 | 0.6727 |
| weighted-sum | 0.7812 | 0.8042 | 0.6775 |

## 过滤与放宽统计（post-filter，ADR-03）

- 候选输入合计 9600，保留 8816
- 按原因丢弃: {'evidence_level': 7, 'publish_year': 777}
- 触发放宽的 query 数: 0 / 48
- 放宽明细: 无

### 硬过滤对 gold 的代价（诊断项，不改阈值、不删难例）

- 被硬规则淘汰的 gold 块: 2 个
  - q047 (dev) 0527_153307_354#(1)#c02 淘汰原因=publish_year 年份=2008 来源=登革热病例监测指南
  - q047 (dev) 0527_153307_354#(一)#c03 淘汰原因=publish_year 年份=2008 来源=登革热病例监测指南
- 关掉硬过滤后的诊断指标（同一批 query，仅供判读，不进主表）: Recall@10=0.9479 MRR=0.8559 nDCG@10=0.7707
- 解读：过滤是 ADR-03 的产品决定（过期指南不应被引用），代价就是被淘汰 gold 所在的 query 必然漏召；关掉过滤后指标回升多少，就是这份产品约束的价格。此处只记录、不改阈值、不删难例——要动只能动数据决策（是否保留过期指南），由验收方定。

## 延迟（rrf 变体，端到端 retrieve，Lite 路径）

- n=48  P50=8.28 ms  P90=8.85 ms  P95=9.47 ms  mean=8.38 ms  max=10.54 ms

## Holdout（test split 12 题）

> 判定口径：**A5/A6/A7 只针对 dev split（48 题）**；本节的 test split 是 holdout，从未参与调参，指标单独列出、不与 dev 合并成一个数糊过去。阈值仍是 docs/00 §9 的 0.85 / 0.62 / 0.70，未做任何调整。

| metric | value | threshold | verdict |
|---|---|---|---|
| recall@10 | 0.7500 | 0.85 | FAIL |
| mrr | 0.7250 | 0.62 | PASS |
| ndcg@10 | 0.6482 | 0.70 | FAIL |

- 结论：**未全部达标**（recall@10=0.7500 FAIL，mrr=0.7250 PASS，ndcg@10=0.6482 FAIL）
- 归因分布: {'hit': 8, 'recall_fail': 1, 'metadata_filtered': 1, 'multi_hop_partial': 2}

### 逐题明细（rrf 生产路径，含硬过滤）

| qid | case_type | difficulty | recall | mrr | nDCG | 归因 | 命中 gold | 未命中 gold(原因) |
|---|---|---|---|---|---|---|---|---|
| q005 | exact_term | medium | 1.00 | 1.00 | 0.83 | hit | 0201_084005_469#1.4#c57 | - |
| q010 | exact_term | medium | 0.00 | 0.00 | 0.14 | recall_fail | - | 1031_100011_575#8.3#c257(recall_miss) |
| q015 | exact_term | easy | 1.00 | 1.00 | 1.00 | hit | 0727_162437_802#(三)#c42 | - |
| q020 | exact_term | medium | 0.00 | 0.00 | 0.00 | metadata_filtered | - | 0527_153307_354#(1)#c02(publish_year) |
| q025 | semantic | medium | 1.00 | 1.00 | 0.83 | hit | 1031_101439_126#三、#c08 | - |
| q030 | semantic | medium | 1.00 | 1.00 | 0.83 | hit | 1031_095848_337#(三)#c16 | - |
| q035 | semantic | medium | 1.00 | 1.00 | 0.92 | hit | 0123_130133_511#4.2.2#c05 | - |
| q040 | semantic | medium | 1.00 | 1.00 | 1.00 | hit | 0531_112855_497#(一)#c09 | - |
| q045 | multi_hop | hard | 0.50 | 1.00 | 0.56 | multi_hop_partial | 0201_084005_469#1.4#c57 | 0201_084005_469#h1185#c59(recall_miss) |
| q050 | multi_hop | medium | 1.00 | 1.00 | 0.86 | hit | 0621_144019_327#5.2.3#c21, 0621_143946_131#(一)#c08 | - |
| q055 | multi_hop | hard | 0.50 | 0.20 | 0.22 | multi_hop_partial | 0531_112855_497#(一)#c14 | 0531_112855_497#(一)#c09(recall_miss) |
| q060 | multi_hop | medium | 1.00 | 0.50 | 0.60 | hit | 0531_112531_215#(一)#c03, 0531_112531_215#五、#c13 | - |

### 归因三类（全部由脚本逐题测定，不是人工判断）

- `metadata_filtered`：未命中的 gold **全部**被硬过滤规则（publish_year/lang/doc_type/evidence_level）淘汰 —— 这是 ADR-03 post-filter 的代价，不是召回失败。
- `multi_hop_partial`：命中了部分 gold、漏了另一部分 —— recall 的分母是 len(gold)，所以 2 个 gold 命中 1 个只得 0.5，这是多跳题的主要失分形态。
- `recall_fail`：一个 gold 都没命中且不是过滤造成 —— 召回本身失败。

## 消融表的读法（docs/03 §T7）

1) **dense-only vs bm25-only**：dev 上 dense-only Recall@10 = 0.9062，bm25-only = 0.8646，差值 +0.0416。按 case_type 拆开看：exact_term 组 dense 0.9375 vs bm25 0.9375（两路持平）、semantic 组 dense 1.0000 vs bm25 1.0000（两路持平）、multi_hop 组 dense 0.7812 vs bm25 0.6562（dense 更强）。字面命中靠 BM25 的倒排精确匹配，改写/近义表述靠 dense 的语义相似，所以两路的强弱必须分组看——单看总量会掩盖它们在 exact_term 与 semantic 上的相反分工，这也正是融合有增益空间的前提。

2) **RRF 是否 ≥ 两路各自最大值**：是，但增益很小。rrf Recall@10 = 0.9271 ≥ max(dense=0.9062, bm25=0.8646) = 0.9062，只高 +0.0209。原因就在两路 gold 命中的重叠度上：Jaccard = 0.9375（交 60 / 并 64），即两路几乎召回同一批 gold，RRF 能额外捞回的只有并集里那少数几个块。按 docs/03 §T7 的判据，重叠高时该去查 BM25 分词与 dense 模型是否在重复劳动，而不是继续调融合参数——本数据集上两路都已命中该命中的块，所以限制 Recall@10 上限的不是融合，而是召回本身（分组下降最大的是 multi_hop 组两路最低 0.6562）。

3) **weighted-sum vs rrf**：两者几乎持平（weighted-sum 0.9271 vs rrf 0.9271，差 +0.0000）。本数据集上加权和并不劣于 RRF，但 weighted-sum 的 alpha=0.5 是在看 dev 之前就冻结的，一旦改成按 dev 调 alpha 就会过拟合；RRF 免调参、对两路量纲不敏感（BM25 分数域 3~35，余弦 [-1,1]，直接加权需要 min-max 归一化），工程上仍选 RRF（ADR-02）。

### 关键数字

- dense-only Recall@10 = 0.9062，bm25-only Recall@10 = 0.8646，两者最大值 = 0.9062
- rrf Recall@10 = 0.9271（≥ 两路最大值）
- 两路 gold 命中重合: intersection=60 / union=64 → Jaccard=0.9375
- weighted-sum Recall@10 = 0.9271（相对 rrf +0.0000）

