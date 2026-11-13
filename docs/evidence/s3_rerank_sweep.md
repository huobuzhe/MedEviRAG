# S3 重排档位扫描（K=10/20/50）

数据集：data/eval/retrieval_dev60.jsonl（60 题全量）。口径：MRR@8 / Recall@8，候选池来自 RRF 融合 Top50。

基线（B0：无重排，直接取 RRF Top8）：MRR@8 = 0.8131，Recall@8 = 0.9500

| K | MRR@8 | Recall@8 | ΔMRR | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|
| 10 | 0.8575 | 0.9500 | +0.0444 | 108.9 | 115.3 |
| 20 | 0.8688 | 0.9667 | +0.0557 | 212.1 | 229.7 |
| 50 | 0.8675 | 0.9667 | +0.0544 | 512.9 | 555.0 |

## 结论

- 延迟（A2）：K=20 P95 = 229.7 ms ≤ 300 ms ✅；K=50 P95 = 555.0 ms ≤ 800 ms ✅
- 收益（A3）：最优档 K=20 的 ΔMRR = **0.0557 < 0.10 门禁** ❌
- 收敛（A4）：K=20 与 K=50 的 ΔMRR 差 = 0.0013 < 0.01，三档数据齐全后收敛到 K=20。
- **最终决策：按 docs/04 §T2 R2 分支否决重排。** node_rerank 直通（取 filtered 的 Top8，保持 RRF 顺序），
  configs/retrieval.yaml 增加 rerank.enabled: false，T1 的重排器代码保留（可用开关重新启用）。

## 否决理由（不凑指标，如实记录）

1. 本项目的 bi-encoder（bge-m3）+ RRF 基线已经较强（MRR@8 = 0.8131），重排可提升的空间被压缩，
   与 docs/04 知识点 2 所述「recall ceiling 与边际收益递减」一致。
2. 实测 ΔMRR = 0.0557，仅约为门禁 0.10 的 56%，不足以抵消每问 200–550 ms 的额外延迟预算。
3. 重排优化的是排序质量而非召回：Recall@8 仅从 0.9500 提升到 0.9667（+1.67 个百分点）。

复现：`PYTHONPATH=src ./.venv/Scripts/python.exe -X utf8 scripts/s3_rerank_sweep.py --ks 10,20,50`