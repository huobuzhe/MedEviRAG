# P2 Agent 轨迹评测 · 消融表

| 配置 | task_success_rate | tool_selection_accuracy | param_accuracy | avg_steps | total_tokens | loop_detection_rate | error_recovery_rate | hitl_trigger_accuracy |
|---|---|---|---|---|---|---|---|---|
| full | 1.0 | 1.0 | 0.8571 | 2.12 | 7450 | 1.0 | 1.0 | 1.0 |
| no_loop_detection | 0.875 | 1.0 | 0.8824 | 2.12 | 9400 | 0.0 | 1.0 | 1.0 |
| no_budget_control | 0.875 | 1.0 | 0.875 | 2.38 | 8750 | 1.0 | 1.0 | 1.0 |
| no_hitl | 0.875 | 1.0 | 0.8571 | 2.12 | 7450 | 1.0 | 1.0 | 0.75 |
| no_error_recovery | 0.75 | 1.0 | 0.8333 | 1.88 | 6550 | 1.0 | 0.0 | 1.0 |
| bare | 0.375 | 1.0 | 0.8824 | 2.12 | 9800 | 0.0 | 0.0 | 0.75 |

> 口径：风险类任务的成功 = 对应机制正确处置（拦住/恢复/审批）；普通任务 = 工具命中且无未处理参数错误
> 成本按 0.001 元/1k tokens 估算，只用于相对比较。
