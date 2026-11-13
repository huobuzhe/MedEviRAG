# P2 单元测试：工具校验、死循环防护、HITL、恢复，以及消融确实改变指标。
from medevirag.agent.reactor import ReActAgent
from medevirag.agent.tools import ToolRegistry
from medevirag.eval.agent_eval import CONFIGS, build_tasks, compute_metrics, run_one_config


def test_tool_registry_rejects_bad_params_without_swallowing():
    reg = ToolRegistry()
    bad = reg.call("calculate_nutrition", {"ingredients": "番茄"})
    assert not bad.ok
    assert bad.error.startswith("PARAM_INVALID")
    good = reg.call("calculate_nutrition", {"ingredients": ["番茄", "鸡蛋"]})
    assert good.ok


def test_unknown_tool_is_reported():
    r = ToolRegistry().call("no_such_tool", {})
    assert not r.ok and r.error.startswith("UNKNOWN_TOOL")


def test_loop_detection_blocks_repeated_action():
    tasks = [t for t in build_tasks(1) if t.kind == "loop_risk"]
    on = ReActAgent().run(tasks[0])
    off = ReActAgent(loop_detection=False, budget_control=False).run(tasks[0])
    assert on.loop_blocked is True
    assert off.loop_blocked is False
    assert len(off.called_tools) > len(on.called_tools)


def test_hitl_and_error_recovery_flags():
    tasks = {t.kind: t for t in build_tasks(1)}
    hitl = ReActAgent().run(tasks["hitl_required"])
    assert hitl.hitl_triggered is True
    rec = ReActAgent().run(tasks["param_error"])
    assert rec.recovery_attempted is True and rec.recovered is True
    no_rec = ReActAgent(error_recovery=False).run(tasks["param_error"])
    assert no_rec.recovered is False


def test_ablation_changes_measured_metrics():
    tasks = build_tasks(2)
    full = run_one_config(tasks, CONFIGS["full"])
    no_loop = run_one_config(tasks, CONFIGS["no_loop_detection"])
    no_hitl = run_one_config(tasks, CONFIGS["no_hitl"])
    no_rec = run_one_config(tasks, CONFIGS["no_error_recovery"])
    assert no_loop["loop_detection_rate"] < full["loop_detection_rate"]
    assert no_hitl["hitl_trigger_accuracy"] < full["hitl_trigger_accuracy"]
    assert no_rec["error_recovery_rate"] < full["error_recovery_rate"]
    assert no_loop["total_tokens"] > full["total_tokens"]


def test_metric_ranges_are_sane():
    tasks = build_tasks(1)
    traces = [ReActAgent().run(t) for t in tasks]
    m = compute_metrics(tasks, traces)
    for k in ("task_success_rate", "tool_selection_accuracy", "param_accuracy",
              "loop_detection_rate", "error_recovery_rate", "hitl_trigger_accuracy"):
        assert 0.0 <= m[k] <= 1.0
    assert m["n_tasks"] == len(tasks)
    assert m["total_tokens"] > 0
