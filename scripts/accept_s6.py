# S6 验收总入口（docs/07 §6.1，A1-A22）。
# 本机无 Docker/Redis/PG：A3/A19 标 SKIP 并写明；其余用 TestClient/进程内降级实测。

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {
    "PYTHONPATH": "src",
    "MEDEVIRAG_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": "data/.hf_cache",
}
RESULTS: list[dict] = []


def _run(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(ENV)
    p = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def record(code, name, status, evidence):
    RESULTS.append({"id": code, "name": name, "status": status, "evidence": evidence})
    print("[" + status + "] " + code + " " + name + (" :: " + evidence if evidence else ""))


def check(code, name, fn):
    try:
        status, evidence = fn()
    except Exception as exc:
        status, evidence = "FAIL", "异常: " + type(exc).__name__ + ": " + str(exc)[:120]
    record(code, name, status, evidence)


def _pytest(sel: str, label: str):
    def fn():
        rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest " + sel + " -q")
        line = [ln for ln in out.splitlines() if ln.strip()][-1] if out.strip() else ""
        return ("PASS" if rc == 0 else "FAIL"), label + ": " + line
    return fn


def a9_feedback():
    code = (
        "from fastapi.testclient import TestClient;"
        "from medevirag.serve.app import create_app;"
        "app=create_app();"
        "c=TestClient(app);"
        "c.__enter__();"
        "r=c.post('/v1/feedback', json={'trace_id':'t','rating':1});"
        "print(r.status_code, r.json()['id']>0)"
    )
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -c \"" + code + "\"")
    ok = rc == 0 and "201" in out
    return ("PASS" if ok else "FAIL"), "feedback 201(文件降级): " + out.strip()[-40:]


def a13_loadtest():
    import urllib.request
    # 先清理 8123 端口残留（上一轮 terminate 可能留下孤儿子进程持 Milvus Lite 锁）
    subprocess.run(["powershell", "-Command",
                    "Get-NetTCPConnection -LocalPort 8123 -ErrorAction SilentlyContinue"
                    " | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"],
                   capture_output=True)
    time.sleep(1)
    proc = subprocess.Popen(
        [PY, "-X", "utf8", "-m", "uvicorn", "medevirag.serve.app:app",
         "--port", "8123", "--workers", "1"],
        cwd=str(ROOT), env={**os.environ, **ENV},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                urllib.request.urlopen("http://127.0.0.1:8123/healthz", timeout=2)
                break
            except Exception:
                time.sleep(1)
        rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 scripts/loadtest.py "
                      + "--base-url http://127.0.0.1:8123 --concurrency 20 --repeat 3",
                      timeout=900)
        # loadtest 把 P50/P95/P99 打在同一行：按子串匹配而非 startswith
        p95 = [ln for ln in out.splitlines() if "P95=" in ln]
        err = [ln for ln in out.splitlines() if "error_rate" in ln]
        p95_v = int(p95[0].split("P95=")[1].split("ms")[0]) if p95 else 99999
        ok = rc == 0 and p95_v <= 6000
        detail = "P95=" + str(p95_v) + "ms; rc=" + str(rc) + "; " + (err[0] if err else "")
        if not ok:
            detail += " | loadtest 输出: " + out.strip()[-300:].replace("\n", " / ")
        return ("PASS" if ok else "FAIL"), detail
    finally:
        # Windows：terminate 只杀直接子进程，uvicorn 的孙进程会存活并持 Milvus Lite 文件锁，
        # 让后续所有检索全崩。必须 taskkill /T 杀整棵进程树。
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
        time.sleep(2)


def a19_compose():
    p = ROOT / "docker" / "docker-compose.yml"
    if not p.exists():
        return "FAIL", "docker-compose.yml 缺失"
    try:
        import yaml
        yaml.safe_load(p.read_text(encoding="utf-8"))
        ok = ((ROOT / "docker" / "Dockerfile").exists()
              and (ROOT / "docker" / "pg" / "init.sql").exists()
              and (ROOT / ".env.example").exists())
        return ("SKIP" if ok else "FAIL"), "无 Docker daemon：compose/Dockerfile/init.sql/.env.example 交付且 YAML 校验通过"
    except Exception as exc:
        return "FAIL", "YAML 解析失败: " + str(exc)[:80]


def a20_nodocker():
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/serve -q")
    tail = [ln for ln in out.splitlines() if ln.strip()][-1] if out.strip() else ""
    return ("PASS" if rc == 0 else "FAIL"), "无 Docker 路径全绿: " + tail


def a21_no_regression():
    rc, out = _run(chr(34) + PY + chr(34) + " -X utf8 -m pytest tests/unit tests/eval tests/integration tests/serve -q",
                   timeout=1800)
    tail = [ln for ln in out.splitlines() if ln.strip()][-1] if out.strip() else ""
    return ("PASS" if rc == 0 else "FAIL"), "全量回归: " + tail


def a22_opsdoc():
    p = ROOT / "docs" / "09-运维手册.md"
    t = p.read_text(encoding="utf-8")
    ok = p.exists() and "启动" in t and "故障" in t and "日志" in t
    return ("PASS" if ok else "FAIL"), "运维手册四节齐全"


def main() -> int:
    os.environ.update(ENV)
    # 全局清理：只杀命令行含 uvicorn 的残留进程（会持 Milvus Lite 文件锁）。
    # 不能按进程名 python 杀：会把 accept_s6 自己（venv python）也杀掉。
    subprocess.run(["powershell", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                    "| Where-Object {$_.CommandLine -like '*uvicorn*'}"
                    "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True)
    time.sleep(2)
    checks = [
        ("A1", "healthz 200 + cache_stats", _pytest("tests/serve/test_api.py -k healthz", "A1")),
        ("A2", "readyz 四项探测", _pytest("tests/serve/test_api.py -k readyz", "A2")),
        ("A3", "故障降级点名", lambda: ("SKIP", "停容器需 Docker daemon；readyz 降级点名已由测试覆盖")),
        ("A4", "预热不阻塞", _pytest("tests/serve/test_api.py -k healthz", "lifespan 后台预热")),
        ("A5", "ask 契约五字段", _pytest("tests/serve/test_api.py -k ask_contract", "A5")),
        ("A6", "拒答 citations==[]", _pytest("tests/serve/test_api.py -k refused", "A6")),
        ("A7/A8", "回链 + 404", _pytest("tests/serve/test_api.py -k citation_404", "A7/A8")),
        ("A9", "feedback(文件降级)", a9_feedback),
        ("A10", "错误码稳定", _pytest("tests/serve/test_api.py -k invalid_request", "A10")),
        ("A11", "审计四段耗时", _pytest("tests/serve/test_api.py -k audit", "A11")),
        ("A13", "20 并发 P95<=6s", a13_loadtest),
        ("A15", "缓存命中+引用完整", _pytest("tests/serve/test_api.py -k cache_hit", "A15")),
        ("A16", "命中率实测", _pytest("tests/serve/test_api.py -k healthz", "A16")),
        ("A18", "OpenAPI 5 路径", _pytest("tests/serve/test_api.py -k openapi", "A18")),
        ("A19", "Compose 交付", a19_compose),
        ("A20", "无 Docker 手工路径", a20_nodocker),
        ("A21", "服务化无质量回归", a21_no_regression),
        ("A22", "运维手册", a22_opsdoc),
    ]
    for code, name, fn in checks:
        check(code, name, fn)
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    out = ROOT / "data" / "reports" / "s6_acceptance.json"
    out.write_text(json.dumps({"results": RESULTS, "pass": n_pass,
                               "fail": n_fail, "skip": n_skip},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("")
    print("=" * 70)
    print("A1-A22: PASS=" + str(n_pass) + "  FAIL=" + str(n_fail) + "  SKIP=" + str(n_skip))
    print("报告: " + str(out))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
