# S6 T9：压测脚本（docs/07 §T9）。asyncio+httpx，输出 P50/P95/P99。
# 注意：docs 用 httpx.AsyncClient；本机 httpx 已随 fastapi 安装。

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, q: str) -> dict:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post("/v1/ask", json={"question": q, "top_k": 50},
                                  timeout=30.0)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                return {"ok": False, "ms": ms, "code": r.status_code}
            b = r.json()
            return {"ok": True, "ms": ms, "status": b["status"],
                    "retrieval": b["trace"].get("retrieval_ms", 0),
                    "rerank": b["trace"].get("rerank_ms", 0),
                    "generation": b["trace"].get("generation_ms", 0)}
        except Exception as exc:
            return {"ok": False, "ms": (time.perf_counter() - t0) * 1000,
                    "code": str(exc)[:60]}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--questions", default="data/eval/questions.txt")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    try:
        with open(a.questions, encoding="utf-8") as fh:
            qs = [ln.strip() for ln in fh if ln.strip()]
    except FileNotFoundError:
        qs = ["成人高血压的诊断标准是什么", "2型糖尿病的血糖控制目标",
              "血脂异常的降脂目标", "COPD稳定期的管理要点", "房颤的抗凝指征"]
    tasks_src = (qs * a.repeat)[: max(a.concurrency, len(qs))]
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(base_url=a.base_url) as client:
        for q in tasks_src[:3]:          # 预热剔除冷启动
            await one(client, sem, q)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(client, sem, q) for q in tasks_src))
        wall = time.perf_counter() - t0

    ok = [r for r in res if r["ok"]]
    lat = [r["ms"] for r in ok]
    err = len(res) - len(ok)
    print(f"requests={len(res)} ok={len(ok)} err={err} wall={wall:.1f}s "
          f"qps={len(res) / wall:.2f}")
    if lat:
        print(f"P50={pct(lat, 50):.0f}ms P95={pct(lat, 95):.0f}ms "
              f"P99={pct(lat, 99):.0f}ms mean={statistics.mean(lat):.0f}ms "
              f"max={max(lat):.0f}ms")
        for k in ("retrieval", "rerank", "generation"):
            vals = [r[k] for r in ok if k in r]
            if vals:
                print(f"{k}_mean={statistics.mean(vals):.0f}ms")
    partial = sum(1 for r in ok if r.get("status") == "partial")
    print(f"partial_rate={partial / max(1, len(ok)):.3f}  "
          f"error_rate={err / max(1, len(res)):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
