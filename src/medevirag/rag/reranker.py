"""T1 重排器：bge-reranker-v2-m3 做 cross-encoder 精排（docs/04 §T1）。

为什么需要：S2 的 bi-encoder 把 10 万级 chunk 缩到 50（负责"别漏"），本阶段用 cross-encoder
（query 与 passage 拼接后一次前向，[CLS] 位置过线性头出标量分）把 50 缩到 8（负责"排准"）。
这是漏斗串联，不是二选一。

本地模型加载需 low_cpu_mem_usage + fp16（约 2.27GB）。CUDA 可用时走 fp16，否则 fp32。
CUDA OOM 时自动减半批大小重试一次。
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
# 避免 transformers 向不可写的 C:\\Users\\11\\.cache 写缓存（实测会报警并重试）。
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "data" / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _default_device() -> str:
    """设备选择：环境变量优先，其次 CUDA 可用性。

    为什么要这个开关：每个 Python 进程都会在 GPU 上建一份模型，多个进程（并行跑验收、
    并行跑 pytest）会争抢 CUDA 上下文，在 Windows 上直接表现为 access violation
    （进程无 traceback 直接死，汇总行显示 passed 但退出码非 0）。
    设 MEDEVIRAG_DEVICE=cpu 即可把并发进程的显存占用归零。
    """
    env = (os.environ.get("MEDEVIRAG_DEVICE") or "").strip().lower()
    if env in ("cpu", "cuda"):
        if env == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    """bge-reranker-v2-m3 精排器；单例在 nodes.get_reranker() 里管理，避免每问重加载。"""

    def __init__(
        self,
        model_dir: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        if model_dir is None:
            model_dir = str(ROOT / "models" / "bge-reranker-v2-m3")
        self.model_dir = model_dir
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if device is None:
            device = _default_device()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=1,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def score(self, query: str, passages: list[str]) -> list[float]:
        """批推理：输入 [query, passage] 对，输出标量分（logits[:, 0]）。"""
        if not passages:
            return []
        bs = self.batch_size
        scores: list[float] = []
        for i in range(0, len(passages), bs):
            batch = passages[i : i + bs]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).logits
            scores.extend(out.float().squeeze(-1).tolist())
        return scores

    def rerank(self, query: str, candidates: list[Any], top_n: int = 8) -> list[Any]:
        """对候选按其 text + context_header 拼接打分，按 (-score, fused_rank) 排序取 top_n。

        返回新的 RetrievedChunk（用 dataclasses.replace 追加 rerank_score / rerank_rank），
        不破坏入参。
        """
        passages = [
            f"{getattr(c, 'text', '')}\n{getattr(c, 'context_header', '')}"
            for c in candidates
        ]
        try:
            raw = self.score(query, passages)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.batch_size > 1:
                torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)
                raw = self.score(query, passages)
            else:
                raise

        scored = []
        for chunk, s in zip(candidates, raw, strict=False):
            fused = getattr(chunk, "fused_rank", 0)
            scored.append((float(s), fused, chunk))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for rank, (s, _fused, chunk) in enumerate(scored[:top_n], start=1):
            out.append(dataclasses.replace(chunk, rerank_score=s, rerank_rank=rank))
        return out
