"""T4：BM25 倒排（jieba 分词 + bm25s，docs/03 §T4）。

中文必须自己分词：bm25s 默认 token_pattern 会把整句切成几个超长 token，
中文召回会直接失效（docs/03 §5.7 有量化例子）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medevirag.index.paths import resolve

__all__ = ["BM25Store", "ChineseTokenizer"]

_BM25_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "params.index.json",
    "vocab.index.json",
)


class ChineseTokenizer:
    """jieba 精确模式 + 自定义医学词典 + 停用词表。"""

    def __init__(self, userdict: str | Path | None = None, stopwords: str | Path | None = None) -> None:
        import jieba

        self._jieba = jieba
        self.terms: list[str] = []
        if userdict:
            path = resolve(userdict)
            if path.exists():
                jieba.load_userdict(str(path))
                self._collect_terms(path)
        self._stop: set[str] = set()
        if stopwords:
            spath = resolve(stopwords)
            if spath.exists():
                for line in spath.read_text(encoding="utf-8").splitlines():
                    word = line.strip()
                    if word and not word.startswith("#"):
                        self._stop.add(word)
        jieba.initialize()

    def _collect_terms(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                self.terms.append(word)

    def add_terms(self, terms: Sequence[str], freq: int = 10000) -> None:
        """把领域词以高词频注册，保证 jieba 不会把它们切碎。"""
        for term in terms:
            if term:
                self._jieba.add_word(str(term), freq=int(freq))

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        for token in self._jieba.lcut(text, cut_all=False):
            word = token.strip().lower()
            if len(word) > 1 and word not in self._stop:
                out.append(word)
        return out


class BM25Store:
    """bm25s 倒排索引的 build/save/load/search 封装。"""

    def __init__(
        self,
        index_dir: str | Path,
        tokenizer: ChineseTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        method: str = "lucene",
    ) -> None:
        import bm25s

        self.dir = resolve(index_dir)
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.method = method
        self.retriever: Any = bm25s.BM25(k1=self.k1, b=self.b, delta=self.delta, method=method, dtype="float32")
        self.chunk_ids: list[str] = []
        self.avgdl: float = 0.0
        self.n_empty_fallback: int = 0

    # ------------------------------------------------------------------ build
    # 表格块里可能出现「| 阳 性 数 | 发 病 数 |」这种被空格打散的单元格，
    # jieba 精确模式切出来全是单字，被 len>1 规则全部过滤 -> 空 token 列表。
    # 这类块在真实语料里确实存在（4023 里有 2 条），既不能丢（A2 要求
    # bm25 的 N 与 chunk_ids.jsonl 行数严格相等），也不能让 build 直接失败，
    # 因此对空 token 的块退化为「按字符切」：它几乎不会命中查询，但保持了行对齐。
    _EMPTY_FALLBACK_MAX_RATIO = 0.05

    def _tokens_or_fallback(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        if tokens:
            return tokens
        chars = [ch.lower() for ch in text if not ch.isspace() and ch.isprintable()]
        self.n_empty_fallback += 1
        return chars or ["\u3000"]

    def build(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError(f"chunk_ids/texts 长度不一致: {len(chunk_ids)} vs {len(texts)}")
        self.n_empty_fallback = 0
        tokens = [self._tokens_or_fallback(str(t)) for t in texts]
        empty = sum(1 for t in tokens if not t)
        if empty or self.n_empty_fallback > max(1, len(texts) * self._EMPTY_FALLBACK_MAX_RATIO):
            raise ValueError(
                f"{empty} chunks tokenized to empty, {self.n_empty_fallback}/{len(texts)} 走了字符兜底"
                " (check tokenizer/stopwords/userdict)"
            )
        self.retriever.index(list(tokens), show_progress=False)
        self.chunk_ids = [str(c) for c in chunk_ids]

    # ------------------------------------------------------------------ 持久化
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.dir))
        indptr = np.load(self.dir / "indptr.csc.index.npy")
        self.avgdl = float(np.mean(np.diff(indptr))) if indptr.size > 1 else 0.0
        with (self.dir / "chunk_ids.jsonl").open("w", encoding="utf-8") as fh:
            for cid in self.chunk_ids:
                fh.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")
        meta = {
            "N": len(self.chunk_ids),
            "n_empty_fallback": self.n_empty_fallback,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "method": self.method,
        }
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def load(self) -> None:
        import bm25s

        missing = [f for f in _BM25_FILES if not (self.dir / f).exists()]
        if missing or not (self.dir / "chunk_ids.jsonl").exists():
            raise FileNotFoundError(f"bm25 index incomplete, missing {missing or ['chunk_ids.jsonl']}")
        self.retriever = bm25s.BM25.load(str(self.dir), load_corpus=False)
        ids: list[str] = []
        with (self.dir / "chunk_ids.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    ids.append(str(json.loads(line)["chunk_id"]))
        self.chunk_ids = ids
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.n_empty_fallback = int(meta.get("n_empty_fallback") or 0)
            self.avgdl = float(meta.get("avgdl") or 0.0)
            self.k1 = float(meta.get("k1", self.k1))
            self.b = float(meta.get("b", self.b))

    # ------------------------------------------------------------------ search
    def search(self, query: str, topk: int = 50) -> list[tuple[str, float]]:
        tokens = self.tokenizer(query)
        if not tokens or not self.chunk_ids:
            return []
        res, scores = self.retriever.retrieve([tokens], k=int(topk), show_progress=False)
        out: list[tuple[str, float]] = []
        for idx, score in zip(res[0], scores[0], strict=False):
            value = float(score)
            if value <= 0:
                continue
            i = int(idx)
            if 0 <= i < len(self.chunk_ids):
                out.append((self.chunk_ids[i], value))
        return out
