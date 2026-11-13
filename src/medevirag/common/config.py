"""四层 YAML + 环境变量覆盖的强类型配置（docs/01 §T2）。

设计要点
--------
1. 一个 BaseSettings 吃掉 4 个 YAML，yaml_file 传 tuple（实测 list 在部分版本会被当成
   单个路径字符串）。
2. 优先级：显式入参 > 环境变量 > .env > YAML。
3. env_prefix="MEDEVIRAG__" 配 env_nested_delimiter="__"：前缀里的双下划线与嵌套分隔符
   对齐，MEDEVIRAG__RETRIEVAL__TOPK 才能落到 retrieval.topk。
4. config_hash() 把整份配置序列化取 sha256 前 12 位，成为评测报告里"当时配置"的指纹。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CFG_DIR = Path(__file__).resolve().parents[3] / "configs"
"""src/medevirag/common/config.py -> 上溯三级 -> F:/MedEviRAG/configs。"""


class _Base(BaseModel):
    """所有配置段的基类：关闭 model_ 保护前缀，允许 model_path / model 之类字段名。"""

    model_config = ConfigDict(protected_namespaces=())


class AppSection(_Base):
    name: str = "medevirag"
    env: Literal["dev", "test", "prod"] = "dev"
    data_dir: str = "data"
    log_level: str = "INFO"
    log_json: bool = True


class ProviderCfg(_Base):
    base_url: str
    model: str
    temperature: float = 0.0
    proxy: str | None = None
    """None 表示直连。空字符串会被 httpx 当成"要开代理"，因此只允许 null。"""
    api_key_env: str
    """只存环境变量名，永远不存明文 Key。"""
    timeout_s: float = 60.0


class LLMSection(_Base):
    default: str = "siliconflow"
    proxy: str | None = None
    providers: dict[str, ProviderCfg] = Field(default_factory=dict)


class EmbeddingSection(_Base):
    model_path: str = "models/bge-m3"
    device: str = "cuda"
    batch_size: int = 16
    dim: int = 1024
    max_length: int = 512
    proxy: str | None = None


class RerankerSection(_Base):
    model_path: str = "models/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    proxy: str | None = None


class RetrievalSection(_Base):
    topk: int = 10
    dense_topk: int = 50
    sparse_topk: int = 50
    rrf_k: int = 60


class FiltersSection(_Base):
    max_publish_age_years: int = 10
    min_evidence_level: str = "C"


class BudgetSection(_Base):
    max_total_tokens: int = 2_000_000
    per_run_warn: int = 200_000


class JudgeSection(_Base):
    provider: str = "dashscope"
    model: str = "qwen3.8-flash"


class SplitSection(_Base):
    dev_ratio: float = 0.8
    seed: int = 20240601


class ThresholdsSection(_Base):
    recall_at_10: float = 0.85
    mrr: float = 0.62
    ndcg_at_10: float = 0.70
    citation_grounding: float = 0.90
    sentence_support: float = 0.85
    faithfulness: float = 0.80
    refusal_recall: float = 0.85
    over_refusal: float = 0.12
    refusal_reason_acc: float = 0.80
    p95_latency_s: float = 6.0


class ConfidenceSection(_Base):
    """置信度综合评分配置（docs/05 §T1/T2）。

    weights 四分量之和必须等于 1.0（加载时断言）。platt 里两套参数分别给
    rank / margin 用；若将来重排器重新启用，rank 改用 rerank logit 参数
    （mid=2.0 scale=1.5，见 confidence.py 的 compute_confidence）。
    """

    weights: dict[str, float] = Field(
        default_factory=lambda: {"rank": 0.35, "citation": 0.30, "evidence": 0.20, "margin": 0.15}
    )
    platt: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "rank": {"mid": 2.0, "scale": 1.5},
            "margin": {"mid": 1.0, "scale": 1.2},
            "rank_dense": {"mid": 0.670, "scale": 0.060},
            "margin_dense": {"mid": 0.1385, "scale": 0.1693},
        }
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {"answered": 0.65, "partial": 0.45}
    )


class RefusalSection(_Base):
    """安全拒答规则阈值（docs/05 §T1/T3-T5）。

    扁平 10 键 + now_year；权重 sum 已在 ConfidenceSection 校验。
    """

    now_year: int = 2025
    stale_window_years: int = 10
    stale_top1_year: int = 2015
    low_top1_score: float = 0.35
    small_margin_raw: float = 0.05
    max_grounding_fail_rate: float = 0.30
    min_sentence_support: float = 0.50
    unknown_ratio_trigger: float = 0.60
    unknown_level_ratio: float = 1.00
    unknown_level_conf_cap: float = 0.55


class EvalJudgeSection(_Base):
    provider: str = "dashscope"
    model: str = "qwen3.8-flash"
    temperature: float = 0.0
    max_tokens: int = 512
    evidence_order_seed: int = 7
    max_retry: int = 2
    min_agree_kappa: float = 0.60


class EvalGenSection(_Base):
    provider: str = "siliconflow"
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    temperature: float = 0.0


class EvalThresholdsSection(_Base):
    """00 §9 阈值；over_refusal_rate/judge_error_rate 是上限型。"""

    recall_at_10: float = 0.85
    mrr: float = 0.62
    ndcg_at_10: float = 0.70
    citation_landing_rate: float = 0.90
    sentence_support_rate: float = 0.85
    faithfulness: float = 0.80
    refusal_recall: float = 0.85
    over_refusal_rate: float = 0.12
    refusal_reason_accuracy: float = 0.80
    judge_error_rate: float = 0.05
    ece: float = 0.10


class EvalRegressionSection(_Base):
    max_drop: float = 0.02
    hard_drop: float = 0.05


class EvalSection(_Base):
    set_path: str = "data/eval/eval_set.jsonl"
    freeze_path: str = "data/eval/freeze.json"
    reports_dir: str = "data/reports"
    sample_seed: int = 20240501
    judge: EvalJudgeSection = Field(default_factory=EvalJudgeSection)
    generator: EvalGenSection = Field(default_factory=EvalGenSection)
    thresholds: EvalThresholdsSection = Field(default_factory=EvalThresholdsSection)
    regression: EvalRegressionSection = Field(default_factory=EvalRegressionSection)

class ServeCacheSection(_Base):
    embedding_ttl_s: int = 604800
    answer_ttl_s: int = 3600


class ServeBudgetSection(_Base):
    retrieval: int = 800
    rerank: int = 600
    generation: int = 3500
    total: int = 6000


class ServeConcurrencySection(_Base):
    max_inflight: int = 32
    rerank_gate: int = 1
    rerank_wait_ms: int = 400
    llm_semaphore: int = 8


class ServeSection(_Base):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:5173", "http://localhost:5173"])
    cache: ServeCacheSection = Field(default_factory=ServeCacheSection)
    budget_ms: ServeBudgetSection = Field(default_factory=ServeBudgetSection)
    concurrency: ServeConcurrencySection = Field(default_factory=ServeConcurrencySection)


class ServeFallbackSection(_Base):
    """无 Docker/Redis/PG 的降级路径（本机实测环境）。"""
    redis_url: str = ""
    pg_dsn: str = ""
    audit_dir: str = "data/logs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDEVIRAG__",
        env_nested_delimiter="__",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=tuple(
            str(CFG_DIR / f) for f in ("app.yaml", "models.yaml", "retrieval.yaml", "eval.yaml")
        ),
        yaml_file_encoding="utf-8",
    )

    app: AppSection = Field(default_factory=AppSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    reranker: RerankerSection = Field(default_factory=RerankerSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    filters: FiltersSection = Field(default_factory=FiltersSection)
    budget: BudgetSection = Field(default_factory=BudgetSection)
    judge: JudgeSection = Field(default_factory=JudgeSection)
    split: SplitSection = Field(default_factory=SplitSection)
    thresholds: ThresholdsSection = Field(default_factory=ThresholdsSection)
    confidence: ConfidenceSection = Field(default_factory=ConfidenceSection)
    refusal: RefusalSection = Field(default_factory=RefusalSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    serve: ServeSection = Field(default_factory=ServeSection)
    serve_fallback: ServeFallbackSection = Field(default_factory=ServeFallbackSection)

    @model_validator(mode="after")
    def _check_confidence_weights(self) -> Settings:
        w = self.confidence.weights
        if abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError(f"confidence.weights 之和必须为 1.0，实际为 {sum(w.values())!r}")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def config_hash(self) -> str:
        """整份配置的指纹；排除日志级别，因为它不影响任何实验结果。"""
        payload: dict[str, Any] = self.model_dump(mode="json")
        payload.get("app", {}).pop("log_level", None)
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()[:12]

    def provider(self, name: str | None = None) -> ProviderCfg:
        key = name or self.llm.default
        if key not in self.llm.providers:
            raise KeyError(f"未配置的 provider: {key!r}；可用: {sorted(self.llm.providers)}")
        return self.llm.providers[key]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内只解析一次；改 .env 后必须重开进程（配置不应静默漂移）。"""
    return Settings()
