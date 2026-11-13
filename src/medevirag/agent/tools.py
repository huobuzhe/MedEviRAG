"""P2：Agent 工具注册中心。工具名沿用两个真实项目里的名字，参数用 Pydantic 校验，错误回传不吞。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""


class SearchInput(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=5, ge=1, le=20)


class NutritionInput(BaseModel):
    ingredients: list[str] = Field(min_length=1)


class RetrieveInput(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=10, ge=1, le=50)


class CitationInput(BaseModel):
    answer: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class InteractionInput(BaseModel):
    drug: str = Field(min_length=2)
    food: str = Field(min_length=2)


SCHEMAS: dict[str, type[BaseModel]] = {
    "search_local_recipes": SearchInput,
    "calculate_nutrition": NutritionInput,
    "retrieve_evidence": RetrieveInput,
    "verify_citation": CitationInput,
    "drug_food_interaction": InteractionInput,
}


def _search_local_recipes(a: SearchInput) -> Any:
    return {"hits": [{"id": "r1", "title": "低钠番茄炒蛋", "score": 0.91}], "k": a.k}


def _calculate_nutrition(a: NutritionInput) -> Any:
    return {"calories": 186.0, "sodium_mg": 312.0, "ingredients": a.ingredients}


def _retrieve_evidence(a: RetrieveInput) -> Any:
    return {"chunk_ids": ["CN-HTN-2024#3.2.1#c01"], "top_k": a.top_k}


def _verify_citation(a: CitationInput) -> Any:
    return {"landing_ok": "[1]" in a.answer, "checked": len(a.evidence)}


def _drug_food_interaction(a: InteractionInput) -> Any:
    known = ("华法林", "维生素K")
    return {"risk": "high" if all(k in (a.drug + a.food) for k in known) else "low",
            "drug": a.drug, "food": a.food}


IMPLS: dict[str, Callable[[Any], Any]] = {
    "search_local_recipes": _search_local_recipes,
    "calculate_nutrition": _calculate_nutrition,
    "retrieve_evidence": _retrieve_evidence,
    "verify_citation": _verify_citation,
    "drug_food_interaction": _drug_food_interaction,
}

DANGEROUS = {"drug_food_interaction"}


class ToolRegistry:
    def __init__(self) -> None:
        self.names = tuple(SCHEMAS)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in SCHEMAS:
            return ToolResult(False, error="UNKNOWN_TOOL: " + name)
        try:
            parsed = SCHEMAS[name].model_validate(args)
        except ValidationError as e:
            return ToolResult(False, error="PARAM_INVALID: " + e.errors()[0]["msg"])
        try:
            return ToolResult(True, data=IMPLS[name](parsed))
        except Exception as e:
            return ToolResult(False, error="TOOL_RAISED: " + type(e).__name__ + ": " + str(e)[:80])

    def is_dangerous(self, name: str) -> bool:
        return name in DANGEROUS
