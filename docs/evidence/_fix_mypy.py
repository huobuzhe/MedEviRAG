"""递归参数化所有嵌套泛型注解。"""
from __future__ import annotations
import ast
import re
from pathlib import Path

ROOT = Path("src/medevirag")
SIMPLE = {"dict": "dict[str, Any]", "list": "list[Any]", "set": "set[Any]",
          "Sequence": "Sequence[Any]", "tuple": "tuple[Any, ...]",
          "Mapping": "Mapping[str, Any]"}


def render(node):
    """把注解节点渲染成源码，递归参数化裸泛型；非泛型返回 None。"""
    if isinstance(node, ast.Name):
        return SIMPLE.get(node.id)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        base = node.value.id
        inner = None
        # Python3.10+: slice 直接是节点
        sl = node.slice
        if isinstance(sl, ast.Tuple):
            parts = []
            for elt in sl.elts:
                r_ = render(elt)
                parts.append(r_ if r_ else ast.unparse(elt))
            inner = ", ".join(parts)
        else:
            r_ = render(sl)
            inner = r_ if r_ else ast.unparse(sl)
        return base + "[" + inner + "]"
    return None


def fix_file(p: Path) -> int:
    src = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return -1
    lines = src.splitlines(keepends=True)
    edits = []
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            targets.append(node.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            targets.append(node.returns)
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            targets.append(node.annotation)
    for ann in targets:
        new = render(ann)
        if new and new != ast.unparse(ann):
            edits.append((ann.lineno, ann.col_offset, ann.end_col_offset, new))
    if not edits:
        return 0
    for ln, col, end, new in sorted(set(edits), key=lambda e: (-e[0], -e[1])):
        line = lines[ln - 1]
        lines[ln - 1] = line[:col] + new + line[end:]
    out = "".join(lines)
    if out != src:
        if re.search(r"^from typing import", out, re.M):
            m = re.search(r"^from typing import (.+)$", out, re.M)
            if "Any" not in m.group(1):
                out = out.replace(m.group(0), "from typing import Any, " + m.group(1), 1)
        else:
            out = "from typing import Any\n" + out
        p.write_text(out, encoding="utf-8")
    return len(edits)


total = 0
for f in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in str(f):
        continue
    n = fix_file(f)
    if n > 0:
        total += n
print("递归修复:", total, "处")
