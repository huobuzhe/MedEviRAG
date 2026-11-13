"""PG 落库（docs/02 4.6 T6）。

设计原则：**PG 是可选增强，不是依赖**。S1 的产物是 chunks.jsonl，数据库只是让
S3/S4 的检索少读一次文件。因此本模块必须在没有 psycopg2、没有 PG 服务的机器上
正常 import、正常返回"已跳过"，绝不把 S1 拖死。

幂等：所有写入走 INSERT ... ON CONFLICT DO UPDATE，重复跑同一篇文档不产生重复行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medevirag.common.logging import get_logger

_log = get_logger(__name__)

DDL = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        publisher TEXT NOT NULL,
        publish_year INTEGER NOT NULL,
        lang TEXT NOT NULL,
        source_url TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        page_count INTEGER NOT NULL,
        ingest_status TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sections (
        section_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        parent_section_id TEXT,
        section_no TEXT NOT NULL,
        title TEXT NOT NULL,
        level INTEGER NOT NULL,
        page_start INTEGER NOT NULL,
        page_end INTEGER NOT NULL,
        order_index INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        section_id TEXT REFERENCES sections(section_id) ON DELETE CASCADE,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        chunk_type TEXT NOT NULL,
        text TEXT NOT NULL,
        token_len INTEGER NOT NULL,
        evidence_level TEXT NOT NULL,
        rec_strength TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id)",
]

UPSERT_DOC = """
INSERT INTO documents (doc_id, title, doc_type, publisher, publish_year, lang,
                       source_url, file_path, file_hash, page_count, ingest_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (doc_id) DO UPDATE SET
    title = EXCLUDED.title,
    doc_type = EXCLUDED.doc_type,
    publisher = EXCLUDED.publisher,
    publish_year = EXCLUDED.publish_year,
    lang = EXCLUDED.lang,
    source_url = EXCLUDED.source_url,
    file_path = EXCLUDED.file_path,
    file_hash = EXCLUDED.file_hash,
    page_count = EXCLUDED.page_count,
    ingest_status = EXCLUDED.ingest_status
"""

UPSERT_SECTION = """
INSERT INTO sections (section_id, doc_id, parent_section_id, section_no, title,
                      level, page_start, page_end, order_index)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (section_id) DO UPDATE SET
    parent_section_id = EXCLUDED.parent_section_id,
    section_no = EXCLUDED.section_no,
    title = EXCLUDED.title,
    level = EXCLUDED.level,
    page_start = EXCLUDED.page_start,
    page_end = EXCLUDED.page_end,
    order_index = EXCLUDED.order_index
"""

UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, section_id, doc_id, chunk_type, text,
                    token_len, evidence_level, rec_strength, content_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    section_id = EXCLUDED.section_id,
    chunk_type = EXCLUDED.chunk_type,
    text = EXCLUDED.text,
    token_len = EXCLUDED.token_len,
    evidence_level = EXCLUDED.evidence_level,
    rec_strength = EXCLUDED.rec_strength,
    content_hash = EXCLUDED.content_hash
"""


def psycopg_available() -> bool:
    """检查驱动是否可用，但不 import 它——避免模块级强依赖。"""
    import importlib.util

    return importlib.util.find_spec("psycopg") is not None or importlib.util.find_spec("psycopg2") is not None


def _connect(dsn: str) -> Any:
    try:
        import psycopg

        return psycopg.connect(dsn)
    except ImportError:
        import psycopg2

        return psycopg2.connect(dsn)


def ensure_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        for ddl in DDL:
            cur.execute(ddl)
    conn.commit()


def write_document(conn: Any, doc: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            UPSERT_DOC,
            (
                doc["doc_id"],
                doc["title"],
                doc["doc_type"],
                doc["publisher"],
                doc["publish_year"],
                doc["lang"],
                doc.get("source_url", ""),
                doc.get("file_path", ""),
                doc.get("file_hash", ""),
                doc.get("page_count", 0),
                doc.get("ingest_status", "ok"),
            ),
        )
    conn.commit()


def write_sections(conn: Any, sections: list[dict[str, Any]]) -> None:
    rows = [
        (
            s["section_id"],
            s["doc_id"],
            s.get("parent_section_id"),
            s["section_no"],
            s["title"],
            s["level"],
            s["page_start"],
            s["page_end"],
            s["order_index"],
        )
        for s in sections
    ]
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SECTION, rows)
    conn.commit()


def write_chunks(conn: Any, chunks: list[dict[str, Any]]) -> None:
    rows = [
        (
            c["chunk_id"],
            c.get("parent_id"),
            c["doc_id"],
            c["chunk_type"],
            c["text"],
            c["token_len"],
            c["evidence_level"],
            c["rec_strength"],
            c["content_hash"],
        )
        for c in chunks
    ]
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(UPSERT_CHUNK, rows)
    conn.commit()


def try_load(dsn: str | None) -> tuple[Any | None, str]:
    """尝试连库；任何失败都返回 (None, 原因)，由调用方降级为只写文件。"""
    if not dsn:
        return None, "未配置 PG DSN（--no-db 或默认行为），跳过落库"
    if not psycopg_available():
        return None, "本机未安装 psycopg/psycopg2 驱动，跳过落库"
    try:
        conn = _connect(dsn)
    except Exception as exc:  # 驱动异常类型随版本变，一律降级
        return None, "连接 PG 失败，跳过落库: " + str(exc)[:200]
    try:
        ensure_schema(conn)
    except Exception as exc:  # 建表失败同样降级
        return None, "建表失败，跳过落库: " + str(exc)[:200]
    return conn, "已连接 PG 并完成建表"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    """chunks.jsonl 是 S1 的唯一权威产物；排序保证同一输入两次跑出的字节序列一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: str(r["chunk_id"]))
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows_sorted:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows_sorted)
