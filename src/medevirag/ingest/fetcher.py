"""语料获取（docs/02 4.1 T1）。

两条通道：
1. 站内 PDF（本机语料来源是上海疾控健康主题站）：直接下载/续传；
2. 开放获取全文（PMC OA）：**Europe PMC 优先，NCBI 作可选降级**。

为什么改道：本机实测 NCBI eutils 与 pmc 直连/走代理均返回 HTTP 000（连不上），
而 Europe PMC REST 的 fullTextXML 端点稳定可用。两个通道都不可用时，本模块
只负责把 blocked 如实记进 manifest 并返回状态，绝不抛异常让整条流水线失败——
语料的可得性不是 S1 的失败条件，报告里的诚实度才是。

合规：本文件所有下载物仅用于本地 RAG 技术验证，文件不入版本库、不入制品、
不随服务对外提供，见 COMPLIANCE。
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from medevirag.common.logging import get_logger

_log = get_logger(__name__)

UA = "MedEviRAG/0.1 (local research prototype; contact: local-user)"
PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPEPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

COMPLIANCE = (
    "仅用于本地 RAG 技术验证与效果评估，不做任何形式的分发、再发布或商用；"
    "文件不入版本库（.gitignore 排除）、不入制品、不随服务对外提供。"
)

# Windows 文件名非法字符，落盘前统一替换
_ILLEGAL = re.compile(r'[<>:"|?*\x00-\x1f]')
_PDF_MAGIC = b"%PDF-"


@dataclass(slots=True)
class FetchRecord:
    """一条语料的获取记录，直接序列化进 manifest。"""

    doc_id: str
    tier: str  # web_pdf | pmc_oa | local
    source_url: str
    file_name: str
    sha256: str
    bytes: int
    fetched_at: str
    license: str
    compliance_note: str
    redistribution: str
    resume_from_bytes: int
    status: str = "ok"  # ok | blocked | rejected
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "doc_id": self.doc_id,
            "tier": self.tier,
            "source_url": self.source_url,
            "file_name": self.file_name,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "fetched_at": self.fetched_at,
            "license": self.license,
            "compliance_note": self.compliance_note,
            "redistribution": self.redistribution,
            "resume_from_bytes": self.resume_from_bytes,
            "status": self.status,
            "note": self.note,
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


def sanitize_file_name(name: str) -> str:
    """去掉 Windows 非法字符与控制字符，保留可读的中文文件名。"""
    cleaned = _ILLEGAL.sub("_", name).strip(" .")
    return cleaned or "unnamed"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """返回 "sha256:" + hex，与 common/types 的 content_hash 前缀约定一致。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return "sha256:" + h.hexdigest()


def is_pdf(path: Path) -> bool:
    """落盘校验：魔数不是 %PDF- 的一律判失败，避免把 HTML 错误页当 PDF 解析。"""
    try:
        with path.open("rb") as fh:
            return fh.read(5) == _PDF_MAGIC
    except OSError:
        return False


def default_headers() -> dict[str, str]:
    return {"User-Agent": UA, "Accept": "*/*"}


def direct_client(timeout: float = 30.0) -> httpx.Client:
    """强制直连的 client。站内站点实测直连可用，走系统代理反而 000。"""
    return httpx.Client(
        headers=default_headers(),
        timeout=timeout,
        follow_redirects=True,
        trust_env=False,
    )


def proxy_client(proxy: str | None, timeout: float = 30.0) -> httpx.Client:
    """按 host 显式指定代理；proxy 为 None 时等价于直连。"""
    kwargs: dict[str, Any] = {
        "headers": default_headers(),
        "timeout": timeout,
        "follow_redirects": True,
        "trust_env": False,
    }
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def download_resumable(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    min_delay: float = 1.5,
    max_delay: float = 3.0,
    max_tries: int = 4,
) -> int:
    """带 .part 临时文件 + Range 续传的下载；返回"续传起点字节数"。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start_bytes = 0
    last_err: Exception | None = None

    for attempt in range(1, max_tries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": "bytes=" + str(have) + "-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416:
                    # 已经下全了：服务端拒绝越界 Range，直接转正
                    part.replace(dest)
                    return start_bytes
                if r.status_code == 206:
                    start_bytes = have
                    with part.open("ab") as fh:
                        for buf in r.iter_bytes(1 << 16):
                            fh.write(buf)
                elif r.status_code == 200:
                    start_bytes = 0
                    with part.open("wb") as fh:
                        for buf in r.iter_bytes(1 << 16):
                            fh.write(buf)
                else:
                    r.raise_for_status()
            part.replace(dest)
            # 站点礼貌退避：两次请求之间随机停顿，不做并发轰炸
            time.sleep(random.uniform(min_delay, max_delay))
            return start_bytes
        except (httpx.HTTPError, OSError) as exc:
            last_err = exc
            _log.warning(
                "download_retry",
                extra={"url": url, "attempt": attempt, "have": have, "err": str(exc)[:200]},
            )
            if attempt >= max_tries:
                break
            wait = min(30.0, 2.0**attempt) + random.uniform(0.0, 1.0)
            time.sleep(wait)

    raise RuntimeError("download failed after " + str(max_tries) + " tries: " + url) from last_err


def _xml_text(xml: str, tag: str) -> str:
    m = re.search(r"<" + tag + r"[^>]*>(.*?)</" + tag + ">", xml, re.S)
    return m.group(1).strip() if m else "unknown"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def fetch_pmc_oa(client: httpx.Client, pmcid: str, out_dir: Path) -> FetchRecord:
    """取 PMC 开放获取全文。

    优先 Europe PMC fullTextXML（本机可用），失败再试 NCBI OA 子集 + efetch（本机不可用）。
    两条通道都拿不到时返回 status="blocked" 的 FetchRecord，不抛异常。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = sanitize_file_name(pmcid + ".xml")
    dest = out_dir / fname
    url = EUROPEPMC_FULLTEXT.format(pmcid=pmcid)

    def _ok(source_url: str, tier: str, lic: str, note: str) -> FetchRecord:
        return FetchRecord(
            doc_id=pmcid,
            tier=tier,
            source_url=source_url,
            file_name=fname,
            sha256=sha256_of(dest),
            bytes=dest.stat().st_size,
            fetched_at=_now(),
            license=lic,
            compliance_note=COMPLIANCE,
            redistribution="prohibited",
            resume_from_bytes=0,
            status="ok",
            note=note,
            extra={"license_raw": lic},
        )

    # 通道 1：Europe PMC
    try:
        r = client.get(url)
        if r.status_code == 200 and "<article" in r.text:
            dest.write_text(r.text, encoding="utf-8")
            lic = _xml_text(r.text, "license")
            _log.info("europepmc_ok", extra={"pmcid": pmcid, "bytes": len(r.text)})
            return _ok(url, "pmc_oa", lic, "europepmc fullTextXML")
        _log.warning("europepmc_miss", extra={"pmcid": pmcid, "status": r.status_code})
    except (httpx.HTTPError, OSError) as exc:
        _log.warning("europepmc_err", extra={"pmcid": pmcid, "err": str(exc)[:200]})

    # 通道 2：NCBI（可选降级）
    try:
        r = client.get(PMC_OA, params={"id": pmcid})
        if r.status_code == 200 and "<error" not in r.text.lower():
            r2 = client.get(
                PMC_EFETCH,
                params={
                    "db": "pmc",
                    "id": pmcid.removeprefix("PMC"),
                    "rettype": "xml",
                    "retmode": "xml",
                    "tool": "medevirag",
                    "email": "local-user@example.invalid",
                },
            )
            if r2.status_code == 200 and "<article" in r2.text:
                dest.write_text(r2.text, encoding="utf-8")
                lic = _xml_text(r.text, "license")
                _log.info("ncbi_ok", extra={"pmcid": pmcid, "bytes": len(r2.text)})
                return _ok(PMC_EFETCH, "pmc_oa", lic, "ncbi efetch")
        _log.warning("ncbi_miss", extra={"pmcid": pmcid, "status": r.status_code})
    except (httpx.HTTPError, OSError) as exc:
        _log.warning("ncbi_err", extra={"pmcid": pmcid, "err": str(exc)[:200]})

    return FetchRecord(
        doc_id=pmcid,
        tier="pmc_oa",
        source_url=url,
        file_name=fname,
        sha256="",
        bytes=0,
        fetched_at=_now(),
        license="unknown",
        compliance_note=COMPLIANCE,
        redistribution="prohibited",
        resume_from_bytes=0,
        status="blocked",
        note="europepmc 与 ncbi 两条通道均不可达（本机实测 NCBI 直连/代理 HTTP 000）",
    )


def fetch_pdf(
    client: httpx.Client,
    url: str,
    out_dir: Path,
    doc_id: str,
    *,
    min_delay: float = 1.5,
    max_delay: float = 3.0,
    max_tries: int = 4,
) -> FetchRecord:
    """站内 PDF 下载：文件名取 URL 尾段，下载后校验魔数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_name = url.rstrip("/").rsplit("/", 1)[-1] or (doc_id + ".pdf")
    fname = sanitize_file_name(raw_name)
    dest = out_dir / fname
    resume_from = download_resumable(
        client, url, dest, min_delay=min_delay, max_delay=max_delay, max_tries=max_tries
    )
    ok = is_pdf(dest)
    return FetchRecord(
        doc_id=doc_id,
        tier="web_pdf",
        source_url=url,
        file_name=fname,
        sha256=sha256_of(dest),
        bytes=dest.stat().st_size,
        fetched_at=_now(),
        license="public-web (CDC health education material)",
        compliance_note=COMPLIANCE,
        redistribution="prohibited",
        resume_from_bytes=resume_from,
        status="ok" if ok else "rejected",
        note="" if ok else "落盘内容不是 PDF（魔数校验失败）",
    )


def record_local(doc_id: str, path: Path, source_url: str) -> FetchRecord:
    """本机已有语料的登记路径：不算下载，只补 manifest 字段。"""
    return FetchRecord(
        doc_id=doc_id,
        tier="local",
        source_url=source_url,
        file_name=path.name,
        sha256=sha256_of(path),
        bytes=path.stat().st_size,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime)),
        license="public-web (CDC health education material)",
        compliance_note=COMPLIANCE,
        redistribution="prohibited",
        resume_from_bytes=0,
        status="ok",
        note="本机已有文件，未重新下载",
    )
