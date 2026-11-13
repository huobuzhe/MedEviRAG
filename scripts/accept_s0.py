"""S0 验收 A1-A12 一键复核（docs/01 §6）。用法：python scripts/accept_s0.py"""
from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, "src")
PY = sys.executable
FAILS: list[str] = []


def say(title: str) -> None:
    print("\n=== " + title + " ===", flush=True)


def py(code: str) -> tuple[int, str]:
    p = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def check(name: str, ok: bool, detail: str) -> None:
    print(("PASS " if ok else "FAIL ") + name + " :: " + detail.replace("\n", " | ")[:220], flush=True)
    if not ok:
        FAILS.append(name)


say("A1 解释器 + GPU + numpy 兼容")
rc, out = py("import platform,torch,numpy;print(platform.python_version(), torch.__version__, torch.cuda.is_available(), numpy.__version__);"
             "x=torch.randn(4,4).cuda(); print('cuda tensor ok', x.detach().cpu().numpy().shape)")
check("A1", rc == 0 and "True" in out and "cuda tensor ok" in out, out)

say("A2 依赖可导入（pip 已装；此处核 requirements.txt 全部包可 import）")
rc, out = py("import importlib.util,re,sys;"
             "mods={'PyMuPDF':'fitz','sentence-transformers':'sentence_transformers','rank-bm25':'rank_bm25',"
             "'psycopg[binary]':'psycopg','python-dotenv':'dotenv','faiss-cpu':'faiss','grpcio':'grpc','pyarrow':'pyarrow'};"
             "names=[l.split('==')[0].strip() for l in open('requirements.txt',encoding='utf-8') if l.strip() and not l.startswith('#')];"
             "bad=[];"
             "[bad.append(n) for n in names if (lambda m: (m is None))(importlib.util.find_spec(mods.get(n,n.replace('-','_'))))];"
             "print('missing', bad); sys.exit(1 if bad else 0)")
check("A2", rc == 0, out)

say("A3 pytest")
p = subprocess.run([PY, "-m", "pytest", "-q"], capture_output=True, text=True, encoding="utf-8", errors="replace")
lines = [ln for ln in p.stdout.splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
check("A3", p.returncode == 0, lines[-1] if lines else p.stdout[-160:])

say("A4 ruff + mypy")
r1 = subprocess.run([PY, "-m", "ruff", "check", ".", "--output-format=concise"], capture_output=True, text=True, encoding="utf-8", errors="replace")
r2 = subprocess.run([PY, "-m", "mypy"], capture_output=True, text=True, encoding="utf-8", errors="replace")
check("A4-ruff", r1.returncode == 0, r1.stdout.strip()[-160:])
check("A4-mypy", r2.returncode == 0, r2.stdout.strip()[-160:])

say("A5 检索参数 10/50/50/60")
rc, out = py("import sys;sys.path.insert(0,'src');from medevirag.common.config import get_settings as g;"
             "s=g();print(s.retrieval.topk,s.retrieval.dense_topk,s.retrieval.sparse_topk,s.retrieval.rrf_k)")
check("A5", out.strip() == "10 50 50 60", out)

say("A6 过滤器 10 年 / C 级")
rc, out = py("import sys;sys.path.insert(0,'src');from medevirag.common.config import get_settings as g;"
             "s=g();print(s.filters.max_publish_age_years,s.filters.min_evidence_level)")
check("A6", out.strip() == "10 C", out)

say("A7 provider 的 proxy 取值（None 表示显式直连）")
rc, out = py("import sys;sys.path.insert(0,'src');from medevirag.common.config import get_settings as g;"
             "s=g();print(repr(s.llm.providers['siliconflow'].proxy), repr(s.llm.providers['deepseek-official'].proxy))")
check("A7", out.strip() == "None 'http://127.0.0.1:7890'", out)

say("A8 结构化日志每行可 json.loads 且含 trace_id / latency_ms")
rc, out = py("""import sys,json,io,logging;sys.path.insert(0,'src')
from medevirag.common.logging import configure_logging,get_logger,bind_trace,JsonFormatter
configure_logging('INFO',True);bind_trace('t-001')
buf=io.StringIO();h=logging.StreamHandler(buf);h.setFormatter(JsonFormatter())
logging.getLogger().addHandler(h)
get_logger('acc').info('boot',extra={'node':'doctor','latency_ms':2.5})
d=json.loads(buf.getvalue().strip())
print('ok' if d['trace_id']=='t-001' and 'latency_ms' in d else 'bad')""")
check("A8", out.splitlines()[0].strip() == "ok", out)

say("A9 FakeLLM 确定性 + 四剧本可辨识")
rc, out = py("""import sys;sys.path.insert(0,'src')
from medevirag.common.llm import FakeLLM,LLMRequest
r=LLMRequest(system='s',user='高血压一线用药？')
f=FakeLLM();same=f.complete(r).text==f.complete(r).text
texts={m:FakeLLM(mode=m).complete(r).text for m in ('normal','bad_citation','refuse','malformed_json')}
print('same',same,'distinct',len(set(texts.values())),'C9','[C9]' in texts['bad_citation'])""")
check("A9", "same True distinct 4 C9 True" in out, out)

say("A10 真实 API 调用数 = 0")
rc, out = py("import sys;sys.path.insert(0,'src');from medevirag.common.token_guard import LEDGER;print('calls',LEDGER.calls)")
check("A10", out.strip() == "calls 0", out)

say("A11 三个反例各抛 ValueError + RefusalReason 15 值")
rc, out = py("""import sys;sys.path.insert(0,'src')
from medevirag.common.types import Chunk,Locator,RefusalReason
base=dict(chunk_id='a',parent_id='p',doc_id='d',doc_title='t',doc_type='guideline',publisher='pub',
 publish_year=2024,lang='zh',section_path=['1'],section_level=3,page_start=1,page_end=1,
 chunk_type='paragraph',text='t',context_header='h',evidence_level='A',rec_strength='I',
 grade_system='GRADE',token_len=9,content_hash='sha256:aa',locator=Locator(1,0,(0.0,0.0,1.0,1.0)))
n=0
for c in ({'content_hash':'bad'},{'page_start':5,'page_end':2},{'locator':Locator(1,0,None)}):
    kw=dict(base);kw.update(c)
    try: Chunk(**kw)
    except ValueError: n+=1
print('raised',n,'reasons',len(list(RefusalReason)))""")
check("A11", "raised 3 reasons 15" in out, out)

say("A12 docker compose 语法门禁（无 daemon 也应退出码 0 并列出 6 服务）")
p = subprocess.run(["docker", "compose", "-f", "docker/docker-compose.yml", "config", "--services"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
svcs = [s for s in p.stdout.split() if s]
check("A12", p.returncode == 0 and len(svcs) == 6, f"exit={p.returncode} services={'/'.join(sorted(svcs))}")  # S6 六服务

say("结果汇总")
print("FAILED: " + (", ".join(FAILS) if FAILS else "无，A1-A12 全通过"), flush=True)
raise SystemExit(1 if FAILS else 0)
