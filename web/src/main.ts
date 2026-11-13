import { askQuestion, fetchCitation, sendFeedback, type AskResponse } from "./api";

const $q = document.querySelector<HTMLInputElement>("#q")!;
const $go = document.querySelector<HTMLButtonElement>("#go")!;
const $ans = document.querySelector<HTMLDivElement>("#answer")!;
const $side = document.querySelector<HTMLDivElement>("#citations")!;

const STATUS_TEXT: Record<string, string> = {
  answered: "已依据证据回答", partial: "证据不足，仅部分回答", refused: "已拒答",
};

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">");
}

function renderAnswer(res: AskResponse): void {
  const head = `<div class="meta ${res.status}">
      <span class="pill">${STATUS_TEXT[res.status]}</span>
      <span>置信度 ${res.confidence.toFixed(2)}</span>
      <span>耗时 ${res.trace.total_ms} ms · trace <code>${res.trace.trace_id}</code></span>
    </div>`;
  const body = escapeHtml(res.answer).replace(/\[C(\d+)\]/g, (_m, n: string) => {
    const ok = Number(n) <= res.citations.length;
    return ok ? `<button class="cite" data-idx="${n}">[C${n}]</button>` : `<span class="bad">[C${n}]</span>`;
  }).replace(/\n/g, "<br/>");
  $ans.innerHTML = head + `<p class="ans">${body}</p>` +
    `<p class="disclaimer">${escapeHtml(res.disclaimer)}</p>`;
  $ans.querySelectorAll<HTMLButtonElement>("button.cite").forEach((btn) => {
    btn.onclick = () => scrollToCard(Number(btn.dataset.idx));
  });
}

function renderCitations(res: AskResponse): void {
  if (res.citations.length === 0) {
    $side.innerHTML = `<p class="empty">本次回答无引用（${res.refusal_reason ?? "证据不足"}）</p>`;
    return;
  }
  $side.innerHTML = res.citations.map((c, i) => `
    <article class="card" id="card-${i + 1}" data-chunk="${c.chunk_id}">
      <header><b>[C${i + 1}]</b> ${escapeHtml(c.doc_title)}</header>
      <div class="path">${escapeHtml(c.section_path.join(" › "))} · 第 ${c.page} 页</div>
      <div class="lvl">证据等级 ${c.evidence_level}</div>
      <blockquote>${escapeHtml(c.quote)}</blockquote>
      <button class="more">查看原文</button>
      <div class="detail" hidden></div>
    </article>`).join("");
  $side.querySelectorAll<HTMLButtonElement>("button.more").forEach((btn) => {
    btn.onclick = () => toggleDetail(btn);
  });
}

async function toggleDetail(btn: HTMLButtonElement): Promise<void> {
  const card = btn.closest<HTMLElement>("article.card")!;
  const box = card.querySelector<HTMLDivElement>(".detail")!;
  if (!box.hidden) { box.hidden = true; btn.textContent = "查看原文"; return; }
  btn.disabled = true;
  try {
    const d = await fetchCitation(card.dataset.chunk!);
    box.innerHTML = `<div class="d-head">${escapeHtml(d.doc_title)} › ${escapeHtml(d.section_path.join(" › "))}</div>
      <div class="d-loc">第 ${d.page_start} 页 · bbox [${d.locator.bbox.join(", ")}] · ${d.chunk_type} · 等级 ${d.evidence_level}</div>
      <div class="d-ctx">${escapeHtml(d.context_header)}</div>
      <pre class="d-text">${escapeHtml(d.text)}</pre>`;
    box.hidden = false;
    btn.textContent = "收起原文";
  } catch (e) {
    box.innerHTML = `<div class="err">${escapeHtml((e as Error).message)}</div>`;
    box.hidden = false;
  } finally { btn.disabled = false; }
}

function scrollToCard(idx: number): void {
  const el = document.querySelector<HTMLElement>(`#card-${idx}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

$go.onclick = async () => {
  const question = $q.value.trim();
  if (question.length < 2) { $ans.textContent = "请输入至少 2 个字的问题"; return; }
  $go.disabled = true; $ans.textContent = "检索中…"; $side.innerHTML = "";
  try {
    const res = await askQuestion(question);
    renderAnswer(res); renderCitations(res);
    const fb = document.createElement("div");
    fb.className = "feedback";
    fb.innerHTML = `<button data-r="1">有用</button><button data-r="-1">无用</button><span></span>`;
    fb.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
      b.onclick = async () => {
        await sendFeedback(res.trace.trace_id, Number(b.dataset.r) as 1 | -1);
        fb.querySelector("span")!.textContent = "已记录，谢谢";
      };
    });
    $ans.appendChild(fb);
  } catch (e) {
    $ans.innerHTML = `<div class="err">${escapeHtml((e as Error).message)}</div>`;
  } finally { $go.disabled = false; }
};
