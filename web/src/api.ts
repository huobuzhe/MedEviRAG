export type Status = "answered" | "partial" | "refused";

export interface CitationOut {
  chunk_id: string; doc_id: string; doc_title: string; section_path: string[];
  page: number; quote: string; url: string | null; evidence_level: string;
}
export interface AskResponse {
  trace: { trace_id: string; cache: string; total_ms: number; nodes: Record<string, number> };
  status: Status; answer: string; citations: CitationOut[];
  confidence: number; refusal_reason: string | null; disclaimer: string;
}
export interface CitationDetail {
  chunk_id: string; parent_id: string; doc_title: string; section_path: string[];
  chunk_type: string; text: string; context_header: string;
  page_start: number; page_end: number;
  locator: { page: number; block_index: number; bbox: number[] };
  evidence_level: string; rec_strength: string;
}

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" },
                               body: JSON.stringify(body) });
  if (!r.ok) {
    const e = await r.json().catch(() => ({ code: "INTERNAL", message: r.statusText }));
    throw new Error(`${e.code}: ${e.message} (trace ${e.trace_id ?? "-"})`);
  }
  return r.json() as Promise<T>;
}

export const askQuestion = (question: string) =>
  jpost<AskResponse>("/v1/ask", { question, top_k: 50, use_cache: true });

export const fetchCitation = async (chunkId: string): Promise<CitationDetail> => {
  const r = await fetch(`/v1/citations/${encodeURIComponent(chunkId)}`);
  if (r.status === 404) throw new Error("该引用已失效（chunk_id 不存在）");
  if (!r.ok) throw new Error(`引用拉取失败 ${r.status}`);
  return r.json() as Promise<CitationDetail>;
};

export const sendFeedback = (traceId: string, rating: 1 | -1) =>
  jpost<{ id: number }>("/v1/feedback", { trace_id: traceId, rating });
