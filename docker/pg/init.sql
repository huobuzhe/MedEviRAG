-- S6 T7：只追加索引，不改 00 §6.2 表结构（docs/07 §T7）。
CREATE INDEX IF NOT EXISTS idx_chunks_doc        ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section    ON chunks (section_id);
CREATE INDEX IF NOT EXISTS idx_sections_doc      ON sections (doc_id, order_index);
CREATE INDEX IF NOT EXISTS idx_qa_log_trace      ON qa_log (trace_id);
CREATE INDEX IF NOT EXISTS idx_qa_log_created    ON qa_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_trace    ON feedback (trace_id);
CREATE INDEX IF NOT EXISTS idx_qa_log_chunk_ids  ON qa_log USING gin (retrieved_chunk_ids);
