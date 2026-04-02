/** API base: use VITE_API_URL for cross-origin; dev uses vite proxy with "". */
const RAW = import.meta.env.VITE_API_URL as string | undefined;
export const API_BASE = (RAW ?? "").replace(/\/$/, "");

export type RubricScores = {
  factual_accuracy: number;
  conceptual_completeness: number;
  reasoning: number;
  domain_terminology: number;
};

export type QuestionResult = {
  question_id: string;
  student_answer: string;
  awarded_marks: number;
  max_marks: number;
  rubric_scores: RubricScores;
  feedback: string;
  grading_confidence: string;
  ocr_confidence: number;
  flags: string[];
};

export type EvaluateResponse = {
  result_id: string;
  sheet_id: string;
  exam_id: string;
  total_marks: number;
  max_total: number;
  confidence_flag: boolean;
  grading_confidence: string;
  prompt_hash: string;
  llm_model: string;
  flags: string[];
  questions: QuestionResult[];
};

async function req(path: string, init?: RequestInit) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = (j.detail as string) || JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

export async function fetchIntegrations() {
  const r = await req("/api/integrations");
  return r.json();
}

export async function uploadKeyJson(body: object) {
  const r = await req("/upload/key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json() as Promise<{ key_ids: string[] }>;
}

export async function uploadKeyFile(
  file: File,
  options?: { examId?: string; defaultMaxMarks?: number },
) {
  const fd = new FormData();
  fd.append("file", file);
  if (options?.examId?.trim()) fd.append("exam_id", options.examId.trim());
  if (options?.defaultMaxMarks != null) fd.append("default_max_marks", String(options.defaultMaxMarks));
  const r = await req("/upload/key/file", { method: "POST", body: fd });
  return r.json() as Promise<{ key_ids: string[] }>;
}

export async function uploadSheet(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await req("/upload/sheet", { method: "POST", body: fd });
  return r.json() as Promise<{ sheet_id: string; filename: string }>;
}

export async function evaluate(payload: {
  sheet_id: string;
  exam_id: string;
  top_k?: number;
  expected_num_regions?: number | null;
  use_consensus_ocr?: boolean;
}): Promise<EvaluateResponse> {
  const r = await req("/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return r.json() as Promise<EvaluateResponse>;
}

export async function fetchResult(resultId: string): Promise<EvaluateResponse> {
  const r = await req(`/result/${resultId}`);
  return r.json() as Promise<EvaluateResponse>;
}

export async function fetchRubric(resultId: string) {
  const r = await req(`/result/${resultId}/rubric`);
  return r.json();
}

export function pdfUrl(resultId: string): string {
  return `${API_BASE}/report/${resultId}/pdf`;
}

export function sheetDownloadUrl(sheetId: string): string {
  return `${API_BASE}/sheet/${sheetId}/file`;
}
