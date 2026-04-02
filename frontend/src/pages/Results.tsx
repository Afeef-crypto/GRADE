import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { EvaluateResponse, fetchResult, fetchRubric, pdfUrl, sheetDownloadUrl } from "../api";

type RubricApi = {
  result_id: string;
  dimension_totals: Record<string, number>;
  questions: EvaluateResponse["questions"];
};

export default function Results() {
  const [search] = useSearchParams();
  const [id, setId] = useState("");
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [rubric, setRubric] = useState<RubricApi | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadById(raw: string) {
    const rid = raw.trim();
    if (!rid) {
      setErr("Enter result_id");
      return;
    }
    setErr("");
    setLoading(true);
    try {
      const [r, rub] = await Promise.all([fetchResult(rid), fetchRubric(rid)]);
      setResult(r);
      setRubric(rub as RubricApi);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
      setResult(null);
      setRubric(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const q = search.get("id");
    if (q) {
      setId(q);
      void loadById(q);
    }
    // Intentionally only react to URL ?id=
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  const sheetId = result?.sheet_id;

  return (
    <div>
      <div className="card">
        <h2>Lookup result</h2>
        <label htmlFor="rid">result_id</label>
        <input id="rid" value={id} onChange={(e) => setId(e.target.value)} placeholder="UUID from /evaluate" />
        <button type="button" disabled={loading} onClick={() => void loadById(id)}>
          Load
        </button>
        {id.trim() && (
          <p style={{ marginTop: "1rem" }}>
            <a className="btn secondary" href={pdfUrl(id.trim())} target="_blank" rel="noreferrer">
              Download PDF report
            </a>
            {sheetId ? (
              <>
                {" "}
                <a className="btn secondary" href={sheetDownloadUrl(sheetId)} target="_blank" rel="noreferrer">
                  Original sheet
                </a>
              </>
            ) : null}
          </p>
        )}
      </div>
      {err && <div className="message err">{err}</div>}
      {result && (
        <>
          <div className="card">
            <h2>Scores</h2>
            <p>
              Exam <code>{result.exam_id}</code> · Sheet <code>{result.sheet_id}</code>
            </p>
            <p>
              Total: <strong>{result.total_marks}</strong> / {result.max_total}{" "}
              {result.flags?.length ? <span className="badge warn">Flags: {result.flags.join(", ")}</span> : null}{" "}
              {result.confidence_flag ? <span className="badge warn">Low confidence</span> : <span className="badge ok">OK</span>}
            </p>
            <p style={{ fontSize: "0.9rem", color: "#475569" }}>
              Grading confidence: <strong>{result.grading_confidence}</strong> · Model: <code>{result.llm_model}</code>
            </p>
            <p style={{ fontSize: "0.75rem", color: "#64748b" }}>
              Audit: <code style={{ wordBreak: "break-all" }}>{result.prompt_hash}</code>
            </p>
            <table className="result">
              <thead>
                <tr>
                  <th>Q</th>
                  <th>Awarded</th>
                  <th>Max</th>
                  <th>OCR conf</th>
                  <th>Student (OCR)</th>
                  <th>Feedback</th>
                </tr>
              </thead>
              <tbody>
                {result.questions.map((s) => (
                  <tr key={s.question_id}>
                    <td>{s.question_id}</td>
                    <td>{s.awarded_marks}</td>
                    <td>{s.max_marks}</td>
                    <td>{typeof s.ocr_confidence === "number" ? s.ocr_confidence.toFixed(2) : String(s.ocr_confidence)}</td>
                    <td style={{ maxWidth: "220px", wordBreak: "break-word" }}>{s.student_answer}</td>
                    <td>{s.feedback}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rubric && (
            <div className="card">
              <h2>Rubric totals (across questions)</h2>
              <p style={{ fontSize: "0.85rem", color: "#64748b" }}>
                Each dimension is scored 0–4 per question; bars use max {4 * result.questions.length} for this exam.
              </p>
              <div className="rubric-grid">
                {Object.entries(rubric.dimension_totals).map(([dim, w]) => {
                  const cap = Math.max(1, 4 * result.questions.length);
                  const pct = Math.min(100, Math.round((100 * w) / cap));
                  return (
                    <div key={dim} className="rubric-item">
                      <strong>{dim.replace(/_/g, " ")}</strong>
                      <span>{w}</span>
                      <div
                        style={{
                          marginTop: "0.35rem",
                          height: "6px",
                          borderRadius: "3px",
                          background: "#e2e8f0",
                          overflow: "hidden",
                        }}
                      >
                        <div style={{ width: `${pct}%`, height: "100%", background: "#3d5a80" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div className="card">
            <h2>Per-question rubric (raw)</h2>
            <pre style={{ fontSize: "0.75rem", overflow: "auto" }}>
              {JSON.stringify(
                result.questions.map((q) => ({
                  question_id: q.question_id,
                  rubric_scores: q.rubric_scores,
                  flags: q.flags,
                })),
                null,
                2
              )}
            </pre>
          </div>
        </>
      )}
    </div>
  );
}
