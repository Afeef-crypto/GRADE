import { useState } from "react";
import { uploadKeyFile, uploadKeyJson } from "../api";

const SAMPLE = `{
  "exam_id": "demo-exam",
  "questions": [
    { "question_id": "Q1", "expected_answer": "Photosynthesis converts light to chemical energy.", "max_marks": 4 },
    { "question_id": "Q2", "expected_answer": "Newton's second law is F = ma.", "max_marks": 4 }
  ]
}`;

export default function UploadKey() {
  const [examId, setExamId] = useState("demo-exam");
  const [jsonText, setJsonText] = useState(SAMPLE);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  async function submitJson() {
    setErr("");
    setMsg("");
    setLoading(true);
    try {
      const data = JSON.parse(jsonText);
      if (!data.exam_id) data.exam_id = examId;
      const out = await uploadKeyJson(data);
      setMsg(`Uploaded ${out.key_ids.length} key row(s). IDs: ${out.key_ids.slice(0, 3).join(", ")}${out.key_ids.length > 3 ? "…" : ""}`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setErr("");
    setMsg("");
    setLoading(true);
    try {
      const out = await uploadKeyFile(f, { examId: examId.trim() || undefined, defaultMaxMarks: 4 });
      setMsg(`Key file accepted. ${out.key_ids.length} row(s) stored.`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Answer key upload — Interface A</h2>
        <p>
          Model answer key: JSON with <code>exam_id</code> + <code>questions[]</code>, or a <strong>text-based PDF</strong>{" "}
          with numbered answers (<code>1.</code> <code>2.</code>). PDF uploads require <strong>exam_id</strong> below.
        </p>
        <label htmlFor="exam">Default exam_id (used if omitted in JSON)</label>
        <input id="exam" value={examId} onChange={(e) => setExamId(e.target.value)} />
        <label htmlFor="json">JSON body → POST /upload/key</label>
        <textarea id="json" value={jsonText} onChange={(e) => setJsonText(e.target.value)} spellCheck={false} />
        <button type="button" disabled={loading} onClick={submitJson}>
          Submit JSON
        </button>
      </div>
      <div className="card">
        <h2>Answer key file — Interface A (file)</h2>
        <p>
          <code>POST /upload/key/file</code> — append form fields <code>exam_id</code> (required for PDF) and optional{" "}
          <code>default_max_marks</code>.
        </p>
        <input type="file" accept=".json,.pdf,application/json,application/pdf" onChange={onFile} disabled={loading} />
      </div>
      {msg && <div className="message ok">{msg}</div>}
      {err && <div className="message err">{err}</div>}
    </div>
  );
}
