import { useState } from "react";
import { Link } from "react-router-dom";
import { sheetDownloadUrl, uploadKeyFile, uploadKeyJson, uploadSheet } from "../api";

const SAMPLE = `{
  "exam_id": "demo-exam",
  "questions": [
    { "question_id": "Q1", "expected_answer": "Photosynthesis converts light to chemical energy.", "max_marks": 4 },
    { "question_id": "Q2", "expected_answer": "Newton's second law is F = ma.", "max_marks": 4 }
  ]
}`;

export default function UploadBoth() {
  const [examId, setExamId] = useState("demo-exam");
  const [jsonText, setJsonText] = useState(SAMPLE);

  const [keyLoading, setKeyLoading] = useState(false);
  const [keyErr, setKeyErr] = useState("");
  const [keyOk, setKeyOk] = useState<string | null>(null);
  const [keyCount, setKeyCount] = useState<number | null>(null);

  const [sheetLoading, setSheetLoading] = useState(false);
  const [sheetErr, setSheetErr] = useState("");
  const [sheetOk, setSheetOk] = useState<string | null>(null);
  const [sheetId, setSheetId] = useState("");

  async function submitKeyJson() {
    setKeyErr("");
    setKeyOk(null);
    setKeyCount(null);
    setKeyLoading(true);
    try {
      const data = JSON.parse(jsonText) as { exam_id?: string };
      if (!data.exam_id) data.exam_id = examId;
      if (data.exam_id) setExamId(String(data.exam_id));
      const out = await uploadKeyJson(data);
      setKeyCount(out.key_ids.length);
      setKeyOk(`Stored ${out.key_ids.length} key row(s) for exam "${data.exam_id}".`);
    } catch (e: unknown) {
      setKeyErr(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyLoading(false);
    }
  }

  async function onKeyFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setKeyErr("");
    setKeyOk(null);
    setKeyCount(null);
    setKeyLoading(true);
    try {
      const buf = await f.arrayBuffer();
      if (f.name.toLowerCase().endsWith(".json")) {
        try {
          const parsed = JSON.parse(new TextDecoder().decode(buf)) as { exam_id?: string };
          if (parsed.exam_id) setExamId(String(parsed.exam_id));
        } catch {
          /* ignore */
        }
      }
      const lower = f.name.toLowerCase();
      const mime =
        f.type ||
        (lower.endsWith(".pdf") ? "application/pdf" : lower.endsWith(".json") ? "application/json" : "application/octet-stream");
      const blobFile = new File([buf], f.name, { type: mime });
      const out = await uploadKeyFile(blobFile, {
        examId: examId.trim() || undefined,
        defaultMaxMarks: 4,
      });
      setKeyCount(out.key_ids.length);
      setKeyOk(`Key file accepted — ${out.key_ids.length} row(s) stored.`);
    } catch (e: unknown) {
      setKeyErr(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyLoading(false);
    }
  }

  async function onSheetFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setSheetErr("");
    setSheetOk(null);
    setSheetId("");
    setSheetLoading(true);
    try {
      const out = await uploadSheet(f);
      setSheetId(out.sheet_id);
      setSheetOk(`Stored answer sheet (${out.filename}).`);
    } catch (e: unknown) {
      setSheetErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSheetLoading(false);
    }
  }

  const keyReady = keyCount !== null && keyCount > 0;
  const readyForEvaluate = Boolean(sheetId && keyReady);
  const evalHref = `/evaluate?sheet_id=${encodeURIComponent(sheetId)}&exam_id=${encodeURIComponent(examId)}`;

  return (
    <div>
      <p className="upload-intro">
        Upload the <strong>answer key</strong> as JSON, or a <strong>text-based PDF</strong> (numbered sections like{" "}
        <code>1.</code> <code>2.</code>) using the exam id below — plus the <strong>student scan</strong> (image or PDF).
      </p>

      <div className="upload-dual-grid">
        <section className="card upload-panel">
          <div className="upload-panel-head">
            <h2>1 · Answer key</h2>
            {keyCount !== null && keyCount > 0 ? <span className="badge ok">Uploaded</span> : <span className="badge muted">Required</span>}
          </div>
          <p className="upload-hint">
          JSON with <code>exam_id</code> and <code>questions[]</code>, or a text PDF. For PDF, <strong>exam id</strong>{" "}
          above is required.
        </p>
          <label htmlFor="both-exam">exam_id default (if missing in JSON)</label>
          <input id="both-exam" value={examId} onChange={(e) => setExamId(e.target.value)} disabled={keyLoading} />
          <label htmlFor="both-json">Key JSON</label>
          <textarea id="both-json" value={jsonText} onChange={(e) => setJsonText(e.target.value)} spellCheck={false} disabled={keyLoading} />
          <div className="upload-actions">
            <button type="button" disabled={keyLoading} onClick={() => void submitKeyJson()}>
              {keyLoading ? "…" : "Submit key JSON"}
            </button>
          </div>
          <label className="upload-file-label">Or upload key file</label>
          <input
            type="file"
            accept=".json,.pdf,application/json,application/pdf"
            onChange={(e) => void onKeyFile(e)}
            disabled={keyLoading}
          />
          {keyOk && <div className="message ok">{keyOk}</div>}
          {keyErr && <div className="message err">{keyErr}</div>}
        </section>

        <section className="card upload-panel">
          <div className="upload-panel-head">
            <h2>2 · Answer sheet (scan)</h2>
            {sheetId ? <span className="badge ok">Uploaded</span> : <span className="badge muted">Required</span>}
          </div>
          <p className="upload-hint">Student handwritten sheet — PNG, JPEG, or PDF.</p>
          <input
            type="file"
            accept="image/png,image/jpeg,application/pdf,.png,.jpg,.jpeg,.pdf"
            onChange={(e) => void onSheetFile(e)}
            disabled={sheetLoading}
          />
          {sheetId && (
            <p className="upload-meta">
              <code>{sheetId}</code> ·{" "}
              <a href={sheetDownloadUrl(sheetId)} target="_blank" rel="noreferrer">
                Download original
              </a>
            </p>
          )}
          {sheetOk && <div className="message ok">{sheetOk}</div>}
          {sheetErr && <div className="message err">{sheetErr}</div>}
        </section>
      </div>

      {readyForEvaluate && (
        <div className="card upload-next">
          <h2>Next step</h2>
          <p>
            Key rows: <strong>{keyCount}</strong> · Exam <code>{examId}</code> · Sheet <code>{sheetId}</code>
          </p>
          <Link className="btn" to={evalHref}>
            Run evaluation
          </Link>
        </div>
      )}

      {!readyForEvaluate && (sheetId || keyReady) && (
        <div className="card upload-next muted-card">
          <p>
            {!keyReady && <span>Upload the answer key above. </span>}
            {!sheetId && <span>Upload the answer sheet above. </span>}
          </p>
        </div>
      )}

      <p className="upload-footnote">
        Separate pages: <Link to="/upload-key">key only</Link> · <Link to="/upload-sheet">sheet only</Link>
      </p>
    </div>
  );
}
