import { useState } from "react";
import { Link } from "react-router-dom";
import { sheetDownloadUrl, uploadSheet } from "../api";

export default function UploadSheet() {
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [sheetId, setSheetId] = useState("");

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setErr("");
    setMsg("");
    setLoading(true);
    try {
      const out = await uploadSheet(f);
      setSheetId(out.sheet_id);
      setMsg(`Stored sheet ${out.sheet_id} (${out.filename}). Use this sheet_id on the Evaluate page.`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Answer sheet upload — Interface B</h2>
        <p>
          This screen is only for the <strong>student scan</strong> (image or PDF), not the key. Backend:{" "}
          <code>POST /upload/sheet</code>.
        </p>
        <input type="file" accept="image/png,image/jpeg,application/pdf,.png,.jpg,.jpeg,.pdf" onChange={onFile} disabled={loading} />
      </div>
      {sheetId && (
        <div className="card">
          <h2>Stored sheet</h2>
          <p>
            <code>{sheetId}</code> —{" "}
            <a href={sheetDownloadUrl(sheetId)} target="_blank" rel="noreferrer">
              Download original
            </a>{" "}
            ·{" "}
            <Link to={`/evaluate?sheet_id=${encodeURIComponent(sheetId)}`}>Go to Evaluate</Link>
          </p>
        </div>
      )}
      {msg && <div className="message ok">{msg}</div>}
      {err && <div className="message err">{err}</div>}
    </div>
  );
}
