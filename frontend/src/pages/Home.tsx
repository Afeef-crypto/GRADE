import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE, fetchIntegrations } from "../api";

export default function Home() {
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetchIntegrations()
      .then(setStatus)
      .catch((e: Error) => setErr(e.message));
  }, []);

  return (
    <div>
      <div className="card">
        <h2>Workflow</h2>
        <p>
          <strong>Fast path:</strong> <Link to="/upload">Upload key + sheet on one page</Link>, then{" "}
          <Link to="/evaluate">Evaluate</Link>. You can still use <Link to="/upload-key">key only</Link> or{" "}
          <Link to="/upload-sheet">sheet only</Link> if you prefer.
        </p>
        <p>
          API base: <code>{API_BASE || "(same origin — Vite proxy)"}</code>
        </p>
      </div>
      <div className="card">
        <h2>Phase integration status</h2>
        {err && <div className="message err">{err}</div>}
        {!err && status && (
          <pre style={{ overflow: "auto", fontSize: "0.8rem" }}>{JSON.stringify(status, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
