import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { evaluate } from "../api";

export default function Evaluate() {
  const [search] = useSearchParams();
  const navigate = useNavigate();
  const [sheetId, setSheetId] = useState("");
  const [examId, setExamId] = useState("demo-exam");
  const [topK, setTopK] = useState(5); // API allows 1–10
  const [expectedRegions, setExpectedRegions] = useState<string>("");
  const [consensus, setConsensus] = useState(true);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const q = search.get("sheet_id");
    if (q) setSheetId(q);
  }, [search]);

  async function run() {
    setErr("");
    setMsg("");
    if (!sheetId.trim()) {
      setErr("sheet_id is required");
      return;
    }
    setLoading(true);
    try {
      const n = expectedRegions.trim();
      const payload = {
        sheet_id: sheetId.trim(),
        exam_id: examId.trim(),
        top_k: topK,
        expected_num_regions: n === "" ? null : Number(n),
        use_consensus_ocr: consensus,
      };
      if (payload.expected_num_regions !== null && Number.isNaN(payload.expected_num_regions as number)) {
        setErr("expected_num_regions must be a number");
        setLoading(false);
        return;
      }
      const out = await evaluate(payload);
      setMsg(`Evaluation complete. result_id: ${out.result_id}`);
      navigate(`/results?id=${encodeURIComponent(out.result_id)}`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2>Run evaluation</h2>
      <p>POST /evaluate — require a stored sheet_id and an exam_id that has a key uploaded.</p>
      <label htmlFor="sh">sheet_id</label>
      <input id="sh" value={sheetId} onChange={(e) => setSheetId(e.target.value)} placeholder="from Upload sheet" />
      <label htmlFor="ex">exam_id</label>
      <input id="ex" value={examId} onChange={(e) => setExamId(e.target.value)} />
      <label htmlFor="top">top_k (retrieval)</label>
      <input id="top" type="number" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
      <label htmlFor="reg">expected_num_regions (optional)</label>
      <input id="reg" value={expectedRegions} onChange={(e) => setExpectedRegions(e.target.value)} placeholder="leave empty to skip check" />
      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <input type="checkbox" checked={consensus} onChange={(e) => setConsensus(e.target.checked)} />
        use_consensus_ocr
      </label>
      <button type="button" disabled={loading} onClick={run}>
        {loading ? "Running…" : "Evaluate"}
      </button>
      {msg && <div className="message ok">{msg}</div>}
      {err && <div className="message err">{err}</div>}
    </div>
  );
}
