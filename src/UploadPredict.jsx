import React, { useState } from "react";
import { predict } from "./api";

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [genHeatmap, setGenHeatmap] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const data = await predict(file, genHeatmap);
      setResult(data);
    } catch (err) {
      setError(err.message || "请求失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h2>CheXpert 推理 Demo</h2>
      <form onSubmit={onSubmit}>
        <input
          type="file"
          accept="image/*,.png,.jpg,.jpeg,.bmp,.tif,.tiff"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <label style={{ marginLeft: 12 }}>
          <input
            type="checkbox"
            checked={genHeatmap}
            onChange={(e) => setGenHeatmap(e.target.checked)}
          /> 生成热力图
        </label>
        <button type="submit" disabled={!file || loading} style={{ marginLeft: 12 }}>
          {loading ? "预测中..." : "上传并预测"}
        </button>
      </form>

      {error && <p style={{ color: "tomato" }}>错误：{error}</p>}

      {result && (
        <div style={{ marginTop: 24 }}>
          <h3>结果（{result.filename}）</h3>

          <details open>
            <summary>阳性标签</summary>
            {result.positive_findings?.length ? (
              <ul>
                {result.positive_findings.map((k) => (
                  <li key={k}>
                    {k} — prob: {result.probabilities[k]?.toFixed(4)} | pred: {result.predictions[k]}
                  </li>
                ))}
              </ul>
            ) : (
              <p>无阳性（所有阈值 < 0.5）。</p>
            )}
          </details>

          <details>
            <summary>全部概率</summary>
            <ul>
              {Object.entries(result.probabilities).map(([k, v]) => (
                <li key={k}>{k}: {v.toFixed(4)}（pred {result.predictions[k]}）</li>
              ))}
            </ul>
          </details>

          {!!result.heatmap_urls?.length && (
            <>
              <h4>热力图</h4>
              <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
                {result.heatmap_urls.map((u, i) => (
                  <img key={i} src={`${process.env.REACT_APP_API_BASE || "http://localhost:8001"}${u}`} alt="heatmap" style={{ width: "100%" }} />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
