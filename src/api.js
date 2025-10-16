const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8001";

export async function predict(file, generateHeatmap = false) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("generate_heatmap", String(generateHeatmap));

  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: fd,
  });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}
