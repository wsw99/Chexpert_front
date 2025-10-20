// src/features/history/api.js
import { API_BASE_URL } from '../../config/api';

export async function fetchAnalysisHistory({ page = 1, pageSize = 10 } = {}) {
  const url = `${API_BASE_URL}/api/v1/history?page=${page}&page_size=${pageSize}`;
  const res = await fetch(url);
  const data = await res.json();
  if (!data?.success) return { total: 0, items: [] };

  return {
    total: data.total || 0,
    items: (data.items || []).map(it => ({
      date: it.date,
      fileName: it.file_name,
      diagnosis: it.diagnosis,
      confidence: it.confidence, // 0~1
      status: it.status || 'completed',
    })),
  };
}
