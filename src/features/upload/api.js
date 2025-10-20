// upload/api.js
// 说明：对接 /api/analyze，支持可选项 generate_heatmap / threshold / alpha / return_top_k
// 依赖（若存在）：../../config/api 提供 API_ENDPOINTS 与 apiPostFormData
 import { API_ENDPOINTS, apiPostFormData, API_BASE_URL } from '../../config/api';

 // 工具：相对 /static/xxx 转绝对 http://localhost:8000/static/xxx
 const toAbs = (u) => {
   if (!u) return null;
   if (/^https?:\/\//i.test(u)) return u;         // 已是绝对地址
   const base = (typeof API_BASE_URL === 'string' && API_BASE_URL) || '';
   return `${base}${u.startsWith('/') ? '' : '/'}${u}`;
 };

// ========== 校验与工具 ==========
export function validateFileType(file) {
  const allowed = ['.dcm', '.jpg', '.jpeg', '.png'];
  const name = (file?.name || '').toLowerCase();
  return allowed.some(ext => name.endsWith(ext));
}

export function validateFileSize(file, limitMB = 50) {
  const max = limitMB * 1024 * 1024;
  return (file?.size || 0) <= max;
}

export function getFilePreviewUrl(file) {
  try {
    return URL.createObjectURL(file);
  } catch {
    return null;
  }
}

// ========== 核心：上传并分析 ==========
export async function uploadAndAnalyzeImage(file, options = {}) {
  const {
    generate_heatmap = true,
    threshold = 0.5,
    alpha = 0.45,
    return_top_k = null,
    request_id = undefined,
  } = options || {};

  if (!validateFileType(file)) {
    return { success: false, error_code: 'BAD_FILE_TYPE', message: 'Only .dcm/.jpg/.jpeg/.png supported.' };
  }
  if (!validateFileSize(file)) {
    return { success: false, error_code: 'FILE_TOO_LARGE', message: 'File must be <= 50MB.' };
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('generate_heatmap', String(generate_heatmap));
  formData.append('threshold', String(threshold));
  formData.append('alpha', String(alpha));
  if (return_top_k != null) formData.append('return_top_k', String(return_top_k));
  if (request_id) formData.append('request_id', request_id);

  let resp;
  try {
    if (API_ENDPOINTS?.IMAGE_ANALYZE && typeof apiPostFormData === 'function') {
      resp = await apiPostFormData(API_ENDPOINTS.IMAGE_ANALYZE, formData);
    } else {
      const r = await fetch('/api/v1/image/analyze', { method: 'POST', body: formData });
      resp = await r.json();
    }
  } catch (e) {
    return { success: false, error_code: 'NETWORK_ERROR', message: e?.message || 'Network error' };
  }

  if (!resp?.success) return resp;

  const classifications = Array.isArray(resp.classifications) ? resp.classifications : [];
  const top = classifications[0] || {};
  const disease = top.label || 'unknown';
  const confidence = typeof top.confidence === 'number' ? top.confidence : 0;
  const explanation = top.description || '暂无详细说明';

  return {
    success: true,
    disease,
    confidence, // 0~1（前端进度条转百分比）
    explanation,
    recommendations: classifications.map(it => ({
      label: it.label,
      confidence: `${Math.round((it.confidence || 0) * 100)}%`,
      description: it.description || ''
    })),
    heatmapUrl: toAbs(resp.heatmap_image_url),
    originalImageUrl: toAbs(resp.original_image_url),
    meta: resp.meta || {},
    rawResponse: resp
  };
}
