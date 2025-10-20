# Chexpert_front-main/app.py
import os, io, uuid, datetime, time, json
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

# =========================
# 路径与静态目录（自动创建）
# =========================
ROOT = os.path.abspath(os.path.dirname(__file__))
STATIC_ROOT = os.path.join(ROOT, "static")
UPLOAD_DIR = os.path.join(STATIC_ROOT, "uploads")
HEATMAP_DIR = os.path.join(STATIC_ROOT, "heatmaps")
for d in (UPLOAD_DIR, HEATMAP_DIR):
    os.makedirs(d, exist_ok=True)

# 历史记录数据文件
DATA_DIR = os.path.join(ROOT, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# App & CORS & 静态资源
# =========================
app = FastAPI(title="CheXpert Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 上线请改成你的前端域名
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

# =========================
# 导入模型（建议文件名为 chex_model.py）
# =========================
from chex_model import init_model, infer as model_infer  # noqa

MODEL_PATH = "final_global_model.pth"  # ← 替换为你的权重文件名/路径

# CheXpert-14 类名
CHEX_CLASSES: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# 仅一次加载模型
_MODEL = None
_CLASS_NAMES: Optional[List[str]] = None
try:
    _MODEL, _CLASS_NAMES = init_model(
        MODEL_PATH,
        class_names=CHEX_CLASSES,
        use_imagenet_norm=False
    )
    print("[app] 模型加载 OK，classes:", len(_CLASS_NAMES))
except Exception as e:
    print("[app] 模型加载失败：", e)

# =========================
# 工具函数
# =========================
def _to_pil_from_upload(f: UploadFile) -> Image.Image:
    name = (f.filename or "").lower()
    data = f.file.read()
    if name.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        else:
            return Image.fromarray(arr[..., :3])
    return Image.open(io.BytesIO(data)).convert("RGB")

def _append_history(item: dict) -> None:
    # item: {"date","file_name","diagnosis","confidence","status"}
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def _read_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    out: List[dict] = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    # 最新的排前面
    return list(reversed(out))

# =========================
# 健康检查
# =========================
@app.get("/healthz")
def healthz():
    ok = _MODEL is not None and _CLASS_NAMES is not None
    return JSONResponse({
        "status": "ok" if ok else "not_ready",
        "model_loaded": ok,
        "model_name": "DenseNet121",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }, status_code=200 if ok else 503)

# =========================
# 历史记录查询
# =========================
@app.get("/api/v1/history")
def get_history(page: int = 1, page_size: int = 10):
    data = _read_history()
    total = len(data)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return {
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": data[start:end],
    }

# =========================
# 推理接口（与前端契约一致）
# =========================
@app.post("/api/v1/image/analyze")
def analyze(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(True),
    threshold: float = Form(0.5),
    alpha: float = Form(0.45),
    return_top_k: Optional[int] = Form(None),
):
    if _MODEL is None:
        raise HTTPException(status_code=424, detail={"error_code": "MODEL_NOT_READY", "message": "Model not loaded"})

    # 1) 读入图片
    try:
        pil = _to_pil_from_upload(file)
    except Exception:
        raise HTTPException(status_code=400, detail={"error_code": "BAD_FILE_TYPE", "message": "Only jpg/png/dcm supported."})

    # 2) 推理
    try:
        t0 = time.time()
        out = model_infer(
            pil,
            generate_heatmap=generate_heatmap,
            threshold=float(threshold),
            alpha=float(alpha),
            return_top_k=return_top_k
        )
        infer_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "INFERENCE_ERROR", "message": str(e)})

    # 3) 保存原图与热力图（当天目录）
    day = datetime.date.today().isoformat()
    up_dir = os.path.join(UPLOAD_DIR, day)
    hm_dir = os.path.join(HEATMAP_DIR, day)
    os.makedirs(up_dir, exist_ok=True)
    os.makedirs(hm_dir, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(up_dir, f"{uid}.png")
    pil.save(orig_path)

    heatmap_url = None
    if generate_heatmap and out.get("heatmap") is not None:
        hm_path = os.path.join(hm_dir, f"{uid}.png")
        out["heatmap"].save(hm_path)
        heatmap_url = f"/static/heatmaps/{day}/{uid}.png"

    original_url = f"/static/uploads/{day}/{uid}.png"

    # 4) 组装返回（与前端契约一致）
    classifications = [
        {"label": it["label"], "confidence": it["confidence"], "description": ""}
        for it in out.get("positive_findings", [])
    ]
    # 若无阳性，也给出 Top-1 作为参考
    best_label, best_prob = None, None
    if out.get("probs"):
        best_label, best_prob = max(out["probs"].items(), key=lambda kv: kv[1])
    if not classifications and best_label is not None:
        classifications = [{"label": best_label, "confidence": float(best_prob), "description": ""}]

    # 5) 记录历史（文件名、诊断、置信度）
    diag_label = classifications[0]["label"] if classifications else (best_label or "-")
    diag_conf = classifications[0]["confidence"] if classifications else float(best_prob or 0.0)
    _append_history({
        "date": day,
        "file_name": file.filename or f"{uid}.png",
        "diagnosis": diag_label,
        "confidence": float(diag_conf),
        "status": "completed"
    })

    return {
        "success": True,
        "classifications": classifications,
        "heatmap_image_url": heatmap_url,     # 相对路径；前端可用 API_BASE_URL 拼绝对
        "original_image_url": original_url,   # 如果前端不展示原图可忽略
        "meta": {
            "model_name": "DenseNet121",
            "inference_time_ms": infer_ms,
            "threshold": float(threshold),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu"
        }
    }
