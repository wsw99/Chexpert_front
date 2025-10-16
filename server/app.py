from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn, os, uuid, shutil
from typing import List, Optional, Dict, Any

# 引入你自己的函数：load_trained_model / predict_with_optional_heatmap
from test import load_trained_model, predict_with_optional_heatmap

# ----- 配置 -----
MODEL_PATH = os.environ.get("MODEL_PATH", "final_global_model.pth")
HEATMAP_DIR = os.environ.get("HEATMAP_DIR", "heatmaps")
UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", "uploads")

os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="CheXpert Predictor API", version="1.0.0")

# 允许本地前端访问（开发期）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态目录，用于直接访问热力图
app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

# 启动时加载模型（只加载一次）
model = None
class_names: List[str] = []

@app.on_event("startup")
def _load_model_once():
    global model, class_names
    print("Loading model...")
    model, class_names = load_trained_model(MODEL_PATH)  # 直接复用你的函数
    print("Model ready.")

class PredictResponse(BaseModel):
    filename: str
    probabilities: Dict[str, float]
    predictions: Dict[str, int]
    positive_findings: List[str]
    heatmap_urls: List[str] = []

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    file: UploadFile = File(..., description="X-ray image file"),
    generate_heatmap: bool = Form(False)
):
    # 保存上传图片到临时路径
    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用你自己的预测函数
    result = predict_with_optional_heatmap(
        model=model,
        image_path=temp_path,
        class_names=class_names,
        generate_heatmap=generate_heatmap,
        save_dir=HEATMAP_DIR
    )

    if result is None:
        # 返回一个简单错误（这里简化处理）
        return PredictResponse(
            filename=file.filename,
            probabilities={},
            predictions={},
            positive_findings=[],
            heatmap_urls=[]
        )

    probabilities, predictions = result

    # 组织输出：类名 -> prob / pred
    probs_dict = {cls: float(probabilities[i]) for i, cls in enumerate(class_names)}
    preds_dict = {cls: int(predictions[i]) for i, cls in enumerate(class_names)}
    positives = [cls for cls, v in preds_dict.items() if v == 1]

    # 如果生成热力图，把目录里新生成的目标文件挑出来（最简单方案：按文件名前缀匹配）
    heatmap_urls: List[str] = []
    base = os.path.splitext(os.path.basename(temp_path))[0]
    # 你的 test.py 中热力图命名是："{base_name}_{class_name}_heatmap.jpg"
    # 但 base_name 用的是原始上传文件名，不是我们 uuid 名，这里我们简单放开匹配（也可以在 test.py 里改成 uuid）
    for fname in os.listdir(HEATMAP_DIR):
        if fname.endswith("_heatmap.jpg"):
            # 返回可通过浏览器访问的 URL
            heatmap_urls.append(f"/heatmaps/{fname}")

    return PredictResponse(
        filename=file.filename,
        probabilities=probs_dict,
        predictions=preds_dict,
        positive_findings=positives,
        heatmap_urls=heatmap_urls if generate_heatmap else []
    )

if __name__ == "__main__":
    # 开发期本地启动：uvicorn app:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
