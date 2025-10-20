# -*- coding: utf-8 -*-
"""
test.py — 推理侧重构（仅单文件修改，不新增目录）
目的：
1) 提供可被后端直接调用的两个函数：
   - init_model(model_path, class_names=None, device=None, use_imagenet_norm=False)
   - infer(pil_image or str(path), generate_heatmap=True, threshold=0.5, alpha=0.45, return_top_k=None)
2) 保持原单脚本可运行（__main__），但不强依赖交互；支持命令行快速测试。

注意：
- 不在此文件内实现 Web 服务，仅作为“模型层”被调用。
- 归一化默认关闭（与原先未做归一化保持一致）；如训练期用了 ImageNet Normalize，可将 use_imagenet_norm=True。
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# ---------------------------
# 全局缓存（进程内只加载一次）
# ---------------------------
_MODEL: Optional[nn.Module] = None
_CLASS_NAMES: Optional[List[str]] = None
_DEVICE: Optional[torch.device] = None
_TRANSFORM: Optional[T.Compose] = None
_LAST_FEATS: Optional[torch.Tensor] = None  # 用于 CAM

def _get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 模型构建与加载
# ---------------------------
def _build_densenet121(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Sigmoid(),  # 多标签
    )
    return model


def init_model(
    model_path: str,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_imagenet_norm: bool = False,
) -> Tuple[nn.Module, List[str]]:
    """
    加载模型与类别名，并注册 CAM 钩子。
    - model_path: 训练得到的权重 .pth / .pt
    - class_names: 类别名列表；如果为空，会从权重推断 num_classes 并使用占位名 C0..Cn-1
    - device: 'cuda' / 'cpu'；默认自动
    - use_imagenet_norm: 若训练期用过 ImageNet 归一化，则置 True
    """
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM, _LAST_FEATS

    _DEVICE = _get_device(device)

    # 读取权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重未找到: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    # 兼容保存的是 {'model_state_dict':..., 'class_names':...}
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # 推断类别数
    if "classifier.0.weight" in state_dict:
        num_classes = state_dict["classifier.0.weight"].shape[0]
    elif "classifier.weight" in state_dict:
        num_classes = state_dict["classifier.weight"].shape[0]
    else:
        # 回退：需要用户显式传入 class_names
        if not class_names:
            raise ValueError("无法从权重推断类别数，请传入 class_names")
        num_classes = len(class_names)

    # 类别名
    if class_names is None:
        # 尝试从权重中恢复
        if isinstance(state, dict) and "class_names" in state and isinstance(state["class_names"], (list, tuple)):
            class_names = list(state["class_names"])
        else:
            class_names = [f"C{i}" for i in range(num_classes)]
    else:
        assert len(class_names) == num_classes, "class_names 长度与权重类别数不一致"

    # 构建与加载
    model = _build_densenet121(num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # 打印提示但不中断
        print(f"[test.py] load_state_dict 提示 missing={missing} unexpected={unexpected}")

    model.eval().to(_DEVICE)

    # 注册特征图钩子（DenseNet 的 features 输出）
    def _hook(module, inp, out):
        global _LAST_FEATS  # ← 这里把 nonlocal 改为 global
        _LAST_FEATS = out.detach()  # [B, C, H, W]

    model.features.register_forward_hook(_hook)

    # 预处理
    tfms = [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
    if use_imagenet_norm:
        tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))
    _TRANSFORM = T.Compose(tfms)

    _MODEL = model
    _CLASS_NAMES = class_names

    print(f"[test.py] 模型已加载: classes={len(class_names)} device={_DEVICE}")
    return _MODEL, _CLASS_NAMES


# ---------------------------
# 工具：图片输入
# ---------------------------
def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"图片不存在: {img}")
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        # BGR -> RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    raise TypeError(f"不支持的输入类型: {type(img)}")


def _make_cam(
    feats: torch.Tensor,  # [1,C,H,W]
    classifier: nn.Module,
    probs: np.ndarray,      # [K]
    pos_indices: List[int], # 需要可视化的类别索引
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    简易 CAM：使用分类器第一层线性权重与 features 做加权求和。
    多标签场景，我们将多个 positive 类的 CAM 做归一化后求和。
    """
    # 取线性层
    if isinstance(classifier, nn.Sequential):
        linear = None
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                linear = m
                break
    elif isinstance(classifier, nn.Linear):
        linear = classifier
    else:
        linear = None

    if linear is None:
        return None

    with torch.no_grad():
        w = linear.weight.detach().cpu().numpy()  # [K, C]
        f = feats.detach().cpu().numpy()[0]       # [C, H, W]

    cams = []
    for idx in pos_indices:
        if idx < 0 or idx >= w.shape[0]:
            continue
        cam = (w[idx][:, None, None] * f).sum(axis=0)  # [H, W]
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cams.append(cam)

    if not cams:
        # 如果没有 positive，就取最大概率那个类
        top_idx = int(np.argmax(probs))
        cam = (w[top_idx][:, None, None] * f).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cams = [cam]

    cam_sum = np.clip(np.sum(cams, axis=0), 0, 1)  # 聚合
    cam_resized = cv2.resize(cam_sum, (target_size[0], target_size[1]))  # (W,H)
    heatmap = (cam_resized * 255).astype(np.uint8)
    return heatmap  # 0~255


def infer(
    image: Union[str, Image.Image, np.ndarray],
    generate_heatmap: bool = True,
    threshold: float = 0.5,
    alpha: float = 0.45,
    return_top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    对单张图片做推理。返回结构便于上层服务装配 JSON。
    - image: 路径 / PIL / ndarray
    - threshold: 0~1；用于将概率二值化
    - return_top_k: 若给定，仅返回前 k 个类别到 positive_findings
    返回：
    {
      'probs': {label: float, ...},
      'preds': {label: 0|1, ...},
      'positive_findings': [{'label','confidence'}, ...],
      'heatmap': PIL.Image or None
    }
    """
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM, _LAST_FEATS

    if _MODEL is None or _CLASS_NAMES is None or _TRANSFORM is None:
        raise RuntimeError("模型未初始化。请先调用 init_model(model_path, ...)")

    pil = _to_pil(image)
    orig_w, orig_h = pil.size

    # 预处理
    tensor = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        _LAST_FEATS = None
        outputs = _MODEL(tensor)             # [1, K]，已 Sigmoid
        probs = outputs.squeeze(0).detach().cpu().numpy()  # [K]

    preds = (probs > float(threshold)).astype(np.int32)     # [K]

    # 组织结果
    probs_dict = {lbl: float(p) for lbl, p in zip(_CLASS_NAMES, probs)}
    preds_dict = {lbl: int(v) for lbl, v in zip(_CLASS_NAMES, preds)}

    # 阳性项（按概率排序）
    idx_sorted = np.argsort(-probs)
    findings = []
    for idx in idx_sorted:
        if preds[idx] == 1:
            findings.append({"label": _CLASS_NAMES[idx], "confidence": float(probs[idx])})
    if return_top_k is not None:
        findings = findings[: int(return_top_k)]

    # 生成热力图
    heat_img = None
    if generate_heatmap and _LAST_FEATS is not None:
        pos_indices = [i for i, v in enumerate(preds) if v == 1]
        cam_u8 = _make_cam(_LAST_FEATS, _MODEL.classifier, probs, pos_indices, (orig_w, orig_h))
        if cam_u8 is not None:
            # 叠加
            base = np.array(pil.convert("RGB"))
            color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            overlay = (alpha * color + (1 - alpha) * base).astype(np.uint8)
            heat_img = Image.fromarray(overlay)

    return {
        "probs": probs_dict,
        "preds": preds_dict,
        "positive_findings": findings,
        "heatmap": heat_img,  # 由上层决定是否保存为文件
    }


# ---------------------------
# 命令行快速测试（可选）
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CheXpert DenseNet121 推理测试")
    parser.add_argument("--model", type=str, default="final_global_model.pth", help="模型权重路径")
    parser.add_argument("--image", type=str, required=True, help="待推理图片路径")
    parser.add_argument("--heatmap", action="store_true", help="是否生成热力图并保存为 <image>_cam.png")
    parser.add_argument("--threshold", type=float, default=0.5, help="阈值 (0~1)")
    parser.add_argument("--imagenet-norm", action="store_true", help="使用 ImageNet 归一化")
    args = parser.parse_args()

    model, classes = init_model(args.model, class_names=None, use_imagenet_norm=args.imagenet_norm)
    result = infer(args.image, generate_heatmap=args.heatmap, threshold=args.threshold)

    print("\n=== Positive Findings ===")
    for itm in result["positive_findings"]:
        print(f"- {itm['label']}: {itm['confidence']:.4f}")

    if result["heatmap"] is not None and args.heatmap:
        out_path = os.path.splitext(args.image)[0] + "_cam.png"
        result["heatmap"].save(out_path)
        print(f"已保存热力图: {out_path}")
