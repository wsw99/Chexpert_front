import torch
state = torch.load("final_global_model.pth", map_location="cpu")
sd = state.get("state_dict", state.get("model_state_dict", state))

# 打印所有关键 classifier 权重的形状，看看 out_features 是多少
for k, v in sd.items():
    if k.endswith("weight") and v.ndim == 2:
        print(k, v.shape)   # 例如 (14, 1024) 表示 14 类
