import json
from pathlib import Path

import torch
from safetensors.torch import safe_open

import os
from glob import glob
MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen3-4B")  # 改成你的路径

# ==============================================
# 1) 读 index
# index_path = MODEL_DIR / "model.safetensors.index.json"
# index = json.loads(index_path.read_text())
# weight_map = index["weight_map"]

# # 2) 选择一个 key（比如 layer0 的 q_proj）
# key = "model.layers.0.self_attn.q_proj.weight"
# shard = weight_map[key]
# print("key in shard:", shard)

# # 3) 打开对应分片，只读这个 tensor
# with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as f:
#     print(f)
    
#     # print("num keys in shard:", len(f.keys()))
#     t = f.get_tensor(key)
#     # print(t)
#     print("keys: ",key," shape: ", t.shape, " dtype: ",t.dtype)
    
# ==============================================



# 4) 打开所有分片 读取所有weight的name 
for file in glob(os.path.join(MODEL_DIR, "*.safetensors")):
    #这里是在找到path中 所有的file
    with safe_open(file, framework="pt", device="cpu") as f:
    # 这里是在打开每一个file
        for weight_name in f.keys():
            print(weight_name)