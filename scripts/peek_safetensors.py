import json
from pathlib import Path

import torch
from safetensors.torch import safe_open

MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen2.5-7B")  # 改成你的路径

# 1) 读 index
index_path = MODEL_DIR / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
weight_map = index["weight_map"]

# 2) 选择一个 key（比如 layer0 的 q_proj）
key = "model.layers.0.self_attn.q_proj.weight"
shard = weight_map[key]
print("key in shard:", shard)

# 3) 打开对应分片，只读这个 tensor
with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as f:
    print(f)
    
    # print("num keys in shard:", len(f.keys()))
    t = f.get_tensor(key)
    # print(t)
    print("keys: ",key," shape: ", t.shape, " dtype: ",t.dtype)
    
