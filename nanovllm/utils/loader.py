from safetensors.torch import safe_open
from pathlib import Path
import os
from glob import glob
import torch.nn as nn

# MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen3-4B")  # 改成你的路径

# # 4) 打开所有分片 读取所有weight的name 
# def load_model(path: str):
#     for file in glob(os.path.join(MODEL_DIR, "*.safetensors")):
#         #这里是在找到path中 所有的file
#         with safe_open(file, framework="pt", device="cpu") as f:
#         # 这里是在打开每一个file
#             for weight_name in f.keys():
#                 print(weight_name)

def load_weights(model: nn.Module, model_dir: str | Path):
      """
      从 model_dir 下的所有 safetensors 分片中加载权重到 model。
      要求 model.named_parameters() 的 key 与 checkpoint 的 key 完全对齐。
      """
      model_dir = Path(model_dir)

      # 1) 建立 name -> param 的映射
      param_map = dict(model.named_parameters())

      loaded = set()

      # 2) 遍历所有 safetensors 分片
      for file in sorted(glob(os.path.join(model_dir, "*.safetensors"))):
          with safe_open(file, framework="pt", device="cpu") as f:
              for weight_name in f.keys():
                  if weight_name not in param_map:
                      print(f"[SKIP] checkpoint has '{weight_name}' but model does not")
                      continue

                  param = param_map[weight_name]
                  loaded_weight = f.get_tensor(weight_name)

                  # 检查 shape 是否匹配
                  if param.shape != loaded_weight.shape:
                      print(f"[ERROR] shape mismatch: {weight_name} "
                            f"model={param.shape} ckpt={loaded_weight.shape}")
                      continue

                  param.data.copy_(loaded_weight)
                  loaded.add(weight_name)

      # 3) 检查是否有遗漏
      missing = set(param_map.keys()) - loaded
      if missing:
          print(f"\n[WARN] {len(missing)} params NOT loaded:")
          for name in sorted(missing):
              print(f"  {name}")

      print(f"\nLoaded {len(loaded)}/{len(param_map)} parameters")