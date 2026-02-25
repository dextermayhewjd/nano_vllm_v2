from safetensors.torch import safe_open
from pathlib import Path
import os
from glob import glob


MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen3-4B")  # 改成你的路径

# 4) 打开所有分片 读取所有weight的name 
def load_model(path: str):
    for file in glob(os.path.join(MODEL_DIR, "*.safetensors")):
        #这里是在找到path中 所有的file
        with safe_open(file, framework="pt", device="cpu") as f:
        # 这里是在打开每一个file
            for weight_name in f.keys():
                print(weight_name)