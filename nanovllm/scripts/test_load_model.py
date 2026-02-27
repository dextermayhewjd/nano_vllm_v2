from transformers import Qwen3Config
from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_weights

config = Qwen3Config.from_pretrained("/home/fredkeira/Data/models/Qwen/Qwen3-8B")
model = Qwen3ForCausalLM(config)
load_weights(model, "/home/fredkeira/Data/models/Qwen/Qwen3-8B")