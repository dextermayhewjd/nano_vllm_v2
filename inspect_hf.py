import inspect
from transformers.models.qwen3 import modeling_qwen3

print(inspect.getsource(modeling_qwen3.Qwen3Attention))