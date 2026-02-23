Phase 0: 环境与基础设施
T-001: 环境搭建 [DONE 2026-02-23, see docs/PROJECT_LOG.md]

创建项目仓库、虚拟环境
安装 transformers, torch, accelerate, sentencepiece/tiktoken
确认 GPU 可用，记录硬件信息（显存大小决定后续用哪个尺寸的模型）

T-002: 选定模型尺寸并下载权重

根据显存选择 Qwen2.5 的具体尺寸（0.5B/1.5B/7B/14B/72B）
用 huggingface-cli 或 snapshot_download 拉取模型到本地
验证 checksum，记录模型文件结构（config.json, tokenizer.json, *.safetensors）


Phase 1: 跑通基本推理
T-003: 最简 pipeline 推理

用 transformers.pipeline("text-generation", model="Qwen/Qwen2.5-XXB") 跑通一句话生成
记录输入/输出，作为后续所有替换的 golden reference
验收标准：能输出合理文本

T-004: 显式调用模型推理（脱离 pipeline）

手动调用 AutoTokenizer + AutoModelForCausalLM
自己写 generate 循环：tokenize → forward → argmax/sample → decode
对比 T-003 的输出，确保一致
这一步的目的：理解 pipeline 背后在干什么

T-005: 实现基础采样策略

实现 greedy decoding（argmax）
实现 temperature sampling
实现 top-k 和 top-p (nucleus) sampling
每种策略都和 HF generate() 的对应参数输出做 diff 验证


Phase 2: 理解模型结构
T-006: 导出并记录 Qwen2.5 完整架构

打印 model.named_modules()，逐层记录
画出完整的计算图（Embedding → N × DecoderLayer → RMSNorm → LM Head）
记录每个 DecoderLayer 内部结构：Self-Attention（GQA）、MLP（SwiGLU）、RMSNorm 的位置
记录关键超参数：hidden_size, num_heads, num_kv_heads, intermediate_size, rope_theta 等
产出：一份架构文档，后续替换时作为 spec

T-007: 权重形状与命名映射表

遍历 model.state_dict()，记录每个 key、shape、dtype
整理成表格，标注每个权重属于哪个模块
这是后续自己搭模型时加载权重的 对照表

T-008: 单层 forward 验证工具

写一个工具函数：给定输入 tensor，跑 HF 模型的某一层，记录输入输出
例如：单独跑第 0 层 DecoderLayer，保存输入/输出 tensor 到文件
目的：后续替换单个模块时，可以逐层做数值对比


Phase 3: 理解关键组件
T-009: Tokenizer 行为分析

测试各种输入（中文、英文、混合、特殊字符、代码）的 tokenize 结果
记录 vocab size、special tokens（BOS/EOS/PAD）、chat template
理解 apply_chat_template() 在做什么
产出：tokenizer spec 文档

T-010: RoPE（旋转位置编码）数值验证

从 HF 模型中提取 RoPE 的 cos/sin cache
自己用公式算一遍，做数值 diff（atol < 1e-5）
记录 rope_theta, max_position_embeddings 等参数
注意 Qwen2.5 可能用了 YaRN 或其他 RoPE 变体

T-011: GQA（Grouped-Query Attention）数值验证

提取 attention 层的 Q/K/V 权重，确认 num_heads vs num_kv_heads 的关系
用 HF 模型跑一次 attention，记录中间的 Q/K/V/Scores/Output
理解 KV heads 如何 broadcast 到 Q heads

T-012: SwiGLU MLP 数值验证

提取 gate_proj, up_proj, down_proj 权重
手算 down_proj(silu(gate_proj(x)) * up_proj(x))
和 HF 输出做 diff

T-013: RMSNorm 数值验证

提取 weight，手算 RMSNorm
和 HF 输出做 diff


Phase 4: KV Cache 与完整生成
T-014: 理解并记录 HF 的 KV Cache 机制

开启 use_cache=True，观察 past_key_values 的结构和形状
记录 prefill 阶段 vs decode 阶段 KV cache 的变化
写文档说明 cache 的索引方式

T-015: 手动管理 KV Cache 的推理循环

不依赖 HF 的 generate()，自己写完整的 autoregressive loop
第一步 prefill：完整输入过模型，拿到 KV cache
后续 decode：每次只输入新 token + 传入 past KV cache
验证输出和 T-004 一致
这是最关键的 ticket，完成后你就完全掌握了推理流程

T-016: 性能基线测量

测量 HF 推理的 tokens/sec（prefill 和 decode 分开测）
测量显存占用
不同 batch size 下的吞吐
产出：性能基线报告，作为后续优化的对照
