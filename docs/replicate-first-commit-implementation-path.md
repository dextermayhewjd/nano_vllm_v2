# 复刻 nano-vllm 首个 commit（a5a4909）的实现路径

> 目标：以 `a5a4909 init commit` 为“验收基线”，从 0 到 1 复刻同等能力的最小可运行推理内核。

## 0. 基线能力定义（你要复刻到什么程度）
以首个 commit 为标准，最终要具备：

1. 能通过 `LLM(...)` 初始化模型、tokenizer、scheduler、runner。
2. 能接受多请求（str 或 token ids）并进行 batch 推理。
3. 有 prefill / decode 两阶段调度。
4. 有分页 KV Cache（block）分配/释放与 prefix 复用。
5. 能加载 Qwen3 权重并完成逐 token 采样。
6. 能跑 `example.py` 与 `bench.py`。

---

## 1. 推荐复刻顺序（Implementation Path）

### Step A：搭基础脚手架（先把模块边界立好）
先创建包结构和空实现，确保 import 链路不炸。

### Step B：先打通“单请求 eager 前向”
不做调度、不做 block 复用，先验证模型能从输入 token 产出下一个 token。

### Step C：补齐 Sequence + Scheduler 最小版本
引入 WAITING/RUNNING/FINISHED，先支持最朴素的 batch 调度。

### Step D：补齐 BlockManager 与 KV cache
实现 allocate/deallocate/may_append，接入 runner 的 slot mapping。

### Step E：实现完整 generate 循环
`add_request -> schedule -> run -> postprocess -> decode` 闭环。

### Step F：补示例与压测脚本
确保能复现初版“能用 + 能测吞吐”。

### Step G：再做首批 correctness 修补
按早期历史经验，优先修：
- decode slot off-by-one
- ignore_eos 行为
- 进度与吞吐可观测

---

## 2. 项目目录设计（按首个 commit 对齐）

```text
nano-vllm-repro/
├── README.md
├── LICENSE
├── requirements.txt
├── example.py
├── bench.py
└── nanovllm/
    ├── __init__.py
    ├── llm.py
    ├── config.py
    ├── sampling_params.py
    ├── engine/
    │   ├── sequence.py
    │   ├── scheduler.py
    │   ├── block_manager.py
    │   ├── model_runner.py
    │   └── llm_engine.py
    ├── models/
    │   └── qwen3.py
    ├── layers/
    │   ├── activation.py
    │   ├── attention.py
    │   ├── embed_head.py
    │   ├── layernorm.py
    │   ├── linear.py
    │   ├── rotary_embedding.py
    │   └── sampler.py
    └── utils/
        ├── context.py
        ├── memory.py
        └── timer.py
```

---

## 3. Tickets 拆解（可直接放 Jira / GitHub Projects）

> 说明：每个 ticket 都给了“输入/输出与验收标准”，你可以并行派工。

### EPIC-0 初始化与依赖

#### T0-1 初始化仓库与基础文件
- 内容：创建包结构、`requirements.txt`、`README`、`LICENSE`。
- 验收：`python -c "import nanovllm"` 成功。

#### T0-2 统一配置对象
- 内容：实现 `Config` 与 `SamplingParams` dataclass。
- 验收：参数可实例化且默认值与首 commit 对齐。

---

### EPIC-1 核心抽象与调度

#### T1-1 Sequence 生命周期模型
- 内容：实现 `SequenceStatus`、`Sequence`（token 缓冲、长度、append）。
- 验收：可完成“建序列->追加 token->统计 completion tokens”。

#### T1-2 Scheduler v1（prefill/decode 两阶段）
- 内容：实现 waiting/running 队列、schedule、postprocess。
- 验收：多请求下状态流转正确，完成请求会释放运行队列。

#### T1-3 停止条件
- 内容：接入 eos 与 max_tokens，支持 `ignore_eos`。
- 验收：`ignore_eos=True` 时不会因 eos 提前停止。

---

### EPIC-2 KV Cache 与内存管理

#### T2-1 Block 与 BlockManager
- 内容：实现 block 分配/释放/append，维护 free/used 列表。
- 验收：随机序列压测 1k 次 allocate/deallocate 后无泄漏。

#### T2-2 Prefix 缓存复用
- 内容：实现 hash 计算与 `hash_to_block_id` 复用策略。
- 验收：相同前缀请求命中缓存时 `num_cached_tokens` 增加。

#### T2-3 GPU 显存估算与 KV cache 预分配
- 内容：按模型参数计算可分配 block 数并创建 KV tensor。
- 验收：启动时打印/记录可用 blocks，模型可绑定到每层 cache。

---

### EPIC-3 模型与算子

#### T3-1 最小层组件
- 内容：实现 linear、norm、rotary、attention、sampler、embedding/head。
- 验收：张量 shape 与 dtype 流水一致，无 device mismatch。

#### T3-2 Qwen3ForCausalLM 前向
- 内容：实现 decoder stack、lm_head、tie embeddings（如需）。
- 验收：给定 input_ids 可产出 logits。

#### T3-3 权重加载 v1
- 内容：支持 safetensors 读取并映射到参数。
- 验收：权重加载后可完成单步生成。

---

### EPIC-4 执行引擎与接口

#### T4-1 ModelRunner（prefill/decode 输入构造）
- 内容：实现 prepare_prefill/prepare_decode/run。
- 验收：prefill 与 decode 都能返回 token ids。

#### T4-2 LLMEngine 主循环
- 内容：实现 add_request/step/generate/is_finished。
- 验收：`generate()` 支持 list[str] 与 list[list[int]]。

#### T4-3 对外 API
- 内容：`LLM` 继承 `LLMEngine`；`__init__.py` 暴露公共对象。
- 验收：`from nanovllm import LLM, SamplingParams` 可直接使用。

---

### EPIC-5 可运行性与验证

#### T5-1 example.py
- 内容：对接 tokenizer chat template + 生成结果打印。
- 验收：可端到端输出 completion。

#### T5-2 bench.py
- 内容：构造随机输入并统计 throughput。
- 验收：脚本可运行并输出 tok/s。

#### T5-3 回归测试（最小集合）
- 内容：补 3 个关键测试：
  1) `ignore_eos` 行为；
  2) decode slot mapping 边界；
  3) block 分配释放计数守恒。
- 验收：测试全部通过。

---

## 4. 里程碑（建议两周节奏）

- **M1（第 1-2 天）**：T0 + T1-1 + T3-1 骨架可跑。
- **M2（第 3-5 天）**：T3-2 + T3-3 + T4-1 单请求可生成。
- **M3（第 6-8 天）**：T1-2/T1-3 + T2-1/T2-2 接入批调度与缓存。
- **M4（第 9-10 天）**：T4-2/T4-3 + T5-1/T5-2 端到端打通。
- **M5（第 11-14 天）**：T5-3 回归测试 + correctness 修补。

---

## 5. Definition of Done（复刻完成标准）

满足以下条件即视为“复刻第一步成功”：

1. 目录结构与模块职责与首 commit 同构。
2. `example.py` 可完成多请求生成。
3. `bench.py` 可输出吞吐。
4. prefill/decode + block manager + eos/max_tokens 行为正确。
5. 至少 3 个关键回归测试通过。

---

## 6. 你可以直接开工的最小任务清单（今日版）

1. 建目录与空类（2h）。
2. 跑通 Qwen3 单步 logits（4h）。
3. 接入 sampler 产出 next token（2h）。
4. 做最小 scheduler（3h）。
5. 接 block manager allocate/deallocate（3h）。
6. 打通 `generate()`（2h）。
7. 写 `example.py` + `bench.py`（2h）。

> 合计约 18 小时净开发量（不含调参与 GPU 环境问题处理）。
