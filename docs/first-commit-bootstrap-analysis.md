# nano-vllm 初始搭建分析（从第一个 commit 开始）

## 分析方法（命令）
- `git log --reverse --oneline --decorate --date=short --pretty=format:'%h %ad %an %s'`
- `git show --stat --name-status --format=fuller a5a4909`
- `git show a5a4909:<path>`（逐个查看首个提交关键文件）
- `git show --unified=3 <commit> -- <path...>`（查看早期修复/重构的关键差异）

## 阶段 0：仓库“冷启动”——a5a4909 (init commit)
首个提交一次性加入了完整最小可运行骨架：

1. **入口与对外 API**
   - `nanovllm/__init__.py` 暴露 `LLM` 与 `SamplingParams`。
   - `nanovllm/llm.py` 让 `LLM` 直接继承 `LLMEngine`，说明最开始就是“薄封装 + 核心引擎”的结构。

2. **配置与参数面**
   - `nanovllm/config.py` 用 dataclass 定义核心运行参数：批处理 token 上限、并发序列上限、显存利用率、KV cache block 大小、是否强制 eager 等。
   - `nanovllm/sampling_params.py` 单独抽象采样参数：温度、最大生成长度、是否忽略 eos。

3. **执行主链路（Engine）**
   - `nanovllm/engine/llm_engine.py` 初始化时串起：
     - HF `AutoConfig` / `AutoTokenizer`
     - `ModelRunner`（模型执行）
     - `Scheduler`（调度）
   - `generate()` 主循环是典型的 vLLM 风格：`add_request -> schedule/step -> postprocess -> decode`。

4. **调度与生命周期管理**
   - `nanovllm/engine/sequence.py` 定义请求状态机与 token 缓冲（WAITING/RUNNING/FINISHED）。
   - `nanovllm/engine/scheduler.py` 实现 prefill / decode 两阶段调度。
   - `nanovllm/engine/block_manager.py` 实现分页 KV cache 的 block 分配、释放与哈希复用（prefix cache）。

5. **模型执行与缓存布局**
   - `nanovllm/engine/model_runner.py` 完成：
     - CUDA 设备与 dtype 上下文切换
     - Qwen3 模型实例化
     - KV cache 预分配和挂载到注意力层
     - prefill/decode 输入张量构造
     - （非 eager）CUDA Graph capture

6. **模型与算子层实现**
   - `nanovllm/models/qwen3.py` 内置 Qwen3 CausalLM 结构。
   - `nanovllm/layers/*` 提供 attention、rotary、parallel linear、norm、sampler、embedding/head 等基础组件。

7. **可运行脚本与依赖**
   - `example.py` 提供聊天模板推理例子。
   - `bench.py` 提供吞吐 benchmark。
   - `requirements.txt` 提供最小依赖集（torch/triton/transformers/cmake/ninja）。

> 结论：第一个提交不是“空仓初始化”，而是直接落地了一个端到端可跑通的单模型推理内核（含调度、缓存、模型、示例与 benchmark）。

## 阶段 1：首日修复——b98e1ca (fix)
在初版骨架后，作者马上修了几类“可用性与正确性”问题：

1. **吞吐可观测性增强**
   - `LLMEngine.generate()` 增加 prefill / decode 实时 tok/s 展示。

2. **decode 索引正确性修复**
   - `ModelRunner.prepare_decode()` 的 `slot_mapping` 从 `... + len(last_block())` 改为 `... + len(last_block()) - 1`，避免 off-by-one。

3. **停止条件修复**
   - `Scheduler.postprocess()` 改为尊重 `ignore_eos`（此前会无条件按 eos 停止）。

4. **命名与小问题修正**
   - `capture_model` 重命名为 `capture_cudagraph`。
   - prompt 处理与参数默认值也做了微调（如 batched token 上限加大）。

## 阶段 2：结构重构——386290d (refactor)
这一提交显示作者开始从“先跑起来”转向“更通用和更稳”：

1. **block_size 从硬编码 256 走向参数化**
   - `Config` 约束从“必须等于 256”放宽到“256 的倍数”。
   - `BlockManager/Sequence` 的若干逻辑去掉硬编码，改为跟随配置。

2. **调度策略防御增强**
   - `can_append(seq)` 按序列是否会跨块来判断是否需要新 block，而不是固定判定。
   - 新增 `can_prefill` 逻辑，开始考虑 cache 水位对 prefill 的影响。

3. **输出结构调整**
   - `generate()` 输出由纯文本变为 `{text, token_ids}`，便于上层系统做可观测与后处理。

## 阶段 3：权重加载工程化——08c84ec (multi file loader)
初版 `Qwen3ForCausalLM.load_weights()` 被抽离成通用 `utils/loader.py`：

1. **支持多 safetensors 文件扫描加载**
   - 从“固定读一个 `model.safetensors`”改成遍历目录中的 `*.safetensors`。

2. **packed 权重映射更清晰**
   - 将 `packed_modules_mapping` 从“模块->列表”转成“原始权重名->(目标模块, shard_id)”映射，减少分支判断。

3. **Runner 与模型解耦**
   - `ModelRunner` 直接调用 `load_model(self.model, path)`，后续新增模型时可复用加载器。

## 阶段 4：继续打磨（紧随其后）
- `fee58d4`、`f16adb7` 等提交继续围绕调度/配置/引擎做修复和简化，说明最初 2~3 天主要是在“稳定核心执行链路”。

## 总体判断：这个项目最开始是如何“搭起来”的
1. **先把最短可运行链路一次性铺齐**：API、配置、调度、KV cache、模型定义、加载、示例、基准全在首个提交就到位。
2. **再快速修 correctness 与可观测**：off-by-one、eos 逻辑、吞吐指标第一时间补齐。
3. **接着做工程化抽象**：参数化 block 大小、统一 loader、输出结构化，为后续多模型/多功能演进打基础。
4. **开发节奏符合“高强度原型迭代”**：先全链路可跑，再连发修复与重构，逐步从 Demo 内核走向可维护项目。
