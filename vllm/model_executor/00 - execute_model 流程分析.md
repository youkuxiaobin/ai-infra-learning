# GPU Model Runner: execute\_model 流程分析

> 源文件：`vllm/v1/worker/gpu_model_runner.py`

***

## 整体概述

`execute_model()` 是 vLLM v1 引擎中 GPU 推理的核心入口，负责将调度器输出转化为模型前向计算结果。在**异步调度模式**下，它与 `sample_tokens()` 配合使用，两者通过 `ExecuteModelState` 传递中间状态：

    execute_model(scheduler_output)
        └── returns None (async) or IntermediateTensors (PP non-final)

    sample_tokens(grammar_output)
        └── returns ModelRunnerOutput or AsyncModelRunnerOutput

在**同步调度模式**下，`execute_model()` 在内部直接完成所有采样并返回 `ModelRunnerOutput`。

***

## 阶段详解

### 阶段 1：状态更新（`_update_states`）

**位置：** `execute_model` → `_update_states()` (L756)

**目的：** 将 `SchedulerOutput` 中的调度信息同步到 Worker 内部的持久状态。

**具体操作：**

1.  **清理已完成请求**
    *   从 `self.requests: dict[str, CachedRequestState]` 中删除 `finished_req_ids`
    *   从 `input_batch` 中调用 `remove_request()` 移除对应槽位

2.  **释放编码器缓存**
    *   遍历 `free_encoder_mm_hashes`，从 `self.encoder_cache` 中删除对应的多模态特征

3.  **移除未被调度的请求**
    *   对比 `input_batch` 中已有的请求和本轮 `scheduled_req_ids`，找出差集（被抢占或暂停的请求）
    *   从 `input_batch` 中移除，但保留 `self.requests` 中的 `CachedRequestState`（未来可能重新调度）

4.  **添加新请求**
    *   遍历 `scheduled_new_reqs`，构建 `CachedRequestState`，包括：
        *   `sampling_params`、`lora_request`、`block_ids`、`num_computed_tokens`
        *   若使用随机种子，初始化 `torch.Generator`
        *   若使用 M-RoPE / XD-RoPE，初始化对应的位置编码

5.  **更新已有（running/resumed）请求的状态**
    *   更新 `num_computed_tokens`（如果有 spec decode，会修正被拒绝的 token 数）
    *   追加新生成的 token 到 `output_token_ids`
    *   更新 KV cache `block_ids`
    *   将 spec token 写入 `input_batch.token_ids_cpu`

6.  **整理 Batch**
    *   `input_batch.condense()`：压缩因移除请求留下的空洞
    *   `_may_reorder_batch()`：允许 Attention Backend 重排请求
    *   `input_batch.refresh_metadata()`：刷新 SamplingMetadata 等元信息

***

### 阶段 2：准备输入（`_prepare_inputs`）

**位置：** `execute_model` → `_prepare_inputs()` (L1294)

**目的：** 在 CPU 上构造推理所需的各类索引和张量，并异步拷贝到 GPU。

**具体操作：**

1.  **Block Table 异步提交**

    *   `block_table.commit_block_table(num_reqs)`：立即启动 GPU 拷贝，与后续 CPU 操作重叠

2.  **计算请求索引和位置**

    ```python
    # req_indices: [0,0, 1,1,1,1,1, 2,2,2] (按调度 token 数展开)
    req_indices = np.repeat(arange[:num_reqs], num_scheduled_tokens)
    # positions: num_computed_tokens[req] + arange[req]
    positions_np = num_computed_tokens_cpu[req_indices] + arange
    ```

3.  **提取 Input Token IDs**

    ```python
    # 将 2D [num_reqs, max_model_len] 展平后用 index_select 高效提取
    token_indices = positions_np + req_indices * max_model_len
    torch.index_select(token_ids_cpu_tensor.flatten(), 0, token_indices_tensor,
                        out=input_ids.cpu[:total_tokens])
    ```

4.  **计算 Slot Mapping**

    *   `block_table.compute_slot_mapping(req_indices, positions_np)` → 计算每个 token 写入哪个 KV cache slot

5.  **构造 Attention 辅助张量**

    *   `query_start_loc`：每个请求的 query 起始位置（cumsum）
    *   `seq_lens`：每个请求的当前序列总长度

6.  **处理 Speculative Decode 元数据**

    *   若无 spec decode：`logits_indices = query_start_loc[1:] - 1`（每个请求最后一个 token）
    *   若有 spec decode：调用 `_calc_spec_decode_metadata()`，生成 `SpecDecodeMetadata`，包含 `target_logits_indices`、`bonus_logits_indices`、`logits_indices`

7.  **位置编码 GPU 拷贝**

    *   标准 1D：`positions.copy_to_gpu()`
    *   M-RoPE（Qwen2-VL）：拷贝 `mrope_positions`
    *   XD-RoPE（HunYuan-VL）：拷贝 `xdrope_positions`

8.  **Hot-Swap LoRA**

    *   若启用 LoRA，调用 `set_active_loras()` 更新当前批次使用的适配器

***

### 阶段 3：确定执行模式与 Padding（`_determine_batch_execution_and_padding`）

**位置：** `execute_model` → `_determine_batch_execution_and_padding()` (L2877)

**目的：** 决定本次前向是否使用 CUDA Graph，以及如何 Padding。

**判断逻辑：**

| 条件                                   | CUDA Graph 模式 |
| ------------------------------------ | ------------- |
| decode-only，token 数符合预捕获尺寸           | `FULL`（完整图）   |
| prefill/mixed，部分层可图化                 | `PARTIAL`     |
| 动态场景（cascade attn, encoder output 等） | `NONE`（eager） |

*   **Uniform Decode 检测：** 当 `max_num_scheduled_tokens == 1` 且所有请求均为 decode（1 token/req），使用均匀 decode 路径
*   **Data Parallel 协调：** 通过 `coordinate_batch_across_dp()` 跨 DP 实例同步 token 数，确保所有实例使用相同的图尺寸
*   **Ubatch 切片：** 若启用 microbatching（DBO），创建 `ubatch_slices` 将 batch 分割为子批次

***

### 阶段 4：构建 Attention Metadata（`_build_attention_metadata`）

**位置：** `execute_model` → `_build_attention_metadata()` (L1517)

**目的：** 为每个 KV cache group 和 Attention Backend 生成推理所需的 `AttentionMetadata`。

**关键输出 `CommonAttentionMetadata`：**

```python
CommonAttentionMetadata(
    query_start_loc=...,       # [num_reqs+1] GPU tensor，cumsum of query lengths
    seq_lens=...,              # [num_reqs] 当前序列总长度
    block_table_tensor=...,    # [num_reqs, max_blocks] KV cache 块表
    slot_mapping=...,          # [num_tokens] 每个 token 对应的 KV cache slot
    max_query_len=...,         # 本批次最大 query 长度
    max_seq_len=...,           # 本批次最大序列长度
    causal=True,
)
```

**Spec Decode 特殊处理：**

*   为 GDN Backend 追加 `num_accepted_tokens`、`num_decode_draft_tokens`

**Cascade Attention（共享前缀优化）：**

*   若启用，`cascade_attn_prefix_lens` 指定每个 KV cache group 的共享前缀长度

***

### 阶段 5：输入预处理（`_preprocess`）

**位置：** `execute_model` → `_preprocess()` (L2538)

**目的：** 准备模型前向所需的 `input_ids`/`inputs_embeds` 和 `positions`。

**三种路径：**

| 模型类型             | 处理方式                                                    |
| ---------------- | ------------------------------------------------------- |
| 多模态模型（VLM）       | 运行 MM Encoder，将视觉特征嵌入融合到 `inputs_embeds`，模型以 embeds 为输入 |
| 启用 prompt embeds | 混合路径：text token → embedding layer；prompt embeds 直接使用    |
| 纯文本模型            | 直接使用 `input_ids`，embedding 层在 CUDA Graph 内部执行（性能最优）     |

**Pipeline Parallel 处理：**

*   First rank：清除 `intermediate_tensors`，从头开始计算
*   非 First rank：接收上一阶段的 `intermediate_tensors` 并切片对齐

***

### 阶段 6：模型前向（`_model_forward`）

**位置：** `execute_model` → `_model_forward()` (L2824)

**目的：** 实际执行模型的 Transformer 前向计算。

**关键上下文管理器 `set_forward_context`：**

*   将 `attn_metadata`、`batch_descriptor`、`cudagraph_runtime_mode` 注入全局上下文
*   Attention Backend 在 forward 中通过 `get_forward_context()` 获取元数据

**KV Connector：**

*   `maybe_get_kv_connector_output()`：在前向执行期间，KV connector（如 Nixl/Mooncake）可并行传输 KV cache

**CUDA Graph 执行：**

*   `FULL` 模式：重放预先捕获的完整图（含 Attention + FFN），仅更新输入 buffer
*   `PARTIAL` 模式：图化 Attention，FFN 动态执行
*   `NONE` 模式：纯 eager 执行

***

### 阶段 7：后处理 Logits

**位置：** `execute_model`（L3229-3299）

**目的：** 从 hidden states 中提取目标位置的 logits。

**操作流程：**

```python
# 提取采样位置的 hidden states
sample_hidden_states = hidden_states[logits_indices]
# 通过 LM Head 计算 logits
logits = model.compute_logits(sample_hidden_states)
```

**Pipeline Parallel 处理：**

*   **非 Last Rank：** 直接返回 `IntermediateTensors`（不计算 logits）
*   **Last Rank（broadcast\_pp\_output=False，常规）：** 直接计算 logits
*   **Last Rank（broadcast\_pp\_output=True，特殊）：** 通过 `send_tensor_dict`/`broadcast_tensor_dict` 将 logits 广播到所有 PP 节点

**Pooling 模型：**

*   跳过 logits，直接调用 `_pool()` 返回 `ModelRunnerOutput`

***

### 阶段 8：采样（`_sample`，在 `sample_tokens` 中执行）

**位置：** `sample_tokens()` → `_sample()` (L2654)

**两种采样路径：**

| 模式            | 采样器                      | 说明                           |
| ------------- | ------------------------ | ---------------------------- |
| 无 spec decode | `self.sampler`           | 标准采样（greedy/top-p/top-k 等）   |
| 有 spec decode | `self.rejection_sampler` | 拒绝采样，验证 draft tokens 并确定接受数量 |

**Grammar 结构化输出：**

*   在采样前调用 `apply_grammar_bitmask()`，将不符合 schema 的 token 概率置为 `-inf`

***

### 阶段 9：草稿 Token 生成（`propose_draft_token_ids`，Speculative Decode 专用）

**位置：** `sample_tokens()` → `propose_draft_token_ids()` (L3348)

**目的：** 为下一步调度生成投机解码的草稿 token。

**执行时机：**

*   **EAGLE（padded batch）：** 在 bookkeeping 前执行，直接使用 GPU 上的 `sampled_token_ids`
*   **其他方法（ngram 等）：** 在 bookkeeping 后执行，使用 CPU 上的 `valid_sampled_token_ids`

**输入：** `hidden_states`、`sample_hidden_states`（以及 `aux_hidden_states` for EAGLE3）

***

### 阶段 10：记账（`_bookkeeping_sync`）

**位置：** `sample_tokens()` → `_bookkeeping_sync()` (L2679)

**目的：** 将采样结果同步回 CPU 并更新 Worker 内部状态。

**操作流程：**

1.  **过滤无效请求：** chunked prefill 的中间步骤不应产生输出 token，通过 `discard_request_mask` 清除

2.  **提取有效 token ID：**
    *   无 spec decode：`sampled_token_ids` 直接转 list
    *   有 spec decode：`RejectionSampler.parse_output()` 解析接受/拒绝结果

3.  **更新缓存状态：**
    *   将采样到的 token 写入 `token_ids_cpu[req_idx, ...]`
    *   追加到 `req_state.output_token_ids`

4.  **异步调度模式：**
    *   不做 CPU 同步，将 `prev_sampled_token_ids`（GPU tensor）缓存
    *   下一步的 `_prepare_inputs` 中用 GPU tensor 直接填充 input\_ids

5.  **计算 Logprobs：**
    *   `logprobs_tensors.tolists()` 转换为 CPU list（同步调度）
    *   `_get_prompt_logprobs_dict()`：若请求了 prompt logprobs，从 `hidden_states` 计算

***

### 阶段 11：EPLB 步骤与构建输出

**位置：** `sample_tokens()`（L3433-3474）

*   **`eplb_step()`：** Expert Parallel Load Balancing，周期性重新分配 MoE 专家到 GPU

*   **构建 `ModelRunnerOutput`：**

    ```python
    ModelRunnerOutput(
        req_ids=...,              # 本批次所有请求 ID
        req_id_to_index=...,      # req_id -> batch index 映射
        sampled_token_ids=...,    # list[list[int]]，每个请求生成的 token
        logprobs=...,             # 采样 logprobs
        prompt_logprobs_dict=..., # prompt logprobs（若请求）
        pooler_output=...,        # embedding 模型输出
        kv_connector_output=...,  # KV transfer 结果
        ec_connector_output=...,  # EC transfer 结果
    )
    ```

*   **异步调度模式：** 额外包装为 `AsyncGPUModelRunnerOutput`，延迟 CPU 拷贝，通过 CUDA event 通知就绪

***

## 核心数据结构

### 输入：`SchedulerOutput`

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]          # 本步新加入的请求
    scheduled_cached_reqs: CachedRequestData          # 已缓存（running/resumed）的请求
    num_scheduled_tokens: dict[str, int]              # req_id -> 本步调度 token 数
    total_num_scheduled_tokens: int                   # 总调度 token 数
    scheduled_spec_decode_tokens: dict[str, list[int]] # req_id -> draft token ids
    scheduled_encoder_inputs: dict[str, list[int]]   # 多模态编码器输入索引
    num_common_prefix_blocks: list[int]               # cascade attention 公共前缀块数
    finished_req_ids: set[str]                        # 已完成的请求 ID
    free_encoder_mm_hashes: list[str]                 # 待释放的编码器缓存
    kv_connector_metadata: KVConnectorMetadata | None # KV transfer 元数据
```

***

### 请求状态缓存：`CachedRequestState`

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    generator: torch.Generator | None          # 用于随机采样的随机数生成器

    block_ids: tuple[list[int], ...]           # KV cache 块 ID（每个 KV cache group 一个 list）
    num_computed_tokens: int                   # 已完成前向计算的 token 数
    output_token_ids: list[int]                # 累计生成的 token

    mrope_positions: torch.Tensor | None       # Qwen2-VL 多维位置编码
    lora_request: LoRARequest | None           # LoRA 适配器
    prev_num_draft_len: int                    # 上一步的 draft token 数（异步 spec decode）
```

***

### 持久批次：`InputBatch`

```python
class InputBatch:
    # CPU 侧持久 buffer（跨步骤复用，避免重复分配）
    token_ids_cpu: np.ndarray              # [max_num_reqs, max_model_len] 所有 token ID
    num_computed_tokens_cpu: np.ndarray    # [max_num_reqs] 每个请求已计算 token 数
    block_table: BlockTable                # KV cache 块表（含 GPU tensor）
    spec_token_ids: list[list[int]]        # spec decode draft token

    # 元数据
    req_ids: list[str]                     # 当前批次请求 ID（有序）
    req_id_to_index: dict[str, int]        # req_id -> 批次槽位索引
    sampling_metadata: SamplingMetadata   # 采样参数（GPU tensor）

    # 异步调度支持
    prev_sampled_token_ids: torch.Tensor | None  # 上一步采样结果（GPU）
    prev_req_id_to_index: dict[str, int] | None  # 上一步的索引映射
```

***

### 中间状态传递：`ExecuteModelState`

```python
class ExecuteModelState(NamedTuple):
    """execute_model() 和 sample_tokens() 之间传递的临时状态"""
    scheduler_output: SchedulerOutput
    logits: torch.Tensor                          # [num_sampled_tokens, vocab_size]
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor                   # [num_tokens, hidden_size]
    sample_hidden_states: torch.Tensor            # hidden_states[logits_indices]
    aux_hidden_states: list[torch.Tensor] | None  # EAGLE3 辅助 hidden states
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
```

***

### Attention 元数据：`CommonAttentionMetadata`

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor     # [num_reqs+1] query 起始位置（GPU，累积和）
    seq_lens: torch.Tensor            # [num_reqs] 序列总长度（GPU）
    block_table_tensor: torch.Tensor  # [num_reqs, max_blocks] KV 块表（GPU）
    slot_mapping: torch.Tensor        # [num_tokens] 每个 token 的 KV slot（GPU）
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    causal: bool
    # 可选：分布式上下文并行
    dcp_local_seq_lens: torch.Tensor | None
```

***

### 投机解码元数据：`SpecDecodeMetadata`

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor        # [num_draft_tokens] 所有 draft token ID（展平）
    num_draft_tokens: list[int]          # [batch_size] 每个请求的 draft token 数
    cu_num_draft_tokens: torch.Tensor    # [batch_size] draft token 数累积和
    cu_num_sampled_tokens: torch.Tensor  # [batch_size] 采样 token 数累积和
    target_logits_indices: torch.Tensor  # draft token 对应的 logits 行索引
    bonus_logits_indices: torch.Tensor   # 每个请求 bonus token 的 logits 行索引
    logits_indices: torch.Tensor         # 合并后的 logits 索引（target + bonus）
```

***

### 最终输出：`ModelRunnerOutput`

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]                        # 本批次请求 ID
    req_id_to_index: dict[str, int]           # req_id -> 索引
    sampled_token_ids: list[list[int]]        # 每个请求本步生成的 token（含 spec decode 接受的）
    logprobs: LogprobsLists | None            # 采样 logprobs
    prompt_logprobs_dict: dict[str, ...]      # prompt logprobs（按需）
    pooler_output: list[torch.Tensor | None]  # embedding 模型输出
    kv_connector_output: KVConnectorOutput | None
    ec_connector_output: ECConnectorOutput | None
    num_nans_in_logits: dict[str, int] | None
    cudagraph_stats: CUDAGraphStat | None
```

***

## 完整流程图

    SchedulerOutput
          │
          ▼
    ┌─────────────────────────────────────────────────────┐
    │                   execute_model()                    │
    │                                                     │
    │  [阶段1] _update_states()                           │
    │    ├── 删除已完成请求                                │
    │    ├── 添加新请求 → CachedRequestState              │
    │    ├── 更新 running 请求状态                         │
    │    └── 整理 InputBatch（condense/reorder/refresh）  │
    │                                                     │
    │  [阶段2] _prepare_inputs()                          │
    │    ├── 计算 req_indices / positions                  │
    │    ├── 提取 input_ids（index_select）               │
    │    ├── 计算 slot_mapping                            │
    │    └── 生成 logits_indices / SpecDecodeMetadata     │
    │                                                     │
    │  [阶段3] _determine_batch_execution_and_padding()   │
    │    └── 决定 CUDAGraphMode / BatchDescriptor         │
    │                                                     │
    │  [阶段4] _build_attention_metadata()                │
    │    └── 生成 CommonAttentionMetadata + 各层 AttnMeta │
    │                                                     │
    │  [阶段5] _preprocess()                              │
    │    ├── 运行多模态编码器（VLM）                       │
    │    └── 准备 input_ids / inputs_embeds / positions   │
    │                                                     │
    │  [阶段6] _model_forward()                           │
    │    └── model(input_ids, positions, attn_metadata)   │
    │         → hidden_states [num_tokens, hidden_dim]    │
    │                                                     │
    │  [阶段7] 后处理 Logits                              │
    │    ├── sample_hidden_states = hidden_states[logits_indices]
    │    └── logits = model.compute_logits(sample_hidden) │
    │                                                     │
    │  ──── 存入 ExecuteModelState，返回 None ────        │
    └─────────────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────────────────────┐
    │                  sample_tokens()                     │
    │                                                     │
    │  [阶段8] apply_grammar_bitmask（结构化输出）         │
    │                                                     │
    │  [阶段9] _sample()                                  │
    │    ├── sampler(logits) → SamplerOutput              │
    │    └── 或 rejection_sampler（spec decode）          │
    │                                                     │
    │  [阶段10] propose_draft_token_ids（spec decode）     │
    │    └── 用 hidden_states 生成下一步 draft tokens     │
    │                                                     │
    │  [阶段11] _bookkeeping_sync()                       │
    │    ├── 更新 token_ids_cpu / output_token_ids        │
    │    └── 计算 logprobs / prompt_logprobs              │
    │                                                     │
    │  [阶段12] eplb_step() + 构建 ModelRunnerOutput      │
    └─────────────────────────────────────────────────────┘
          │
          ▼
    ModelRunnerOutput → EngineCore → Scheduler

***

## 关键设计要点

| 设计点                      | 说明                                                                           |
| ------------------------ | ---------------------------------------------------------------------------- |
| **持久 Batch（InputBatch）** | 跨推理步骤复用 CPU buffer，仅做增量更新，避免全量拷贝                                             |
| **CPU-GPU 重叠**           | block\_table 提交、tensor 拷贝均为异步，与 CPU 计算并行                                     |
| **CUDA Graph**           | decode 阶段固定 batch size，重放预捕获图，消除 kernel 启动开销                                 |
| **异步调度**                 | `execute_model` 返回 None，GPU 继续执行；CPU 在 `sample_tokens` 后才做同步                 |
| **Spec Decode 集成**       | `logits_indices` 统一管理 target/bonus logits 索引，`rejection_sampler` 在采样时验证      |
| **PP（流水线并行）**            | 中间节点返回 `IntermediateTensors`；最后节点计算 logits；`broadcast_pp_output` 支持非最后节点参与采样 |
| **混合 KV Cache**          | 通过 `kv_cache_groups` 支持不同层使用不同 cache 规格（如 Mamba 和 Attention 混合模型）            |

