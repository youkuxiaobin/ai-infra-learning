# `_model_forward` 及其调用上下文详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py`\
> `_model_forward` 定义：L2824–L2854\
> 调用上下文：L3198–L3299（`execute_model` 内）

***

## 函数签名与返回值

```python
def _model_forward(
    self,
    input_ids: torch.Tensor | None = None,      # token id 张量（纯文本路径）
    positions: torch.Tensor | None = None,       # 位置编码张量
    intermediate_tensors: IntermediateTensors | None = None,  # PP 中间层激活
    inputs_embeds: torch.Tensor | None = None,   # embedding 张量（多模态路径）
    **model_kwargs: dict[str, Any],              # 模型额外参数
) -> Any:                                        # hidden_states 或 IntermediateTensors
```

**注意：** `_model_forward` 本身是一个可被子类覆盖的薄封装，核心逻辑在其调用上下文中——包括 `set_forward_context`（注入 attention metadata）和 `maybe_get_kv_connector_output`（KV transfer 并行）。本文档将两者合并讲解。

***

## 示例场景

与 `_prepare_inputs` 文档使用**完全相同的 batch**（见 `prepare_inputs_flow.md`），`max_model_len=2048`：

| req\_idx | req\_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明                                          |
| -------- | ------- | ----------- | ------------ | --------------------------------------------- |
| 0        | req-A   | 0           | 4            | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1        | req-B   | 100         | 1            | decode（无 spec），送入上一步采样的 1 个 token             |
| 2        | req-C   | 50          | 3            | decode（有 spec），1 个真实 token + 2 个 draft token  |

来自上游函数的输入：

```python
# 来自 _preprocess（L2538）：
input_ids            = tensor([p0,p1,p2,p3, t100, t50,t51,t52])  # shape [8]
inputs_embeds        = None
positions            = tensor([0, 1, 2, 3,  100,  50, 51, 52])   # shape [8]
intermediate_tensors = None       # 单节点 PP，无中间层
model_kwargs         = {}

# 来自 _determine_batch_execution_and_padding（L2877）：
cudagraph_mode       = CUDAGraphMode.PIECEWISE
num_tokens_padded    = 8
num_tokens_across_dp = None       # 单卡，无 DP

# 来自 _build_attention_metadata（L1517）：
attn_metadata        = {"layer.0.self_attn": <FlashAttentionMetadata>, ...}

# 来自 maybe_create_ubatch_slices（L61）：
ubatch_slices_padded = None       # 单卡，无 DBO

# 来自 _prepare_inputs（L1294）：
logits_indices       = tensor([3, 4, 5, 6, 7])   # spec decode 场景
```

***

## 各步骤详解

### Step 1 — KV Scales 检查（按需降级为 Eager）

**目的：** FP8 量化模型在初次推理时需要动态计算 KV cache 的量化 scale，该操作涉及动态形状，与 CUDA Graph 不兼容，因此强制降级为 eager 模式，并在完成后关闭该标志。

**代码（L3198–3204）：**

```python
# Set cudagraph mode to none if calc_kv_scales is true.
# KV scales calculation involves dynamic operations that are incompatible
# with CUDA graph capture.
if self.calculate_kv_scales:
    cudagraph_mode = CUDAGraphMode.NONE
    # Mark KV scales as calculated after the first forward pass
    self.calculate_kv_scales = False
```

**示例数据变化：**

    calculate_kv_scales = False   # 标准 BF16 模型，不需要计算 KV scale

    → 跳过，cudagraph_mode 保持 CUDAGraphMode.PIECEWISE

> **FP8 模型初次推理时的对比：**
>
>     calculate_kv_scales = True
>     cudagraph_mode = CUDAGraphMode.NONE   # 强制 eager
>     calculate_kv_scales = False           # 下一步起不再触发

***

### Step 2 — 设置 Forward Context（注入全局推理状态）

**目的：** 将 `attn_metadata`、`cudagraph_mode`、`batch_descriptor`、`num_tokens_across_dp` 等推理状态注入全局 `forward_context`，使模型内部各层的 attention kernel 可以通过 `get_forward_context()` 获取这些信息，而无需通过函数参数层层传递。

**代码（L3208–3217）：**

```python
with (
    set_forward_context(
        attn_metadata,                          # 每层的 AttentionMetadata（layer_name → metadata）
        self.vllm_config,
        num_tokens=num_tokens_padded,           # padding 后 token 数，供 CUDA Graph wrapper 使用
        num_tokens_across_dp=num_tokens_across_dp,  # DP 各 rank 的 token 数
        cudagraph_runtime_mode=cudagraph_mode,  # NONE / PIECEWISE / FULL
        batch_descriptor=batch_desc,            # 描述 batch shape 的 descriptor
        ubatch_slices=ubatch_slices_padded,     # DBO microbatch 切片（None=不切）
    ),
    ...
):
```

`set_forward_context` 内部（`vllm/forward_context.py` L266–L324）：

```python
# 若是多卡 DP 且未传入 num_tokens_across_dp，则发起 all-gather 获取各 rank token 数
if vllm_config.parallel_config.data_parallel_size > 1 and num_tokens_across_dp is None:
    _, num_tokens_across_dp, _ = coordinate_batch_across_dp(...)

dp_metadata = DPMetadata.make(parallel_config, num_tokens, num_tokens_across_dp)

# 创建 forward_context 对象，写入全局状态
forward_context = create_forward_context(
    attn_metadata, vllm_config, virtual_engine=0,
    dp_metadata, cudagraph_runtime_mode, batch_descriptor, ubatch_slices,
)

with override_forward_context(forward_context):
    yield   # 在 with 块内所有模型层可通过 get_forward_context() 读取上述信息
```

**示例数据变化：**

    # 注入的上下文内容：
    attn_metadata = {
        "model.layers.0.self_attn":  <FlashAttentionMetadata>,
        ...
        "model.layers.31.self_attn": <FlashAttentionMetadata>,
    }
    num_tokens_padded    = 8
    num_tokens_across_dp = None      # 单卡，set_forward_context 内不会再发起 all-gather
    cudagraph_mode       = CUDAGraphMode.PIECEWISE
    batch_desc           = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False)
    ubatch_slices_padded = None      # 无 DBO

    # 全局 forward_context 设置完毕
    # 模型内部每一层 attention 调用时：
    #   metadata = get_forward_context().attn_metadata["model.layers.N.self_attn"]
    #   → 取到对应层的 FlashAttentionMetadata，无需函数参数传递

***

### Step 3 — KV Connector 并行传输（可选）

**目的：** 若启用了 KV cache transfer（如 Disaggregated Prefill/Decode，使用 Nixl/Mooncake 等 KV connector），在模型 forward 执行期间同步发起后台 KV cache 传输，让网络传输与 GPU 计算重叠，结束时等待传输完成并收集结果。

**代码（L3219，配合 L3217 的 `with` 块）：**

```python
self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
```

`maybe_get_kv_connector_output` 内部（`kv_connector_model_runner_mixin.py` L98–136）：

```python
# 若无 KV transfer group，返回 nullcontext()（无操作）
return KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output) \
    if has_kv_transfer_group() else nullcontext()

# 有 KV transfer group 时：
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())  # 后台异步传输 KV cache
try:
    yield output   # 与 _model_forward 的 GPU 计算并行执行
finally:
    kv_connector.wait_for_save()   # 等待 KV 传输完成
    output.finished_sending, output.finished_recving = kv_connector.get_finished(...)
```

**示例数据变化：**

    has_kv_transfer_group() = False   # 单机推理，无 KV transfer

    → 返回 nullcontext()，kv_connector_output = None
    → 跳过 KV 传输，不影响 forward 执行

***

### Step 4 — 执行模型前向（`_model_forward`）

**目的：** 在已设置好的 forward context 中，调用模型的 `__call__`，执行完整的 Transformer 前向计算（embedding → attention → FFN × N 层）。在 PIECEWISE CUDA Graph 模式下，attention 层的 kernel 会复用预捕获的图，FFN 部分 eager 执行。

**代码（L2824–2854，在 L3221 处被调用）：**

```python
# _model_forward 定义（薄封装，可被子类覆盖）：
def _model_forward(self, input_ids, positions, intermediate_tensors,
                   inputs_embeds, **model_kwargs) -> Any:
    return self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )

# execute_model 中的调用（L3221）：
model_output = self._model_forward(
    input_ids=input_ids,               # tensor([p0,p1,p2,p3, t100, t50,t51,t52])
    positions=positions,               # tensor([0,1,2,3, 100, 50,51,52])
    intermediate_tensors=intermediate_tensors,  # None
    inputs_embeds=inputs_embeds,       # None
    **model_kwargs,                    # {}
)
```

**示例数据变化：**

    # 输入：
    input_ids   = tensor([p0,p1,p2,p3, t100, t50,t51,t52])   # shape [8]
    positions   = tensor([0, 1, 2, 3,  100,  50, 51, 52])    # shape [8]
    inputs_embeds = None
    intermediate_tensors = None   # 从 embedding 层开始

    # 模型内部执行流程（LLaMA-7B 为例，32 层）：
    # 1. embedding：input_ids → hidden [8, 4096]
    # 2. layer 0~31：
    #    - attention：读取 forward_context.attn_metadata["layer.N.self_attn"]
    #                 → FlashAttention(q,k,v, block_table, slot_mapping, ...)
    #                 → PIECEWISE 模式下 attention kernel 复用预捕获图
    #    - FFN：eager 执行（PIECEWISE 模式）
    # 3. RMSNorm

    # 输出：
    model_output = hidden_states   # shape [8, 4096]，对应 8 个 token 的最终 hidden state

**CUDA Graph 三种模式对比：**

| 模式          | attention kernel | FFN kernel | 说明                        |
| ----------- | ---------------- | ---------- | ------------------------- |
| `FULL`      | 重放预捕获图           | 重放预捕获图     | 纯 decode，batch shape 固定   |
| `PIECEWISE` | 重放预捕获图           | eager 执行   | 混合 prefill+decode（本例）     |
| `NONE`      | eager 执行         | eager 执行   | 动态 shape，cascade attn 等场景 |

***

### Step 5 — 解包模型输出

**目的：** 从 `model_output` 中取出 `hidden_states`（标准情况）或 `(hidden_states, aux_hidden_states)`（EAGLE 3 投机解码），为后续计算 logits 做准备。

**代码（L3229–3236）：**

```python
if self.use_aux_hidden_state_outputs:
    # True when EAGLE 3 is used.
    # EAGLE 3 drafter 需要中间层的辅助 hidden states
    hidden_states, aux_hidden_states = model_output
else:
    # Common case.
    hidden_states = model_output
    aux_hidden_states = None
```

**示例数据变化：**

    use_aux_hidden_state_outputs = False   # 非 EAGLE 3（本例使用标准推理）

    hidden_states    = model_output        # shape [8, 4096]
    aux_hidden_states = None

***

### Step 6 — Pipeline Parallel 处理与计算 Logits

**目的：** 根据 PP 位置和 `broadcast_pp_output` 配置，决定是直接计算 logits（末节点）还是将中间激活发往下一 PP 节点。计算 logits 时只取 `logits_indices` 指定的行（即每个请求的最后一个 token），而非全部 8 个 token。

**代码（L3238–3285）：**

```python
if not self.broadcast_pp_output:
    # Common case（不广播 logits）
    if not get_pp_group().is_last_rank:
        # 非末节点：返回中间激活给下一 PP 节点
        assert isinstance(hidden_states, IntermediateTensors)
        hidden_states.kv_connector_output = kv_connector_output
        self.kv_connector_output = kv_connector_output
        return hidden_states   # 函数提前返回

    if self.is_pooling_model:
        # Pooling 模型（embedding 任务）：走 pool 路径，不计算 logits
        output = self._pool(hidden_states, ...)
        return output

    # 末节点 + 生成模型（常规路径）：
    sample_hidden_states = hidden_states[logits_indices]  # 只取采样位置的 hidden state
    logits = self.model.compute_logits(sample_hidden_states)  # LM head 线性变换

else:
    # Rare case（broadcast_pp_output=True）：
    # 末节点计算 logits，通过 all-gather 广播给所有 PP 节点
    sample_hidden_states = hidden_states[logits_indices]
    if not get_pp_group().is_last_rank:
        get_pp_group().send_tensor_dict(hidden_states.tensors, ...)
        logits = None
    else:
        logits = self.model.compute_logits(sample_hidden_states)
    broadcasted = get_pp_group().broadcast_tensor_dict({"logits": logits}, src=last_rank)
    logits = broadcasted["logits"]
```

**示例数据变化（常规单节点路径）：**

    broadcast_pp_output = False   # 常规路径
    is_last_rank        = True    # 单节点，既是首节点也是末节点
    is_pooling_model    = False   # 生成模型，非 embedding

    # logits_indices 来自 _prepare_inputs Step 10（spec decode 场景）：
    logits_indices = tensor([3, 4, 5, 6, 7])
    # 含义：
    #   [3]    → req-A 最后 1 个 token（prefill 末 token，位置 3）
    #   [4]    → req-B 的 decode token（位置 4）
    #   [5,6]  → req-C 的 2 个 draft token（位置 5,6）
    #   [7]    → req-C 的 bonus token（位置 7）

    # 从 hidden_states [8, 4096] 中取出 5 行：
    sample_hidden_states = hidden_states[[3, 4, 5, 6, 7]]
    # shape [5, 4096]

    # 通过 LM head（线性层）计算 logits：
    logits = self.model.compute_logits(sample_hidden_states)
    # shape [5, vocab_size]，例如 [5, 32000]（LLaMA vocab）

    # 各行含义：
    # logits[0] → req-A 末 token 的 logits（被 discard_mask 丢弃，不采样）
    # logits[1] → req-B decode token 的 logits（直接采样下一个 token）
    # logits[2] → req-C draft token[51] 的 logits（rejection sampler 验证）
    # logits[3] → req-C draft token[52] 的 logits（rejection sampler 验证）
    # logits[4] → req-C bonus token 的 logits（draft 全被拒绝时的回退）

**无 spec decode 时的对比（req-C 为纯 decode）：**

    logits_indices = tensor([3, 4, 7])   # 每请求只取最后 1 个 token
    # query_start_loc[1:] - 1 = [4-1, 5-1, 8-1] = [3, 4, 7]

    sample_hidden_states = hidden_states[[3, 4, 7]]   # shape [3, 4096]
    logits = self.model.compute_logits(sample_hidden_states)  # shape [3, 32000]
    # logits[0] → req-A（被 discard_mask 丢弃）
    # logits[1] → req-B 采样
    # logits[2] → req-C 采样

***

### Step 7 — 保存执行状态并返回

**目的：** 将本次 forward 的所有中间产物（logits、hidden\_states、spec decode 元数据等）打包为 `ExecuteModelState`，供后续 `sample_tokens()` 消费，实现异步调度模式下的解耦。

**代码（L3287–3299）：**

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,                        # shape [num_sampled_tokens, vocab_size]
    spec_decode_metadata,          # SpecDecodeMetadata 或 None
    spec_decode_common_attn_metadata,  # drafter 用的 attention metadata
    hidden_states,                 # shape [num_tokens, hidden_dim]，全部 token
    sample_hidden_states,          # shape [num_sampled_tokens, hidden_dim]，采样位置
    aux_hidden_states,             # EAGLE 3 辅助层，或 None
    ec_connector_output,           # EC KV transfer 结果，或 None
    cudagraph_stats,               # CUDA Graph padding 统计，或 None
)
self.kv_connector_output = kv_connector_output
return None   # 异步调度：execute_model 返回 None，等待 sample_tokens 调用
```

**示例最终输出（完整数据链路）：**

    # Step 1 → calculate_kv_scales=False，cudagraph_mode 保持 PIECEWISE
    # Step 2 → forward_context 注入 attn_metadata，模型各层可通过 get_forward_context() 读取
    # Step 3 → 无 KV connector，kv_connector_output=None
    # Step 4 → model_output = hidden_states，shape [8, 4096]
    # Step 5 → hidden_states=[8,4096], aux_hidden_states=None
    # Step 6 → sample_hidden_states=[5,4096], logits=[5,32000]

    execute_model_state = ExecuteModelState(
        logits                = tensor([5, 32000]),  # [req-A末, req-B, req-C的draft×2, req-C bonus]
        spec_decode_metadata  = SpecDecodeMetadata(logits_indices=tensor([3,4,5,6,7]), ...),
        hidden_states         = tensor([8, 4096]),   # 所有 8 个 token 的 hidden state
        sample_hidden_states  = tensor([5, 4096]),   # 采样位置的 hidden state
        aux_hidden_states     = None,
        ec_connector_output   = None,
        cudagraph_stats       = None,
    )

    return None   # sample_tokens() 将读取 execute_model_state 进行采样

***

## 关键设计要点

| 设计点                          | 具体做法                                                      | 收益                                                        |
| ---------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| **全局 forward context**       | `set_forward_context` 将 attn\_metadata 注入全局，而非层层传参        | 避免模型每层函数签名携带大量 metadata 参数，解耦推理状态                         |
| **PIECEWISE CUDA Graph**     | attention kernel 重放捕获图，FFN eager 执行                       | prefill/decode 混合 batch 也能部分利用 CUDA Graph，减少 kernel 启动开销  |
| **KV transfer 与 compute 重叠** | `maybe_get_kv_connector_output` 在 forward 期间后台传输 KV cache | Disaggregated Prefill/Decode 场景下网络传输被 GPU 计算掩盖            |
| **只取采样位置的 logits**           | `hidden_states[logits_indices]` 先切片再过 LM head             | LM head 是大矩阵乘法，只算必要的行可节省大量计算（从 8 行减到 5 行）                 |
| **`_model_forward` 薄封装**     | 仅封装 `self.model(...)` 调用，可被子类覆盖                           | 子类（如 speculative decoding 的 drafter runner）可替换 forward 实现 |
| **ExecuteModelState 解耦**     | forward 结果存入 `execute_model_state`，函数返回 `None`            | 支持异步调度：CPU 可立即开始下一轮调度，GPU 继续当前轮采样，两者流水执行                  |
| **KV scales 首次推理降级**         | `calculate_kv_scales=True` 时强制 `NONE` 模式，完成后关闭标志          | FP8 量化首次需要动态统计，之后 scale 固定，可恢复 CUDA Graph                 |

