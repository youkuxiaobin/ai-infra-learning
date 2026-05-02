# `_determine_batch_execution_and_padding` 函数详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py` L2877–L2996

***

## 函数签名与返回值

```python
def _determine_batch_execution_and_padding(
    self,
    num_tokens: int,                          # 本次 batch 未 padding 的总 token 数
    num_reqs: int,                            # 请求数量
    num_scheduled_tokens_np: np.ndarray,      # 每个请求调度的 token 数
    max_num_scheduled_tokens: int,            # 所有请求中最大的调度 token 数
    use_cascade_attn: bool,                   # 是否启用 cascade attention
    allow_microbatching: bool = True,
    force_eager: bool = False,                # 强制 eager 模式，跳过 CUDA Graph
    force_uniform_decode: bool | None = None,
    force_has_lora: bool | None = None,
    num_encoder_reqs: int = 0,
) -> tuple[
    CUDAGraphMode,          # NONE / PIECEWISE / FULL
    BatchDescriptor,        # 描述 padding 后 batch 的结构
    bool,                   # should_ubatch：是否需要拆分为 microbatch（DP 场景）
    torch.Tensor | None,    # num_tokens_across_dp：各 DP rank 的 token 数
    CUDAGraphStat | None,   # 可观测性统计
]:
```

**核心目标：** 根据 batch 的状态（token 数、是否 decode-only、是否有 LoRA 等）决定本次前向用哪种执行模式（CUDA Graph / eager），以及需要 padding 到多少 token。

***

## 关键数据结构

```python
# CUDAGraphMode（vllm/config/compilation.py L52）
class CUDAGraphMode(enum.Enum):
    NONE      = 0   # 不使用 CUDA Graph，完全 eager
    PIECEWISE = 1   # 部分图化（Attention 层复用图，FFN eager）
    FULL      = 2   # 全图化（整个 forward pass 重放预捕获图）

# BatchDescriptor（vllm/forward_context.py L28）
class BatchDescriptor(NamedTuple):
    num_tokens: int           # padding 后的 token 数
    num_reqs: int | None      # padding 后的请求数（PIECEWISE 模式下可为 None）
    uniform: bool = False     # 是否所有请求 token 数相同（纯 decode 场景）
    has_lora: bool = False    # 是否有活跃的 LoRA adapter
```

***

## 示例场景

与 `_prepare_inputs` 文档使用**完全相同的 batch**（见 `prepare_inputs_flow.md`），`max_model_len = 2048`：

| req_idx | req_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明 |
| --- | --- | --- | --- | --- |
| 0 | req-A | 0 | 4 | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1 | req-B | 100 | 1 | decode（无 spec），送入上一步采样的 1 个 token |
| 2 | req-C | 50 | 3 | decode（有 spec），1 个真实 token + 2 个 draft token |

`_prepare_inputs` 执行完毕后，`execute_model` 计算出以下值并传入本函数（L3100–3138）：

```python
num_reqs                 = 3
num_tokens_unpadded      = scheduler_output.total_num_scheduled_tokens = 8
num_scheduled_tokens_np  = np.array([4, 1, 3], dtype=np.int32)
max_num_scheduled_tokens = int(num_scheduled_tokens_np.max()) = 4
use_cascade_attn         = (cascade_attn_prefix_lens is not None) = False
num_encoder_reqs         = len(scheduler_output.scheduled_encoder_inputs) = 0
```

其他环境假设：

| 参数                     | 值                       | 说明                |
| ---------------------- | ----------------------- | ----------------- |
| `force_eager`          | `False`                 | 允许 CUDA Graph     |
| `data_parallel_size`   | 1                       | 单机单卡，无 DP         |
| `tensor_parallel_size` | 1                       | 无 TP，未启用 SP       |
| 预捕获 CUDA Graph 尺寸      | `{1, 2, 4, 8, 16, ...}` | PIECEWISE 模式下常见配置 |

***

## 各步骤详解

### Step 1 — 判断是否为 Uniform Decode

**目的：** 检测 batch 是否为"所有请求均为 decode 且每个请求 token 数相同"的均匀 decode 场景。这是 FULL CUDA Graph 最理想的条件，因为 batch shape 完全固定。

**代码（L2898–2904）：**

```python
uniform_decode = self._is_uniform_decode(
    max_num_scheduled_tokens=max_num_scheduled_tokens,
    uniform_decode_query_len=self.uniform_decode_query_len,  # 通常为 1
    num_tokens=num_tokens,
    num_reqs=num_reqs,
    force_uniform_decode=force_uniform_decode,
)
```

`_is_uniform_decode` 内部（L2857–2875）：

```python
return (
    (max_num_scheduled_tokens == uniform_decode_query_len)  # 所有请求只调度1个token
    and (num_tokens == max_num_scheduled_tokens * num_reqs)  # 总数 = 每请求数 * 请求数（无例外）
) if force_uniform_decode is None else force_uniform_decode
```

**示例数据变化：**

    # 来自示例场景：
    num_scheduled_tokens_np  = [4, 1, 3]
    max_num_scheduled_tokens = 4       # req-A 是 chunked prefill，调度了 4 个 token
    num_tokens               = 8       # 总计 4+1+3
    num_reqs                 = 3
    uniform_decode_query_len = 1       # 标准 decode 每请求 1 个 token

    条件1: max_num_scheduled_tokens == uniform_decode_query_len
           → 4 == 1 → False           # req-A 调度了 4 个，不是纯 decode
    条件2: 不需要再判断（短路）

    uniform_decode = False

> **对比（纯 decode batch）：** 若三个请求全是 decode（`num_scheduled_tokens=[1,1,1]`）：
>
>     max_num_scheduled_tokens = 1
>     num_tokens = 3 = 1 * 3
>     条件1: 1 == 1 → True
>     条件2: 3 == 1 * 3 → True
>     uniform_decode = True  → 满足 FULL CUDA Graph 条件

***

### Step 2 — 检测 Encoder Output 和 LoRA 状态

**目的：** encoder-decoder 模型在有编码器输出时无法使用 FULL CUDA Graph（动态输入）；LoRA 的存在影响图的 key，需要区分有/无 LoRA 的图。

**代码（L2905–2915）：**

```python
# Encoder-decoder 模型只有 decoder_step > 0 时支持 CG（无 enc_output）
has_encoder_output = (
    self.model_config.is_encoder_decoder and num_encoder_reqs > 0
)

has_lora = (
    len(self.input_batch.lora_id_to_lora_request) > 0
    if force_has_lora is None
    else force_has_lora
)
```

**示例数据变化：**

    # 来自示例场景：
    model_config.is_encoder_decoder = False  # 纯 decoder 模型（如 LLaMA）
    num_encoder_reqs                 = 0     # scheduler_output.scheduled_encoder_inputs 为空

    has_encoder_output = False and (0 > 0) = False
    # → 不会因 encoder output 而禁用 FULL 图

    input_batch.lora_id_to_lora_request = {}  # 三个请求均未使用 LoRA
    has_lora = len({}) > 0 = False
    # → CUDA Graph key 中不需要区分 lora/非lora 版本

***

### Step 3 — SP Padding（序列并行对齐）

**目的：** 若启用 Sequence Parallelism（SP），token 数必须是 `tensor_parallel_size` 的整数倍，否则 all-gather 操作无法整除分配。

**代码（L2917）：**

```python
num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
```

`_pad_for_sequence_parallelism` 内部（L2519–2525）：

```python
tp_size = self.vllm_config.parallel_config.tensor_parallel_size
if self.compilation_config.pass_config.enable_sp and tp_size > 1:
    return round_up(num_scheduled_tokens, tp_size)  # 向上取整到 tp_size 的倍数
return num_scheduled_tokens
```

**示例数据变化：**

    # 来自示例场景：
    num_tokens = 8    # _prepare_inputs 处理的总 token 数（4+1+3）
    tp_size    = 1    # 单卡，无 TP
    enable_sp  = False

    num_tokens_padded = 8  # round_up 不触发，直接返回 8

> **对比（启用 SP）：** 若 `tp_size=4`，`num_tokens=8`（已是倍数）→ 不变；\
> 若 batch 只有 6 个 token → `round_up(6, 4) = 8`，padding 2 个空位。

***

### Step 4 — CUDA Graph Dispatcher 决策

**目的：** 调用 `cudagraph_dispatcher.dispatch()` 查表，根据 `(num_tokens, uniform_decode, has_lora)` 找出最匹配的预捕获图，确定执行模式和 padding 后的 `BatchDescriptor`。若 `force_eager=True` 则直接返回 `NONE` 模式。

**代码（L2918–2931）：**

```python
# 定义 dispatch lambda，force_eager 时直接短路为 NONE
dispatch_cudagraph = (
    lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
        num_tokens=num_tokens,
        has_lora=has_lora,
        uniform_decode=uniform_decode,
        disable_full=disable_full,
    )
    if not force_eager
    else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
)

# 首次 dispatch：cascade_attn 或 encoder_output 时禁用 FULL 图
cudagraph_mode, batch_descriptor = dispatch_cudagraph(
    num_tokens_padded,
    disable_full=use_cascade_attn or has_encoder_output,  # True 时只考虑 PIECEWISE
)
num_tokens_padded = batch_descriptor.num_tokens
```

`cudagraph_dispatcher.dispatch` 内部查找逻辑（`vllm/v1/cudagraph_dispatcher.py` L156–183）：

```python
# 1. 超出最大捕获 size → NONE
if num_tokens > max_cudagraph_capture_size:
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

# 2. 构造 padded batch descriptor（找 >= num_tokens 的最小预捕获尺寸）
batch_desc = self._create_padded_batch_descriptor(num_tokens, uniform_decode, has_lora)
relaxed_batch_desc = batch_desc.relax_for_mixed_batch_cudagraphs()

# 3. 优先 FULL 图
if not disable_full:
    if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
        return CUDAGraphMode.FULL, batch_desc
    if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
        return CUDAGraphMode.FULL, relaxed_batch_desc

# 4. 其次 PIECEWISE 图
if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
    return CUDAGraphMode.PIECEWISE, relaxed_batch_desc

# 5. 最终回退 NONE
return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

**示例数据变化：**

    # 来自前面各 Step：
    num_tokens_padded = 8      # Step 3 SP padding 后不变
    uniform_decode    = False  # Step 1：req-A 是 prefill，不是均匀 decode
    has_lora          = False  # Step 2：无 LoRA
    disable_full      = use_cascade_attn or has_encoder_output
                      = False  or False
                      = False  # 不禁用 FULL，允许查找 FULL 图

    # --- dispatcher 内部查找 ---

    # 1. num_tokens=8 <= max_cudagraph_capture_size（假设 256）→ 不回退

    # 2. 构造 batch_desc：
    #    uniform=False（含 prefill），has_lora=False
    #    找 >= 8 的最小预捕获尺寸 → 8 本身存在
    batch_desc = BatchDescriptor(num_tokens=8, num_reqs=3, uniform=False, has_lora=False)

    # 3. 查找 FULL 图：
    #    FULL 图要求 uniform=True（所有请求 token 数相同）
    #    batch_desc.uniform=False → FULL 图 key 不匹配，跳过

    # 4. 查找 PIECEWISE 图（relaxed_batch_desc，忽略 num_reqs）：
    #    relaxed = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False, has_lora=False)
    #    PIECEWISE 图预捕获了 size=8 → 命中

    cudagraph_mode    = CUDAGraphMode.PIECEWISE
    batch_descriptor  = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False, has_lora=False)
    num_tokens_padded = 8   # batch_descriptor.num_tokens，无额外 padding

> **对比（纯 decode batch）：** 若 `uniform_decode=True`，`num_reqs=3`，`num_tokens=3`：
>
>     batch_desc = BatchDescriptor(num_tokens=3, num_reqs=3, uniform=True, has_lora=False)
>     → FULL 图 key 存在（预捕获了 3-req uniform decode）→ 命中 FULL
>     cudagraph_mode = CUDAGraphMode.FULL
>     batch_descriptor = BatchDescriptor(num_tokens=3, num_reqs=3, uniform=True)

***

### Step 5 — Data Parallel 跨 Rank 协调（可选）

**目的：** 多 DP rank 并行时，各 rank 的 batch 大小可能不同。需要通过 all-reduce 类操作协商出统一的 padding 目标，确保所有 rank 使用相同的图尺寸，否则 NCCL 通信会因 shape 不一致而报错。同时决定是否启用 microbatching（ubatch）。

**代码（L2945–2979）：**

```python
should_ubatch, num_tokens_across_dp = False, None
if self.vllm_config.parallel_config.data_parallel_size > 1:
    # eager 模式下禁用 DP padding，避免 prefill 过度 padding
    allow_dp_padding = (
        self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    )

    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens,          # 本 rank 真实 token 数
            parallel_config=self.parallel_config,
            allow_microbatching=allow_microbatching,
            allow_dp_padding=allow_dp_padding,
            num_tokens_padded=num_tokens_padded,     # 本 rank padding 后 token 数
            uniform_decode=uniform_decode,
            num_scheduled_tokens_per_request=num_scheduled_tokens_np,
            cudagraph_mode=cudagraph_mode.value,
        )
    )

    # 用跨 rank 协商后的 token 数重新 dispatch
    if num_tokens_across_dp is not None:
        dp_rank = self.parallel_config.data_parallel_rank
        num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
        )
        assert batch_descriptor.num_tokens == num_tokens_padded
```

**示例数据变化：**

    # 来自示例场景：
    data_parallel_size = 1  # 单机单卡，条件 data_parallel_size > 1 不成立

    直接跳过整个 if 块，保持初始值：
    should_ubatch        = False  # 不拆 microbatch
    num_tokens_across_dp = None   # 无跨 rank 协调结果
    cudagraph_mode       = CUDAGraphMode.PIECEWISE  # 维持 Step 4 的结论不变
    batch_descriptor     = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False, has_lora=False)

> **DP 场景示意（补充）：** 若 `dp_size=2`，rank0 有 8 token，rank1 有 12 token：
>
>     coordinate_batch_across_dp 执行 all-gather：
>     rank0 上报 8，rank1 上报 12
>     all-gather 后两 rank 均看到 [8, 12]
>     allow_dp_padding=True → 统一 padding 到 max=12
>     num_tokens_across_dp = tensor([12, 12])
>
>     rank0 重新 dispatch(12)：找到预捕获尺寸 16 → PIECEWISE
>     rank1 重新 dispatch(12)：同上
>     → 两个 rank 使用相同图尺寸，NCCL 通信 shape 一致

***

### Step 6 — 收集可观测性统计（可选）

**目的：** 若启用了 `cudagraph_metrics` 观测，记录 padding 前后 token 数差值，用于监控 padding 浪费率。

**代码（L2981–2988）：**

```python
cudagraph_stats = None
if self.vllm_config.observability_config.cudagraph_metrics:
    cudagraph_stats = CUDAGraphStat(
        num_unpadded_tokens=num_tokens,                        # 原始 token 数
        num_padded_tokens=batch_descriptor.num_tokens,         # padding 后 token 数
        num_paddings=batch_descriptor.num_tokens - num_tokens, # 浪费的 padding 数
        runtime_mode=str(cudagraph_mode),                      # 执行模式字符串
    )
```

**示例数据变化：**

    observability_config.cudagraph_metrics = False  # 默认关闭，跳过

    cudagraph_stats = None

> **开启时示例（基于本场景数据）：**
>
>     # num_tokens（原始）= 8，batch_descriptor.num_tokens = 8
>     cudagraph_stats = CUDAGraphStat(
>      num_unpadded_tokens = 8,
>      num_padded_tokens   = 8,
>      num_paddings        = 8 - 8 = 0,   # 恰好命中 size=8 的预捕获图，零浪费
>      runtime_mode        = "CUDAGraphMode.PIECEWISE",
>     )
>     # 若 num_tokens=6 而预捕获最小尺寸为 8：
>     #   num_paddings = 8 - 6 = 2，说明浪费了 2 个 token 的计算

***

### Step 7 — 返回结果

**代码（L2990–2996）：**

```python
return (
    cudagraph_mode,       # CUDAGraphMode.PIECEWISE
    batch_descriptor,     # BatchDescriptor(num_tokens=8, ...)
    should_ubatch,        # False
    num_tokens_across_dp, # None
    cudagraph_stats,      # None
)
```

**示例最终输出（完整数据链路）：**

    # 输入（来自 _prepare_inputs 同一 batch）：
    #   req-A: prefill 4 token，req-B: decode 1 token，req-C: spec decode 3 token
    #   num_tokens=8, num_reqs=3, num_scheduled_tokens_np=[4,1,3]

    # Step 1 → uniform_decode    = False  （含 prefill，非均匀 decode）
    # Step 2 → has_encoder_output= False，has_lora = False
    # Step 3 → num_tokens_padded = 8     （无 SP，不变）
    # Step 4 → cudagraph_mode    = PIECEWISE，batch_descriptor.num_tokens = 8
    # Step 5 → should_ubatch     = False，num_tokens_across_dp = None（单卡跳过）
    # Step 6 → cudagraph_stats   = None  （metrics 关闭）

    返回值：
    cudagraph_mode   = CUDAGraphMode.PIECEWISE
    batch_descriptor = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False, has_lora=False)
    should_ubatch        = False
    num_tokens_across_dp = None
    cudagraph_stats      = None

调用方 `execute_model`（L3149–3152）使用返回值构建后续元数据：

```python
num_tokens_padded = batch_desc.num_tokens
# = 8，_build_attention_metadata 按此 padding attention 张量

num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
# = None → 取 num_reqs = 3
# PIECEWISE 模式不固定 num_reqs，attention kernel 按实际 3 个请求处理
```

***

## 关键设计要点

| 设计点                                       | 具体做法                                        | 收益                                          |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| **分级图模式**                                 | FULL > PIECEWISE > NONE，按条件降级               | 尽可能复用预捕获图，减少 kernel 启动开销                    |
| **uniform decode 优先 FULL**                | `max_token==1 && total==reqs` 才允许 FULL      | FULL 图要求 batch shape 完全固定，纯 decode 正好满足     |
| **cascade attn / encoder output 禁用 FULL** | `disable_full=True`                         | 这两种场景输入动态，FULL 图无法重放                        |
| **DP 两阶段 dispatch**                       | 先本地 dispatch，再 all-gather 协商，重新 dispatch    | 确保所有 DP rank 使用相同图尺寸，NCCL 不报 shape mismatch |
| **eager 模式禁用 DP padding**                 | `allow_dp_padding = cudagraph_mode != NONE` | prefill 场景 token 数变化大，padding 到最大值浪费严重      |
| **SP padding 前置**                         | dispatch 前先对齐 TP size                       | 保证 SP 的 all-gather 整除，避免后续断言失败              |
| **lambda 封装 dispatch**                    | `dispatch_cudagraph` 复用于本地和 DP 协商后的两次调用     | 避免重复写 dispatch 逻辑，DP 重新 dispatch 时直接复用      |

