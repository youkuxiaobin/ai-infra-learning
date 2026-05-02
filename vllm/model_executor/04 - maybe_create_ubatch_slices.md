# `maybe_create_ubatch_slices` 函数详解

> 源文件：`vllm/v1/worker/ubatch_utils.py` L61–L112\
> 调用位置：`vllm/v1/worker/gpu_model_runner.py` L3153–L3159

***

## 函数签名与返回值

```python
def maybe_create_ubatch_slices(
    should_ubatch: bool,                      # 是否需要拆分为 microbatch（来自 DP 协调）
    num_scheduled_tokens: np.ndarray,         # 每个请求本次调度的 token 数，如 [4, 1, 3]
    num_tokens_padded: int,                   # padding 后的总 token 数
    num_reqs_padded: int,                     # padding 后的请求数
    num_ubatches: int,                        # 拆分为几个 microbatch
    split_point: list[int] | int | None = None,  # 自定义拆分点，None 则均匀切分
) -> tuple[
    UBatchSlices | None,   # ubatch_slices：按真实 token 范围切分的 slice 列表
    UBatchSlices | None,   # ubatch_slices_padded：尾部 padding 到 num_tokens_padded 的版本
]:
```

**核心目标：** 在 Data Parallel + microbatching（DBO）场景下，将一个大 batch 按 token 数均匀切成若干 microbatch（`UBatchSlice`），每个 slice 记录它覆盖的请求范围和 token 范围，供后续 attention metadata 和模型前向分批处理。

***

## 关键数据结构

```python
# vllm/v1/worker/ubatch_utils.py L12–L27

@dataclass
class UBatchSlice:
    request_slice: slice   # 该 microbatch 包含哪些请求（req 维度的切片）
    token_slice: slice     # 该 microbatch 包含哪些 token（token 维度的切片）

    def is_empty(self) -> bool:
        return (self.request_slice.start == self.request_slice.stop
                or self.token_slice.start == self.token_slice.stop)

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

UBatchSlices: TypeAlias = list[UBatchSlice]
```

***

## 示例场景

与 `_prepare_inputs` 文档使用**完全相同的 batch**（见 `prepare_inputs_flow.md`）：

| req_idx | req_id | 已计算 token 数 | 本次调度 token 数 | 阶段说明 |
| --- | --- | --- | --- | --- |
| 0 | req-A | 0 | 4 | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1 | req-B | 100 | 1 | decode（无 spec），送入上一步采样的 1 个 token |
| 2 | req-C | 50 | 3 | decode（有 spec），1 个真实 token + 2 个 draft token |

来自上游函数的输入值：

```python
# 来自 _prepare_inputs / _determine_batch_execution_and_padding：
should_ubatch           = True           # DP 协调后决定启用 microbatching
num_scheduled_tokens    = [4, 1, 3]      # np.ndarray，每请求调度 token 数
num_tokens_padded       = 8              # padding 后总 token 数（本例无额外 padding）
num_reqs_padded         = 3             # padding 后请求数
num_ubatches            = 2             # 拆成 2 个 microbatch（DBO 配置）
split_point             = None          # 均匀切分
```

> **注意：** 本函数在 `should_ubatch=False`（单卡场景）时直接返回 `(None, None)`。\
> 本例为演示 DBO microbatching 逻辑，**假设 `should_ubatch=True`**。

***

## 各步骤详解

### Step 1 — 早退：should_ubatch=False 时直接跳过

**目的：** 绝大多数情况下（单卡或 DP 无需 microbatch）不需要拆分，直接返回 `None` 避免任何计算开销。

**代码（L69–70）：**

```python
if not should_ubatch:
    return None, None
```

**示例数据变化：**

    # 单卡场景（_prepare_inputs 原始示例）：
    should_ubatch = False
    → 直接 return (None, None)，函数结束

    # 本文档后续步骤基于 should_ubatch=True 的 DBO 场景继续推导

***

### Step 2 — 计算均匀切分点

**目的：** 若未指定 `split_point`，按 `num_tokens_padded // num_ubatches` 均匀划分，得到每个 microbatch 的 token 数上限；再生成各切分边界的 token 下标列表（不含终点）。

**代码（L72–75）：**

```python
if split_point is None:
    split_point = int(num_tokens_padded) // num_ubatches  # 每个 microbatch 的 token 数

# 生成各切分边界（不含 0 和末尾）
# 例：num_ubatches=2, split_point=4 → token_split_points=[4]
token_split_points = [split_point * i for i in range(1, num_ubatches)]
```

**示例数据变化：**

    num_tokens_padded = 8
    num_ubatches      = 2

    split_point = 8 // 2 = 4   # 每个 microbatch 最多 4 个 token

    token_split_points = [4 * 1] = [4]
    # 含义：在 token 下标 4 处切一刀，分成 [0,4) 和 [4,8) 两段

***

### Step 3 — 计算累积 token 数（cu\_num\_tokens）

**目的：** 将每个请求的 token 数转换为累积和数组，用于后续用二分查找快速定位"某个 token 下标属于哪个请求"。

**代码（L79–80）：**

```python
# cu_num_tokens[0]=0, cu_num_tokens[i]=前i个请求的token总数
cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])
```

**示例数据变化：**

    num_scheduled_tokens = [4, 1, 3]

    cu_num_tokens = np.zeros(4, dtype=np.int32)  # [0, 0, 0, 0]
    np.cumsum([4,1,3], out=cu_num_tokens[1:])

    cu_num_tokens = [0, 4, 5, 8]
    #                ^  ^  ^  ^
    #                |  |  |  所有请求结束（token 8）
    #                |  |  req-B 结束（token 5）
    #                |  req-A 结束（token 4）
    #                起始

含义：

*   token 下标 `[0, 4)` 属于 req-A
*   token 下标 `[4, 5)` 属于 req-B
*   token 下标 `[5, 8)` 属于 req-C

***

### Step 4 — 循环构造每个 microbatch 的 UBatchSlice

**目的：** 对每个切分区间 `[start_token, end_token)`，用二分查找确定该区间覆盖的请求范围，构造 `UBatchSlice(request_slice, token_slice)`。

**代码（L82–104）：**

```python
ubatch_slices = []
start_token = 0

# 将终点追加到切分点列表，便于统一迭代
all_points = token_split_points + [cu_num_tokens[-1]]
# e.g., [4] + [8] = [4, 8]

for end_token in all_points:
    token_slice = slice(start_token, end_token)

    # 找包含 start_token 的请求：
    # searchsorted(cu_num_tokens, start_token, side="right") 返回插入位置
    # -1 后得到 start_token 所在请求的下标
    req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

    # 找第一个起始位置 >= end_token 的请求（即不再属于本 microbatch）
    req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

    req_slice = slice(req_start, req_stop)
    ubatch_slices.append(UBatchSlice(req_slice, token_slice))

    start_token = end_token
```

**示例数据变化（第 1 次循环，end\_token=4）：**

    start_token = 0
    end_token   = 4
    token_slice = slice(0, 4)

    # 确定 req_start：
    # searchsorted([0,4,5,8], 0, side="right") = 1  （0 右边第一个插入位置）
    # req_start = 1 - 1 = 0   → 从 req-A（idx=0）开始

    # 确定 req_stop：
    # searchsorted([0,4,5,8], 4, side="left") = 1   （4 第一次出现的位置）
    # req_stop = 1             → 不包含 req-B（idx=1），只到 req-A

    req_slice = slice(0, 1)   # 仅 req-A

    ubatch_slices[0] = UBatchSlice(
        request_slice = slice(0, 1),   # req-A
        token_slice   = slice(0, 4),   # token [0,4)
    )
    start_token = 4

**示例数据变化（第 2 次循环，end\_token=8）：**

    start_token = 4
    end_token   = 8
    token_slice = slice(4, 8)

    # 确定 req_start：
    # searchsorted([0,4,5,8], 4, side="right") = 2  （4 右边插入位置，即跳过 4）
    # req_start = 2 - 1 = 1   → 从 req-B（idx=1）开始

    # 确定 req_stop：
    # searchsorted([0,4,5,8], 8, side="left") = 3   （8 第一次出现的位置）
    # req_stop = 3             → 包含到 req-C（idx=2），不含 idx=3（不存在）

    req_slice = slice(1, 3)   # req-B 和 req-C

    ubatch_slices[1] = UBatchSlice(
        request_slice = slice(1, 3),   # req-B + req-C
        token_slice   = slice(4, 8),   # token [4,8)
    )

    # 循环结束，ubatch_slices 共 2 个
    ubatch_slices = [
        UBatchSlice(request_slice=slice(0,1), token_slice=slice(0,4)),  # microbatch 0
        UBatchSlice(request_slice=slice(1,3), token_slice=slice(4,8)),  # microbatch 1
    ]

***

### Step 5 — 对最后一个 slice 做 padding（ubatch\_slices\_padded）

**目的：** `ubatch_slices` 是按真实 token 数切分的，但 CUDA Graph 要求最终的 batch shape 等于 `num_tokens_padded`。因此需要将最后一个 slice 的 token/request 范围扩展到 padding 后的边界，生成 `ubatch_slices_padded`，供需要固定 shape 的 kernel（如 FULL CUDA Graph 模式下的 attention）使用。

**代码（L106–108）：**

```python
ubatch_slices_padded = _pad_out_ubatch_slices(
    ubatch_slices, num_tokens_padded, num_reqs_padded
)
```

`_pad_out_ubatch_slices` 内部（L49–58）：

```python
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    # 将最后一个 slice 的终止点扩展到 padding 后的边界
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice   = slice(last_slice.token_slice.start,   num_total_tokens)

    # 前面的 slice 保持不变，只替换最后一个
    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]
```

**示例数据变化：**

    ubatch_slices[-1] = UBatchSlice(request_slice=slice(1,3), token_slice=slice(4,8))
    num_tokens_padded = 8
    num_reqs_padded   = 3

    # 本例真实 token 数 = padding 后 token 数 = 8，且 num_reqs_padded = 3
    # 最后 slice 终点已经是 8 和 3，padding 后无变化

    padded_last_request_slice = slice(1, 3)   # 不变
    padded_last_token_slice   = slice(4, 8)   # 不变

    ubatch_slices_padded = [
        UBatchSlice(request_slice=slice(0,1), token_slice=slice(0,4)),  # microbatch 0（不变）
        UBatchSlice(request_slice=slice(1,3), token_slice=slice(4,8)),  # microbatch 1（padding 后）
    ]

    # 校验（L110）：
    sum(s.num_tokens for s in ubatch_slices_padded) = 4 + 4 = 8 == num_tokens_padded ✓

> **有 padding 时的对比：** 若 `num_tokens_padded=16`（CUDA Graph 扩展），`num_reqs_padded=4`：
>
>     最后 slice 从 slice(4,8) 扩展为 slice(4,16)
>     最后 req slice 从 slice(1,3) 扩展为 slice(1,4)
>
>     ubatch_slices_padded[-1] = UBatchSlice(
>      request_slice = slice(1, 4),   # 含 padding 请求槽
>      token_slice   = slice(4, 16),  # 含 padding token 槽
>     )
>     sum(num_tokens) = 4 + 12 = 16 == num_tokens_padded ✓

***

### Step 6 — 返回结果

**代码（L112）：**

```python
return ubatch_slices, ubatch_slices_padded
```

**示例最终输出：**

    ubatch_slices = [
        UBatchSlice(request_slice=slice(0,1), token_slice=slice(0,4)),
        #           ↑ 仅 req-A（prefill 4 token）
        UBatchSlice(request_slice=slice(1,3), token_slice=slice(4,8)),
        #           ↑ req-B + req-C（decode 1 + spec decode 3 = 4 token）
    ]

    ubatch_slices_padded = [
        UBatchSlice(request_slice=slice(0,1), token_slice=slice(0,4)),   # 同上（无 padding）
        UBatchSlice(request_slice=slice(1,3), token_slice=slice(4,8)),   # 同上（无 padding）
    ]

调用方 `execute_model`（L3167–3171）使用返回值：

```python
pad_attn = cudagraph_mode == CUDAGraphMode.FULL  # PIECEWISE → False

# attention 按未 padding 的 slice 处理（PIECEWISE 支持动态 shape）
ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices
# = ubatch_slices（因为 pad_attn=False）

# 后续 _build_attention_metadata 对每个 microbatch 分别构建 AttnMetadata：
# microbatch 0: req-A，token [0,4)   → prefill attention
# microbatch 1: req-B+req-C，token [4,8) → decode attention
```

***

## 关键设计要点

| 设计点                                 | 具体做法                                                        | 收益                                                              |
| ----------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| **早退优化**                            | `should_ubatch=False` 直接返回 `(None, None)`                   | 单卡/无 DBO 场景零开销                                                  |
| **二分查找定位请求**                        | `np.searchsorted(cu_num_tokens, ...)`                       | O(log N) 定位 token 边界属于哪个请求，避免线性扫描                               |
| **`side="right"` vs `side="left"`** | `req_start` 用 `right`，`req_stop` 用 `left`                   | 正确处理 token 边界恰好落在请求边界上的情况（同一 token 不重复计入两个 slice）               |
| **双版本 slice**                       | 同时返回 `ubatch_slices`（真实）和 `ubatch_slices_padded`（padding 后） | FULL CUDA Graph 用 padded 版（shape 固定），PIECEWISE/eager 用真实版（避免浪费） |
| **只 pad 最后一个 slice**                | `_pad_out_ubatch_slices` 只替换最后一个元素                          | 前面的 slice 已对齐切分点，形状确定；只有最后一个需要补到 padding 边界                     |
| **终点追加简化循环**                        | `all_points = token_split_points + [cu_num_tokens[-1]]`     | 不需要对最后一段做特殊处理，统一 for 循环覆盖所有区间                                   |

