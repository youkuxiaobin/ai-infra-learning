# `_update_states` 函数详解

> 源文件：`vllm/v1/worker/gpu_model_runner.py` L874–L1119

***

## 函数签名

```python
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
```

**核心目标：** 将调度器输出同步到 CPU 侧缓存（`self.requests`）和持久化 batch（`input_batch`），使下一步 `_prepare_inputs` 能用最新状态构建 GPU 张量。

***

## 两层状态的含义

| 层级 | 对象 | 类型 | 用途 |
| --- | --- | --- | --- |
| CPU 字典 | `self.requests` | `dict[str, CachedRequestState]` | Python 侧完整请求状态，供逻辑判断使用 |
| 持久化 batch | `self.input_batch` | `InputBatch`（numpy/pinned 数组） | 直接 H2D 拷贝的 CPU buffer，与 GPU 对齐 |

***

## 示例场景

以下所有 Step 均基于同一场景推导，假设 **sync scheduling**、`is_last_rank=True`（非 PP）：

| `req_idx` | `req_id` | 已计算 token 数 | 本次调度 token 数 | 阶段说明 |
| --- | --- | --- | --- | --- |
| 0 | req-A | 0 | 4 | chunked prefill，prompt 共 10 个 token，本次处理前 4 个 |
| 1 | req-B | 100 | 1 | decode（无 spec），送入上一步采样的 1 个 token |
| 2 | req-C | 50 | 3 | decode（有 spec），1 个真实 token + 2 个 draft token |

**调用前 `self.requests`（CPU 字典，req-A 尚未到来）：**

```python
self.requests = {
    "req-B": CachedRequestState(
        prompt_token_ids=[...]*100, num_computed_tokens=99,
        output_token_ids=[42],      prev_num_draft_len=0,
    ),
    "req-C": CachedRequestState(
        prompt_token_ids=[...]*50,  num_computed_tokens=49,
        output_token_ids=[77],      prev_num_draft_len=2,  # 上步提交了 2 个 draft
    ),
}
```

**调用前 `input_batch`（persistent batch，上步结束后）：**

| slot | `req_id` | `num_computed_tokens_cpu` | `num_tokens_no_spec` | `spec_token_ids`   |
| ---- | -------- | ------------------------- | -------------------- | ------------------ |
| 0    | req-B    | 99                        | 100                  | []                 |
| 1    | req-C    | 49                        | 50                   | [draft-1, draft-2] |

```python
input_batch.req_id_to_index = {"req-B": 0, "req-C": 1}
```

**本步 `SchedulerOutput` 关键字段：**

```python
finished_req_ids             = {}
num_scheduled_tokens         = {"req-A": 4, "req-B": 1, "req-C": 3}
scheduled_new_reqs           = [NewRequestData(req_id="req-A",
                                    prompt_token_ids=[10, 20, ..., 100],
                                    num_computed_tokens=0,
                                    block_ids=([4, 5],))]
scheduled_cached_reqs        = CachedRequestData(
    req_ids             = ["req-B", "req-C"],
    num_computed_tokens = [100, 50],   # 上步执行完后 KV 中的 token 数
    num_output_tokens   = [1, 1],
    new_block_ids       = [None, None],
    resumed_req_ids     = {},
)
scheduled_spec_decode_tokens = {"req-C": [200, 201]}
```

***

## 各步骤详解

### Step 1 — 清除已完成请求，释放 Encoder 缓存

**目的：** 把已完成（finished）的请求从 `self.requests` 和 `input_batch` 中删除，并回收其 encoder 多模态特征缓存，防止内存泄漏。

**代码（L884–899）：**

```python
for req_id in scheduler_output.finished_req_ids:
    self.requests.pop(req_id, None)
    self.num_prompt_logprobs.pop(req_id, None)

# NOTE: 若同一 req_id 先 abort 再重提交，finished 与 scheduled 会重叠，
# 此处先清旧，后续走新请求路径。
for req_id in scheduler_output.finished_req_ids:
    self.input_batch.remove_request(req_id)

for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

**示例数据变化：**

`finished_req_ids = {}` → 此步为空操作，状态不变。

若本场景中 req-B 已完成（假设占 slot 0）：

```python
self.requests.pop("req-B")
input_batch.remove_request("req-B")
# input_batch._req_ids[0]      = None
# input_batch.req_id_to_index  删除 "req-B"
# batch_update_builder.removed = [0]   ← 登记空洞槽位
```

***

### Step 2 — 计算并移除未调度请求

**目的：** 把上步在 batch 中、但本步未被调度的请求（被抢占或暂时跳过）从 `input_batch` 移除，保持 persistent batch 只含本步要运行的请求。它们的 `CachedRequestState` 不删，等待未来复用。

**代码（L901–921）：**

```python
scheduled_req_ids  = scheduler_output.num_scheduled_tokens.keys()
cached_req_ids     = self.input_batch.req_id_to_index.keys()
resumed_req_ids    = scheduler_output.scheduled_cached_reqs.resumed_req_ids

# 仍在 batch 中、但本步不调度的（resumed 的也纳入：需先清除再重加）
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)

for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)
```

**示例数据变化：**

```python
scheduled_req_ids   = {"req-A", "req-B", "req-C"}
cached_req_ids      = {"req-B", "req-C"}
resumed_req_ids     = {}

unscheduled_req_ids = {"req-B", "req-C"} - ({"req-A", "req-B", "req-C"} - {})
                    = {}   # req-B 和 req-C 都被调度了，无需移除
```

变体：若 req-B 本步被抢占（不在 `scheduled_req_ids` 中）：

```python
unscheduled_req_ids = {"req-B"}
input_batch.remove_request("req-B")   # slot 0 置空
# self.requests["req-B"] 保留（等待下次调度）
```

***

### Step 3 — 注册新请求的 CachedRequestState

**目的：** 遍历 `scheduled_new_reqs`，为首次出现的请求在 CPU 侧创建 `CachedRequestState`，写入 `self.requests`，并加入 `reqs_to_add` 列表（等待 Step 10 写入 `input_batch`）。

**代码（L923–984）：**

```python
reqs_to_add: list[CachedRequestState] = []

for new_req_data in scheduler_output.scheduled_new_reqs:
    req_id = new_req_data.req_id   # "req-A"

    if req_id in self.requests:    # 流式多轮请求，本例跳过
        req_state = self._update_streaming_request(req_id, new_req_data)
        reqs_to_add.append(req_state)
        continue

    if (sampling_params
            and sampling_params.sampling_type == SamplingType.RANDOM_SEED):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling_params.seed)
    else:
        generator = None

    req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=new_req_data.prompt_token_ids,
        prompt_embeds=new_req_data.prompt_embeds,
        mm_features=new_req_data.mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        generator=generator,
        block_ids=new_req_data.block_ids,
        num_computed_tokens=new_req_data.num_computed_tokens,
        output_token_ids=[],
        lora_request=new_req_data.lora_request,
    )
    self.requests[req_id] = req_state
    reqs_to_add.append(req_state)
    # M-RoPE / XD-RoPE 初始化（仅多模态模型，本例跳过）
```

**示例数据变化：**

```python
# 执行后 self.requests：
{
    "req-B": CachedRequestState(num_computed_tokens=99, output_token_ids=[42], ...),
    "req-C": CachedRequestState(num_computed_tokens=49, output_token_ids=[77],
                                prev_num_draft_len=2, ...),
    "req-A": CachedRequestState(num_computed_tokens=0,  output_token_ids=[],
                                prompt_token_ids=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                block_ids=([4, 5],), prev_num_draft_len=0),
}

reqs_to_add = [req_state_A]   # req-A 待写入 input_batch
```

***

### Step 4 — 获取 async spec decode 有效 token 数

**目的：** 在 async scheduling 模式下，上步 GPU 的 spec verify 结果（接受/拒绝了多少 draft token）在本步才能读取。此处同步 GPU 事件，供 Step 5 校正 `num_computed_tokens`；sync scheduling 时直接返回空列表。

**代码（L986–993）：**

```python
is_last_rank = get_pp_group().is_last_rank

req_data              = scheduler_output.scheduled_cached_reqs
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

valid_sampled_token_count = self._get_valid_sampled_token_count()
# sync scheduling → []；async scheduling → 等待 GPU 事件后返回各请求的有效 token 数
```

**示例数据变化：**

| 模式                                                | `valid_sampled_token_count` | 含义                                                         |
| --------------------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| sync scheduling（本例）                             | `[]`                        | 不使用                                                       |
| async scheduling，req-C 上步 2 draft，GPU 接受 1 个 | `[_, 2]`                    | index 1 对应 req-C 上步 slot，值 = 1 real + 1 accepted draft |

***

### Step 5 — 主循环：async scheduling + spec decode 路径校正

**目的：** 当 async scheduling 与 spec decode 同时启用且上步提交了 draft token 时，调度器发来的 `num_computed_tokens` 包含"假设 draft 全被接受"的估算偏差，此步根据 GPU 验证结果校正。

**代码（L995–1025）：**

```python
for i, req_id in enumerate(req_data.req_ids):   # ["req-B", "req-C"]
    req_state               = self.requests[req_id]
    num_computed_tokens     = req_data.num_computed_tokens[i]
    new_block_ids           = req_data.new_block_ids[i]
    resumed_from_preemption = req_id in req_data.resumed_req_ids
    num_output_tokens       = req_data.num_output_tokens[i]
    req_index               = self.input_batch.req_id_to_index.get(req_id)

    if req_state.prev_num_draft_len and self.use_async_scheduling:
        if req_index is None:
            req_state.prev_num_draft_len = 0   # 上步被抢占，无法校正
        else:
            prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
            num_accepted   = valid_sampled_token_count[prev_req_index] - 1
            num_rejected   = req_state.prev_num_draft_len - num_accepted
            num_computed_tokens -= num_rejected
            req_state.output_token_ids.extend([-1] * num_accepted)
```

**示例数据变化：**

- `i=0, req_id="req-B"`：`prev_num_draft_len=0` → 条件 False，跳过
- `i=1, req_id="req-C"`：`prev_num_draft_len=2`，但 `use_async_scheduling=False` → 跳过
- `num_computed_tokens` 保持 `[100, 50]`，不变

变体（async scheduling，req-C 上步 draft=[d1, d2]，GPU 接受 1 个，拒绝 1 个）：

```python
# 调度器估算 num_computed_tokens = 50 + 1(real) + 2(draft) = 53
prev_req_index = 1
num_accepted   = valid_sampled_token_count[1] - 1   # = 2 - 1 = 1
num_rejected   = 2 - 1                              # = 1
num_computed_tokens = 53 - 1                        # = 52（校正后）
req_state.output_token_ids                          # [77] → [77, -1]（-1 是占位符）
```

***

### Step 6 — 更新 `num_computed_tokens` 与 `output_token_ids`

**目的：** 将（已校正的）`num_computed_tokens` 写入 `req_state`；根据是否为 PP 最后一个 rank，走不同路径同步 `output_token_ids`；若发生 KV 缓存同步失败导致 token 被截断，也在此处修正。

**代码（L1027–1060）：**

```python
req_state.num_computed_tokens = num_computed_tokens   # L1028

if not is_last_rank:
    if not req_data.new_token_ids:
        new_token_ids: list[int] = []   # async PP：token 通过 GPU broadcast 传播
    else:
        new_token_ids  = req_data.new_token_ids[i]   # 同步 PP：末 rank 回传
        num_new_tokens = (
            num_computed_tokens + len(new_token_ids) - req_state.num_tokens
        )
        if num_new_tokens == 1:
            req_state.output_token_ids.append(new_token_ids[-1])
        elif num_new_tokens > 0:
            req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

elif num_output_tokens < len(req_state.output_token_ids):
    # KV sync 失败导致部分 output token 被截断
    del req_state.output_token_ids[num_output_tokens:]
    if req_index is not None:
        end_idx = self.input_batch.num_prompt_tokens[req_index] + num_output_tokens
        self.input_batch.num_tokens_no_spec[req_index] = end_idx
```

**示例数据变化（`is_last_rank=True`，无 KV 失败）：**

| 请求  | `num_computed_tokens` 变化 | `output_token_ids` | 说明                                                         |
| ----- | -------------------------- | ------------------ | ------------------------------------------------------------ |
| req-B | 99 → 100                   | `[42]`（不变）     | `is_last_rank=True`，跳过 PP 路径，`num_output_tokens==len` 不截断 |
| req-C | 49 → 50                    | `[77]`（不变）     | 同上                                                         |

变体（PP 首 rank，同步模式，`new_token_ids=[99]`，`num_computed_tokens=101`）：

```python
# req_state.num_tokens = num_prompt_tokens(100) + len(output_token_ids)(1) = 101
num_new_tokens = 101 + 1 - 101   # = 1
req_state.output_token_ids.append(99)   # [42] → [42, 99]
```

***

### Step 7 — 更新 `block_ids`

**目的：** 根据 KV cache 分配结果更新 `req_state.block_ids`。正常运行时追加新块；从抢占恢复时整体替换（旧块已被系统回收）。

**代码（L1062–1073）：**

```python
if not resumed_from_preemption:
    if new_block_ids is not None:
        for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
            block_ids.extend(new_ids)
else:
    assert req_index is None
    assert new_block_ids is not None
    req_state.block_ids = new_block_ids   # 整体替换
```

**示例数据变化：**

- `req-B`：`new_block_ids=None` → 无操作
- `req-C`：`new_block_ids=None` → 无操作

变体（req-C 本步新分配 KV 块，`new_block_ids=([6],)`）：

```python
# 执行前：req_state.block_ids = ([2, 3],)
[2, 3].extend([6])   # → ([2, 3, 6],)
```

变体（请求从抢占恢复，`resumed_from_preemption=True`，`new_block_ids=([10, 11],)`）：

```python
# 执行前：req_state.block_ids = ([0, 1],)（旧块已被系统回收）
req_state.block_ids = ([10, 11],)   # 整体替换
```

***

### Step 8 — 处理不在 persistent batch 中的请求

**目的：** 对于本步调度、但 `req_index is None` 的请求（抢占后恢复，或上步未调度），将其加入 `reqs_to_add`，等 Step 10 统一写入。async scheduling 模式下还需从 `all_token_ids` 恢复完整的 `output_token_ids`。

**代码（L1075–1087）：**

```python
if req_index is None:
    if self.use_async_scheduling and num_output_tokens > 0:
        resumed_token_ids = req_data.all_token_ids[req_id]
        req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

    reqs_to_add.append(req_state)
    continue   # 跳过后续的 persistent batch 就地更新
```

**示例数据变化：**

- `req-B`：`req_index=0` → 不为 None，跳过此分支
- `req-C`：`req_index=1` → 不为 None，跳过此分支
- `reqs_to_add` 仍为 `[req_state_A]`（只有 Step 3 加入的 req-A）

变体（req-C 上步被抢占，本步恢复，async scheduling，`num_output_tokens=3`）：

```python
req_data.all_token_ids["req-C"] = [p1, ..., p50, o1, o2, o3]
req_state.output_token_ids = [o1, o2, o3]   # 恢复 3 个 output token
reqs_to_add.append(req_state_C)
```

***

### Step 9 — 就地更新 persistent batch 中的已有请求

**目的：** 对 `req_index is not None` 的请求，直接修改 `input_batch` 对应行的 CPU 数组，避免重新 `add_request` 的开销；最后写入本步的 spec draft token，并记录 `prev_num_draft_len` 供下步 Step 5 使用。

**代码（L1089–1106）：**

```python
self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens

if new_block_ids is not None:
    self.input_batch.block_table.append_row(new_block_ids, req_index)

if not is_last_rank:
    start_token_index = num_computed_tokens
    end_token_index   = num_computed_tokens + len(new_token_ids)
    self.input_batch.token_ids_cpu[
        req_index, start_token_index:end_token_index
    ] = new_token_ids
    self.input_batch.num_tokens_no_spec[req_index] = end_token_index

self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
```

`update_req_spec_token_ids` 内部（L443–467）：

```python
cur_spec_token_ids = self.spec_token_ids[req_index]
cur_spec_token_ids.clear()                         # 清除上步旧 draft

spec_token_ids = scheduled_spec_tokens.get(req_id, ())
request.prev_num_draft_len = len(spec_token_ids)   # 供下步 Step 5 使用

start_index     = self.num_tokens_no_spec[req_index]
end_token_index = start_index + len(spec_token_ids)
self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
cur_spec_token_ids.extend(spec_token_ids)
```

**示例数据变化：**

req-B（`req_index=0`）：

```python
num_computed_tokens_cpu[0] = 100           # 99 → 100
# new_block_ids=None        → block_table 不变
# is_last_rank=True         → 跳过 token_ids_cpu 写入
# update_req_spec_token_ids：
spec_token_ids[0]              = []        # "req-B" 不在 scheduled_spec_decode_tokens
req_state_B.prev_num_draft_len = 0
```

req-C（`req_index=1`）：

```python
num_computed_tokens_cpu[1] = 50            # 49 → 50
# update_req_spec_token_ids：
# spec_token_ids[1].clear()                  清除旧 [draft-1, draft-2]
start_index     = num_tokens_no_spec[1]   # = 50
end_token_index = 50 + 2                  # = 52
token_ids_cpu[1, 50:52]        = [200, 201]   # 写入新 draft token
spec_token_ids[1]              = [200, 201]
req_state_C.prev_num_draft_len = 2            # 下步 Step 5 据此校正
```

***

### Step 10 — 将 `reqs_to_add` 写入 persistent batch

**目的：** 把 Step 3（新请求）和 Step 8（恢复的请求）收集到的 `reqs_to_add`，逐一写入 `input_batch`。`add_request` 优先填充 `batch_update_builder.removed` 中登记的空洞槽位；无空洞则追加到末尾。

**代码（L1108–1112）：**

```python
for request in reqs_to_add:
    self.input_batch.add_request(request)
    self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)
```

`add_request` 关键逻辑（L278–342）：

```python
def _register_add_request(self, request):
    if (new_req_index := self.batch_update_builder.pop_removed()) is None:
        new_req_index = self.num_reqs   # 无空洞，追加到末尾
    return new_req_index
```

**示例数据变化：**

```python
# reqs_to_add = [req_state_A]
# batch_update_builder.removed = []（Step 1-2 未移除任何请求，无空洞）

# add_request(req_state_A)：
# pop_removed() → None → new_req_index = num_reqs = 2（追加到末尾）
_req_ids[2]                = "req-A"
req_id_to_index            = {"req-B": 0, "req-C": 1, "req-A": 2}
token_ids_cpu[2, 0:10]     = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_prompt_tokens[2]       = 10
num_tokens_no_spec[2]      = 10    # 10 prompt + 0 output
num_computed_tokens_cpu[2] = 0
temperature_cpu[2]         = 0.8
# block_table.add_row(([4, 5],), req_index=2)

# update_req_spec_token_ids(req_state_A)：
# "req-A" 不在 scheduled_spec_decode_tokens
spec_token_ids[2]              = []
req_state_A.prev_num_draft_len = 0
```

***

### Step 11 — 压缩、重排、刷新元数据

**目的：** `condense()` 消除 batch 数组中残余的"空洞"，保证 index 连续；`_may_reorder_batch()` 允许特定 attention backend（如 MLA）对请求排序优化；`refresh_metadata()` 将变更的 sampling 参数同步到 GPU 侧张量。

**代码（L1114–1119）：**

```python
self.input_batch.condense()
self._may_reorder_batch(scheduler_output)
self.input_batch.refresh_metadata()
```

`condense()` 核心逻辑（L626–755）：

```python
def condense(self):
    if not (empty_req_indices := self.batch_update_builder.removed):
        return   # 无空洞，直接返回
    # 将末尾有效请求搬入最小空洞槽位（O(holes) 次 numpy 行复制）
```

`refresh_metadata()` 核心逻辑（L756–772）：

```python
def refresh_metadata(self):
    batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
    for logit_proc in self.logitsprocs.all:
        logit_proc.update_state(batch_update)
    if batch_update:   # batch 有增删 → 重建 SamplingMetadata 并 H2D 拷贝
        self.sampling_metadata = self._make_sampling_metadata()
```

**示例数据变化：**

| 函数                   | 结果                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `condense()`           | `removed=[]` → 直接返回，无数组移动                          |
| `_may_reorder_batch()` | `reorder_batch_threshold=None` → 跳过                        |
| `refresh_metadata()`   | `batch_changed=True`（req-A 被添加）→ 重建 `SamplingMetadata`，`temperature`/`top_p`/`top_k` 等同步到 GPU |

`_update_states` 执行完毕后的最终 `input_batch` 状态：

| slot | `req_id` | `num_computed_tokens_cpu` | `num_tokens_no_spec` | `spec_token_ids` |
| ---- | -------- | ------------------------- | -------------------- | ---------------- |
| 0    | req-B    | 100                       | 100                  | []               |
| 1    | req-C    | 50                        | 50                   | [200, 201]       |
| 2    | req-A    | 0                         | 10                   | []               |

```python
token_ids_cpu[1, 50:52] = [200, 201]          # req-C draft token
token_ids_cpu[2,  0:10] = [10, 20, ..., 100]  # req-A prompt
```

***

## 关键设计要点

| 设计决策                             | 具体做法                                                     | 目的                                                         |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **持久化 batch**                     | 请求在 batch 中原地更新而非每步重建                          | 避免反复分配/初始化大型 CPU 数组（最大 `max_num_reqs × max_model_len`） |
| **两层状态管理**                     | `self.requests`（Python dict）+ `input_batch`（numpy/pinned 数组） | dict 做快速 Python 侧逻辑判断；数组支持直接 H2D 拷贝         |
| **空洞优先填充**                     | `batch_update_builder.removed` 记录空洞，`add_request` 优先填入 | `condense()` 的移动次数最小化；保证 GPU 侧 index 连续        |
| **condense 延迟执行**                | 多次 `remove_request` 后统一调用一次 `condense()`            | 批量合并"空洞填充"的 memcpy，避免每次删除都重排              |
| **spec token 写在真实 token 之后**   | Step 9 末尾单独写入，明确分界 `num_tokens_no_spec`           | verify 后若被拒，只需重置 `spec_token_ids`，不影响前段真实 token |
| **async spec decode 校正时机**       | Step 5 读取 GPU 同步结果校正 `num_computed_tokens`           | 调度器无法预知 spec verify 结果，此处修正才能保证 KV 位置正确 |
| **PP token 回传仅首 rank 处理**      | `if not is_last_rank` 才追加 `output_token_ids`              | 末 rank 已在 `_update_states_after_model_execute` 中更新，避免重复写入 |
| **resumed 请求整体替换 `block_ids`** | `resumed_from_preemption=True` 时不追加而替换                | 抢占期间旧块已被回收，追加会导致 KV 地址错误                 |
| **`refresh_metadata` 按需更新**      | 仅 `batch_changed=True` 时才重建并 H2D 拷贝                  | `SamplingMetadata` 重建涉及多个 GPU tensor 同步，代价高      |
主要是更新了self.input_batch 对象和self.request对象